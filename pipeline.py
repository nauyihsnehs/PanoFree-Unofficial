import argparse
import copy
import json
import os
from datetime import datetime

import cv2
import pipeline_helper as helper

import numpy as np
from PIL import Image


def build_timestamped_run_dir(base_dir):
    base_dir = os.path.abspath(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, "{}_{:02d}".format(timestamp, suffix))
        suffix += 1
    return run_dir


def build_view(base_view, yaw_deg, pitch_deg, fov_deg):
    view = copy.deepcopy(base_view)
    view["yaw_deg"] = yaw_deg
    view["pitch_deg"] = pitch_deg
    view["fov_deg"] = fov_deg
    return view


def build_inpaint_input(warped, missing_mask):
    result = warped.copy()
    result[missing_mask > 0] = 0
    return result


def generate_or_load_initial_view(config, generator):
    source_image_path = config["input"].get("source_image")
    source_size = (config["source_view"]["width"], config["source_view"]["height"])
    if source_image_path:
        return helper.load_source_image(source_image_path, source_size), True
    return helper.generate_initial_view(config, generator=generator), False


def build_record_index(records):
    return {r["name"]: r for r in records}


def build_steps_manifest(view_records):
    manifest = []
    for record in view_records:
        manifest.append(
            {
                "name": record["name"],
                "kind": record["kind"],
                "yaw_deg": record["view"]["yaw_deg"],
                "pitch_deg": record["view"]["pitch_deg"],
                "source_name": record.get("source_name", ""),
                "guidance_name": record.get("guidance_name", ""),
            }
        )
    return manifest


def build_central_schedule(config):
    stride = config["central_band"]["yaw_stride_deg"]
    steps = config["central_band"]["steps_each_direction"]
    schedule = []
    for i in range(1, steps + 1):
        schedule.append({"name": f"x{i}", "yaw_deg": stride * i, "direction": 1, "step_index": i})
        schedule.append({"name": f"x-{i}", "yaw_deg": -stride * i, "direction": -1, "step_index": i})
    return schedule


def compute_initial_risks(config, initial_record):
    initial_center = helper.view_center_on_equirectangular(
        config["source_view"],
        config["output"]["pano_width"],
        config["output"]["pano_height"],
    )
    initial_record["guidance_image"] = initial_record["image"]
    initial_record["guided_input"] = initial_record["image"]
    initial_record["risk_maps"] = helper.compute_view_risk_maps(
        initial_record,
        [initial_record],
        config,
        initial_center,
        config["central_band"],
    )
    return initial_center


def build_risk_masks(base_missing_mask, known_mask, warped_combined_risk, risk_config):
    selected_mask = helper.select_risky_known_pixels(
        warped_combined_risk,
        known_mask,
        risk_config["erase_ratio"],
        risk_config["risk_fallback_threshold"],
    )
    remasked = np.where((base_missing_mask > 0) | (selected_mask > 0), 255, 0).astype(np.uint8)
    smoothed = helper.smooth_remask(
        remasked,
        risk_config["risk_gaussian_kernel"],
        risk_config["risk_gaussian_sigma"],
        risk_config["mask_median_kernel"],
        risk_config["mask_dilate_kernel"],
        risk_config["mask_dilate_iterations"],
    )
    return selected_mask, remasked, smoothed


def build_merge_composite(left_warped, right_warped, left_known_mask, right_known_mask):
    left_valid = left_known_mask > 0
    right_valid = right_known_mask > 0
    composite = np.zeros_like(left_warped)
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    both = left_valid & right_valid
    composite[left_only] = left_warped[left_only]
    composite[right_only] = right_warped[right_only]
    composite[both] = (
            (left_warped[both].astype(np.float32) + right_warped[both].astype(np.float32)) * 0.5
    ).astype(np.uint8)
    missing_mask = np.where(left_valid | right_valid, 0, 255).astype(np.uint8)
    return composite, missing_mask


class InpaintStep:
    """Unified inpainting step executor for central, expansion, and pole targets."""

    def __init__(self, config, generator, initial_center):
        self.config = config
        self.generator = generator
        self.initial_center = initial_center

    def run(self, target_view, composite_bundle, guidance_image, risk_config_key, record_name, record_kind, context_records, extra_fields=None):
        """Execute the shared inpainting workflow."""
        risk_config = self.config[risk_config_key]
        selected_mask, remasked_mask, smoothed_mask = build_risk_masks(
            composite_bundle["base_missing_mask"], composite_bundle["known_mask"],
            composite_bundle["warped_combined_risk"], risk_config)
        inpaint_input = build_inpaint_input(composite_bundle["composite"], smoothed_mask)
        guided_input, output_image = helper.run_guided_inpaint(
            self.config["prompt"], inpaint_input, smoothed_mask, guidance_image,
            self.config, generator=self.generator, strength=risk_config["sdedit_t0"],
            guidance_scale=risk_config.get("guidance_scale"), noise_variance_multiplier=risk_config.get("noise_variance_multiplier"))
        record = {
            "name": record_name, "kind": record_kind, "view": target_view, "image": output_image,
            "guidance_image": guidance_image, "guided_input": guided_input,
            "base_missing_mask": composite_bundle["base_missing_mask"],
            "known_mask": composite_bundle["known_mask"],
            "warped_combined_risk": composite_bundle["warped_combined_risk"],
            "risk_selected_mask": selected_mask, "remasked_mask": remasked_mask,
            "smoothed_mask": smoothed_mask, "missing_mask": smoothed_mask,
        }
        if extra_fields:
            record.update(extra_fields)
        record["risk_maps"] = helper.compute_view_risk_maps(
            record, context_records + [record], self.config, self.initial_center, risk_config)
        return record


def run_central_step(config, step_spec, view_records, generator, initial_center):
    step_index, direction = step_spec["step_index"], step_spec["direction"]
    source_name = "x0" if step_index == 1 else f"{'x' if direction > 0 else 'x-'}{step_index - 1}"
    guidance_name = "x0" if step_index == 1 else (f"x-{step_index - 1}" if direction > 0 else f"x{step_index - 1}")
    idx = build_record_index(view_records)
    target_view = build_view(config["source_view"], step_spec["yaw_deg"], config["central_band"]["pitch_deg"], config["central_band"]["fov_deg"])
    homography = helper.build_view_homography(idx[source_name]["view"], target_view)
    warped, known_mask, base_missing_mask = helper.warp_image_and_mask(idx[source_name]["image"], homography, (target_view["width"], target_view["height"]))
    warped_combined_risk = helper.warp_risk_map(idx[source_name]["risk_maps"]["combined"], homography, (target_view["width"], target_view["height"]))
    composite_bundle = {"composite": warped, "base_missing_mask": base_missing_mask, "known_mask": known_mask, "warped_combined_risk": warped_combined_risk}
    return InpaintStep(config, generator, initial_center).run(
        target_view, composite_bundle, idx[guidance_name]["image"], "central_band",
        step_spec["name"], "step", view_records,
        {"source_name": source_name, "guidance_name": guidance_name, "warped": warped})


def run_central_merge(config, view_records, generator, initial_center):
    idx = build_record_index(view_records)
    target_view = build_view(config["source_view"], config["central_band"]["merge_yaw_deg"], config["central_band"]["pitch_deg"], config["central_band"]["fov_deg"])
    left_h = helper.build_view_homography(idx["x3"]["view"], target_view)
    right_h = helper.build_view_homography(idx["x-3"]["view"], target_view)
    left_warped, left_known_mask, _ = helper.warp_image_and_mask(idx["x3"]["image"], left_h, (target_view["width"], target_view["height"]))
    right_warped, right_known_mask, _ = helper.warp_image_and_mask(idx["x-3"]["image"], right_h, (target_view["width"], target_view["height"]))
    merge_composite, base_missing_mask = build_merge_composite(left_warped, right_warped, left_known_mask, right_known_mask)
    left_warped_risk = helper.warp_risk_map(idx["x3"]["risk_maps"]["combined"], left_h, (target_view["width"], target_view["height"]))
    right_warped_risk = helper.warp_risk_map(idx["x-3"]["risk_maps"]["combined"], right_h, (target_view["width"], target_view["height"]))
    warped_combined_risk = helper.merge_warped_risks(left_warped_risk, right_warped_risk, left_known_mask, right_known_mask)
    known_mask = np.where((left_known_mask > 0) | (right_known_mask > 0), 255, 0).astype(np.uint8)
    composite_bundle = {"composite": merge_composite, "base_missing_mask": base_missing_mask, "known_mask": known_mask, "warped_combined_risk": warped_combined_risk}
    return InpaintStep(config, generator, initial_center).run(
        target_view, composite_bundle, idx["x0"]["image"], "central_band",
        "x_merge", "merge", view_records,
        {"source_name": "x3+x-3", "guidance_name": "x0", "left_warped": left_warped, "right_warped": right_warped,
         "left_known_mask": left_known_mask, "right_known_mask": right_known_mask,
         "left_warped_risk": left_warped_risk, "right_warped_risk": right_warped_risk, "merge_composite": merge_composite})


def build_prior_crop(image, width, height, use_top, crop_ratio):
    image_height = image.shape[0]
    crop_height = max(int(round(image_height * float(crop_ratio))), 1)
    cropped = image[:crop_height, :, :] if use_top else image[image_height - crop_height:, :, :]
    resized = Image.fromarray(cropped.astype(np.uint8), mode="RGB").resize((width, height), resample=Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def build_overlap_source(record, target_view):
    map_x, map_y, valid = helper.build_view_to_view_remap(record["view"], target_view)
    if int(valid.sum()) <= 0:
        return None
    warped = helper.remap_with_visibility(record["image"], map_x, map_y, valid, cv2.INTER_LINEAR).astype(np.uint8)
    candidate_known_mask = np.where(valid, 255, 0).astype(np.uint8)
    source_weight = helper.build_boundary_weight_map(record["view"]["width"], record["view"]["height"]).astype(np.float32)
    candidate_weight = helper.remap_with_visibility(source_weight, map_x, map_y, valid, cv2.INTER_LINEAR).astype(np.float32)
    candidate_weight = np.where(valid, np.maximum(candidate_weight, 1e-6), 0.0).astype(np.float32)
    candidate_risk = helper.remap_with_visibility(record["risk_maps"]["combined"].astype(np.float32), map_x, map_y, valid, cv2.INTER_LINEAR).astype(np.float32)
    return {
        "name": record["name"],
        "warped": warped,
        "known_mask": candidate_known_mask,
        "weight": candidate_weight,
        "risk": candidate_risk,
    }


def build_target_from_overlaps(target_view, completed_records):
    composite_sum = np.zeros((target_view["height"], target_view["width"], 3), dtype=np.float32)
    weight_sum = np.zeros((target_view["height"], target_view["width"]), dtype=np.float32)
    known_mask = np.zeros((target_view["height"], target_view["width"]), dtype=np.uint8)
    warped_combined_risk = np.zeros((target_view["height"], target_view["width"]), dtype=np.float32)
    overlap_sources = []
    for record in completed_records:
        overlap_source = build_overlap_source(record, target_view)
        if overlap_source is None:
            continue
        valid = overlap_source["known_mask"] > 0
        composite_sum += overlap_source["warped"].astype(np.float32) * overlap_source["weight"][..., None]
        weight_sum += overlap_source["weight"]
        known_mask = np.where(valid, 255, known_mask).astype(np.uint8)
        warped_combined_risk = np.where(valid, np.maximum(warped_combined_risk, overlap_source["risk"]), warped_combined_risk).astype(np.float32)
        overlap_sources.append({
            "name": overlap_source["name"],
            "warped": overlap_source["warped"],
            "known_mask": overlap_source["known_mask"],
            "weight": overlap_source["weight"],
        })
    composite = np.zeros((target_view["height"], target_view["width"], 3), dtype=np.uint8)
    nonzero = weight_sum > 0.0
    composite[nonzero] = np.clip(composite_sum[nonzero] / weight_sum[nonzero, None], 0.0, 255.0).astype(np.uint8)
    base_missing_mask = np.where(known_mask > 0, 0, 255).astype(np.uint8)
    return {
        "composite": composite,
        "known_mask": known_mask,
        "base_missing_mask": base_missing_mask,
        "warped_combined_risk": warped_combined_risk,
        "overlap_sources": overlap_sources,
    }


def run_expansion_target(config, name, kind, yaw_deg, pitch_deg, completed_records, risk_context_records, generator, initial_center, use_top):
    target_view = build_view(config["source_view"], yaw_deg, pitch_deg, config["expansion"]["fov_deg"])
    composite_bundle = build_target_from_overlaps(target_view, completed_records)
    guidance_name_map = {"u0": "x0", "u1": "u0", "u-1": "u1", "u_merge": "x_merge", "d0": "x0", "d1": "d0", "d-1": "d1", "d_merge": "x_merge"}
    guidance_name = guidance_name_map[name]
    guidance_image = build_prior_crop(build_record_index(completed_records)[guidance_name]["image"], target_view["width"], target_view["height"], use_top, config["expansion"]["prior_crop_ratio"])
    extra = {"source_name": ",".join([s["name"] for s in composite_bundle["overlap_sources"]]), "guidance_name": guidance_name,
             "overlap_source_names": [s["name"] for s in composite_bundle["overlap_sources"]], "overlap_sources": composite_bundle["overlap_sources"]}
    return InpaintStep(config, generator, initial_center).run(target_view, composite_bundle, guidance_image, "expansion", name, kind, risk_context_records, extra)


def run_expansion_ring(config, prefix, pitch_deg, central_records, upward_records, downward_records, generator, initial_center, use_top):
    ring_records = upward_records if prefix == "u" else downward_records
    schedule = [
        {"name": prefix + "0", "kind": "seed", "yaw_deg": 0.0},
        {"name": prefix + "1", "kind": "step", "yaw_deg": config["expansion"]["yaw_stride_deg"]},
        {"name": prefix + "-1", "kind": "step", "yaw_deg": -config["expansion"]["yaw_stride_deg"]},
        {"name": prefix + "_merge", "kind": "merge", "yaw_deg": 180.0},
    ]
    for step_spec in schedule:
        completed_records = central_records + ring_records
        record = run_expansion_target(
            config,
            step_spec["name"],
            step_spec["kind"],
            step_spec["yaw_deg"],
            pitch_deg,
            completed_records,
            ring_records,
            generator,
            initial_center,
            use_top,
        )
        ring_records.append(record)
    return ring_records


def find_nearest_pole_record(records, pole_yaw_deg, pole_pitch_deg):
    pole_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32) if pole_pitch_deg > 0.0 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
    best_record = None
    best_alignment = -1e9
    best_yaw_distance = 1e9
    center_ray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for record in records:
        rotation = helper.build_view_rotation(record["view"])
        direction = rotation @ center_ray
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        alignment = float(direction @ pole_direction)
        yaw_distance = abs(float(record["view"]["yaw_deg"]) - float(pole_yaw_deg))
        while yaw_distance > 180.0:
            yaw_distance -= 360.0
        yaw_distance = abs(yaw_distance)
        if best_record is None or alignment > best_alignment + 1e-8 or (abs(alignment - best_alignment) <= 1e-8 and yaw_distance < best_yaw_distance):
            best_record = record
            best_alignment = alignment
            best_yaw_distance = yaw_distance
    if best_record is None:
        raise RuntimeError("Missing pole guidance record.")
    return best_record


def run_pole_target(config, name, pitch_deg, completed_records, risk_context_records, guidance_records, generator, initial_center, use_top):
    target_view = build_view(config["source_view"], 0.0, pitch_deg, config["pole_closure"]["fov_deg"])
    composite_bundle = build_target_from_overlaps(target_view, completed_records)
    guidance_record = find_nearest_pole_record(guidance_records, 0.0, pitch_deg)
    guidance_image = build_prior_crop(guidance_record["image"], target_view["width"], target_view["height"], use_top, config["pole_closure"]["prior_crop_ratio"])
    extra = {"source_name": ",".join([s["name"] for s in composite_bundle["overlap_sources"]]), "guidance_name": guidance_record["name"],
             "overlap_source_names": [s["name"] for s in composite_bundle["overlap_sources"]], "overlap_sources": composite_bundle["overlap_sources"]}
    return InpaintStep(config, generator, initial_center).run(target_view, composite_bundle, guidance_image, "pole_closure", name, "pole", risk_context_records, extra)


def run_pipeline(config_path):
    config = helper.load_pipeline_config(config_path)
    config["output"]["run_dir"] = build_timestamped_run_dir(config["output"]["run_dir"])
    device = "cuda" if helper.torch.cuda.is_available() else "cpu"
    generator = helper.make_torch_generator(device, config.get("seed"))
    w, h = config["output"]["pano_width"], config["output"]["pano_height"]

    initial_view, source_image_used = generate_or_load_initial_view(config, generator)
    initial_record = {"name": "x0", "kind": "initial", "view": copy.deepcopy(config["source_view"]), "image": initial_view}
    initial_center = compute_initial_risks(config, initial_record)

    # Central band
    central_records = [initial_record]
    for spec in build_central_schedule(config):
        central_records.append(run_central_step(config, spec, central_records, generator, initial_center))
    central_records.append(run_central_merge(config, central_records, generator, initial_center))

    # Expansion rings
    upward_records = run_expansion_ring(config, "u", config["expansion"]["pitch_offset_deg"], central_records, [], [], generator, initial_center, True)
    downward_records = run_expansion_ring(config, "d", -config["expansion"]["pitch_offset_deg"], central_records, upward_records, [], generator, initial_center, False)

    # Pole closure
    pre_pole_records = central_records + upward_records + downward_records
    top_pole_record = run_pole_target(config, "top_pole", 90.0, pre_pole_records, upward_records, upward_records, generator, initial_center, True)
    bottom_pole_record = run_pole_target(config, "bottom_pole", -90.0, pre_pole_records, downward_records, downward_records, generator, initial_center, False)
    pole_records = [top_pole_record, bottom_pole_record]

    # Stitching
    def stitch(records, pitch_min=None, pitch_max=None):
        return helper.stitch_equirectangular_views(records, w, h, pitch_min, pitch_max)

    central_panorama, _ = stitch(central_records, config["central_band"]["stitch_pitch_min_deg"], config["central_band"]["stitch_pitch_max_deg"])
    upward_panorama, _ = stitch(upward_records)
    downward_panorama, _ = stitch(downward_records)
    pre_pole_panorama, _ = stitch(pre_pole_records)
    top_pole_panorama, _ = stitch([top_pole_record])
    bottom_pole_panorama, _ = stitch([bottom_pole_record])
    full_panorama, full_coverage = stitch(pre_pole_records + pole_records)

    uncovered_pixels = int((full_coverage <= 0).sum())
    expected_result_reached = uncovered_pixels == 0
    pipeline_note = "Pipeline reached its expected result: final full spherical panorama was produced end-to-end with top and bottom poles closed." if expected_result_reached else f"Pipeline did not reach its expected result: final full-sphere coverage still has {uncovered_pixels} uncovered pixels."

    artifacts = {
        "config": config, "prompt": config["prompt"], "initial_view": initial_view,
        "central_manifest": build_steps_manifest(central_records), "upward_manifest": build_steps_manifest(upward_records),
        "downward_manifest": build_steps_manifest(downward_records), "pole_manifest": build_steps_manifest(pole_records),
        "central_records": central_records, "upward_records": upward_records, "downward_records": downward_records, "pole_records": pole_records,
        "central_panorama": central_panorama,
        "upward_panorama": upward_panorama,
        "downward_panorama": downward_panorama,
        "pre_pole_panorama": pre_pole_panorama,
        "top_pole_panorama": top_pole_panorama,
        "bottom_pole_panorama": bottom_pole_panorama,
        "full_panorama": full_panorama,
        "full_coverage": full_coverage,
        "pipeline_note": pipeline_note,
        "pano_width": config["output"]["pano_width"],
        "pano_height": config["output"]["pano_height"],
    }

    helper.save_pipeline_outputs(config["output"]["run_dir"], artifacts, config["output"]["debug"])

    return {
        "run_dir": config["output"]["run_dir"],
        "final_panorama_path": os.path.join(config["output"]["run_dir"], "12_full_sphere_equirect.png"),
        "source_image_used": source_image_used,
        "expected_result_reached": expected_result_reached,
    }


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(root, "pipeline.toml")

    parser = argparse.ArgumentParser(description="Run the full PanoFree panorama pipeline.")
    parser.add_argument(
        "--config",
        required=False,
        default=default_config,
        help="Path to the pipeline TOML config.",
    )
    args = parser.parse_args()

    result = run_pipeline(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
