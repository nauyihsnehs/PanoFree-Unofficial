import json
import os

import numpy as np
from PIL import Image, ImageDraw

from .camera import project_perspective_to_equirectangular, view_center_on_equirectangular


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def save_image(path, image):
    if image.ndim == 2:
        Image.fromarray(image.astype(np.uint8), mode="L").save(path)
        return
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def build_contact_sheet(images, labels):
    width, height = images[0].size
    label_height = 32
    sheet = Image.new("RGB", (width * len(images), height + label_height), color=(24, 24, 24))
    drawer = ImageDraw.Draw(sheet)

    for index, image in enumerate(images):
        x0 = index * width
        sheet.paste(image, (x0, label_height))
        drawer.text((x0 + 10, 8), labels[index], fill=(255, 255, 255))
    return sheet


def build_equirectangular_debug(image, view, pano_width, pano_height, point_color):
    canvas, mask = project_perspective_to_equirectangular(image, view, pano_width, pano_height)
    overlay = Image.fromarray(canvas, mode="RGB")
    drawer = ImageDraw.Draw(overlay)
    center_x, center_y = view_center_on_equirectangular(view, pano_width, pano_height)
    radius = 10
    drawer.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        outline=point_color,
        width=3,
    )
    return np.array(overlay, dtype=np.uint8), mask


def save_phase1_debug_artifacts(run_dir, artifacts):
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "00_config.json"), artifacts["config"])
    save_text(os.path.join(run_dir, "01_prompt.txt"), artifacts["prompt"])
    save_image(os.path.join(run_dir, "02_initial_view.png"), artifacts["initial_view"])

    homography_path = os.path.join(run_dir, "03_homography.txt")
    with open(homography_path, "w", encoding="utf-8") as handle:
        matrix = artifacts["homography"]
        for row in matrix:
            handle.write(" ".join("{:.8f}".format(float(value)) for value in row))
            handle.write("\n")

    save_image(os.path.join(run_dir, "04_warped.png"), artifacts["warped"])
    save_image(os.path.join(run_dir, "05_known_mask.png"), artifacts["known_mask"])
    save_image(os.path.join(run_dir, "06_missing_mask.png"), artifacts["missing_mask"])
    save_image(os.path.join(run_dir, "07_inpaint_input.png"), artifacts["inpaint_input"])
    save_image(os.path.join(run_dir, "08_inpaint_output.png"), artifacts["inpaint_output"])

    images = [
        Image.fromarray(artifacts["initial_view"], mode="RGB"),
        Image.fromarray(artifacts["warped"], mode="RGB"),
        Image.fromarray(artifacts["inpaint_output"], mode="RGB"),
    ]
    labels = ["initial", "warped", "inpainted"]
    contact_sheet = build_contact_sheet(images, labels)
    contact_sheet.save(os.path.join(run_dir, "09_contact_sheet.png"))

    source_overlay, _ = build_equirectangular_debug(
        artifacts["initial_view"],
        artifacts["source_view"],
        artifacts["pano_width"],
        artifacts["pano_height"],
        (255, 80, 80),
    )
    target_overlay, _ = build_equirectangular_debug(
        artifacts["inpaint_output"],
        artifacts["target_view"],
        artifacts["pano_width"],
        artifacts["pano_height"],
        (80, 220, 255),
    )
    save_image(os.path.join(run_dir, "10_source_on_equirect.png"), source_overlay)
    save_image(os.path.join(run_dir, "11_target_on_equirect.png"), target_overlay)


def save_phase2_debug_artifacts(run_dir, artifacts):
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "00_config.json"), artifacts["config"])
    save_text(os.path.join(run_dir, "01_prompt.txt"), artifacts["prompt"])
    save_json(os.path.join(run_dir, "02_steps_manifest.json"), artifacts["steps_manifest"])
    save_image(os.path.join(run_dir, "03_initial_view.png"), artifacts["initial_view"])
    save_image(os.path.join(run_dir, "04_central_360_equirect.png"), artifacts["stitched_panorama"])
    save_image(os.path.join(run_dir, "05_central_360_coverage.png"), artifacts["stitched_coverage"])
    save_image(os.path.join(run_dir, "06_central_360_band_crop.png"), artifacts["band_crop"])

    contact_images = [Image.fromarray(artifacts["initial_view"], mode="RGB")]
    contact_labels = ["x0"]
    for record in artifacts["view_records"]:
        contact_images.append(Image.fromarray(record["image"], mode="RGB"))
        contact_labels.append(record["name"])
    build_contact_sheet(contact_images, contact_labels).save(
        os.path.join(run_dir, "07_view_contact_sheet.png")
    )

    for index, record in enumerate(artifacts["view_records"]):
        step_dir = os.path.join(run_dir, "{:02d}_{}".format(index + 1, record["name"]))
        ensure_dir(step_dir)
        save_json(
            os.path.join(step_dir, "00_meta.json"),
            {
                "name": record["name"],
                "kind": record["kind"],
                "yaw_deg": record["view"]["yaw_deg"],
                "pitch_deg": record["view"]["pitch_deg"],
                "source_name": record.get("source_name", ""),
                "guidance_name": record.get("guidance_name", ""),
            },
        )
        save_image(os.path.join(step_dir, "01_output.png"), record["image"])
        save_image(os.path.join(step_dir, "02_guidance.png"), record["guidance_image"])
        save_image(os.path.join(step_dir, "03_guided_input.png"), record["guided_input"])
        save_image(os.path.join(step_dir, "04_stitched_valid_mask.png"), record["stitched_valid_mask"])
        save_image(
            os.path.join(step_dir, "05_stitched_weight_map.png"),
            np.clip(record["stitched_weight_map"], 0.0, 255.0).astype(np.uint8),
        )
        overlay, _ = build_equirectangular_debug(
            record["image"],
            record["view"],
            artifacts["pano_width"],
            artifacts["pano_height"],
            (80, 220, 255),
        )
        save_image(os.path.join(step_dir, "06_on_equirect.png"), overlay)

        if record["kind"] == "step":
            save_image(os.path.join(step_dir, "07_warped.png"), record["warped"])
            save_image(os.path.join(step_dir, "08_missing_mask.png"), record["missing_mask"])
        else:
            save_image(os.path.join(step_dir, "07_left_warped.png"), record["left_warped"])
            save_image(os.path.join(step_dir, "08_right_warped.png"), record["right_warped"])
            save_image(os.path.join(step_dir, "09_left_known_mask.png"), record["left_known_mask"])
            save_image(os.path.join(step_dir, "10_right_known_mask.png"), record["right_known_mask"])
            save_image(os.path.join(step_dir, "11_merge_composite.png"), record["merge_composite"])
            save_image(os.path.join(step_dir, "12_missing_mask.png"), record["missing_mask"])


def save_phase3_debug_artifacts(run_dir, artifacts):
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "00_config.json"), artifacts["config"])
    save_text(os.path.join(run_dir, "01_prompt.txt"), artifacts["prompt"])
    save_json(os.path.join(run_dir, "02_steps_manifest.json"), artifacts["steps_manifest"])
    save_image(os.path.join(run_dir, "03_initial_view.png"), artifacts["initial_view"])
    save_image(os.path.join(run_dir, "04_central_360_equirect.png"), artifacts["stitched_panorama"])
    save_image(os.path.join(run_dir, "05_central_360_coverage.png"), artifacts["stitched_coverage"])
    save_image(os.path.join(run_dir, "06_central_360_band_crop.png"), artifacts["band_crop"])

    contact_images = []
    contact_labels = []
    for record in artifacts["view_records"]:
        contact_images.append(Image.fromarray(record["image"], mode="RGB"))
        contact_labels.append(record["name"])
    build_contact_sheet(contact_images, contact_labels).save(
        os.path.join(run_dir, "07_view_contact_sheet.png")
    )

    for index, record in enumerate(artifacts["view_records"]):
        step_dir = os.path.join(run_dir, "{:02d}_{}".format(index, record["name"]))
        ensure_dir(step_dir)
        save_json(
            os.path.join(step_dir, "00_meta.json"),
            {
                "name": record["name"],
                "kind": record["kind"],
                "yaw_deg": record["view"]["yaw_deg"],
                "pitch_deg": record["view"]["pitch_deg"],
                "source_name": record.get("source_name", ""),
                "guidance_name": record.get("guidance_name", ""),
                "overlap_source_names": record.get("overlap_source_names", []),
            },
        )
        save_image(os.path.join(step_dir, "01_output.png"), record["image"])
        save_image(os.path.join(step_dir, "02_guidance.png"), record["guidance_image"])
        save_image(os.path.join(step_dir, "03_guided_input.png"), record["guided_input"])
        save_image(os.path.join(step_dir, "04_stitched_valid_mask.png"), record["stitched_valid_mask"])
        save_image(
            os.path.join(step_dir, "05_stitched_weight_map.png"),
            np.clip(record["stitched_weight_map"], 0.0, 255.0).astype(np.uint8),
        )
        overlay, _ = build_equirectangular_debug(
            record["image"],
            record["view"],
            artifacts["pano_width"],
            artifacts["pano_height"],
            (80, 220, 255),
        )
        save_image(os.path.join(step_dir, "06_on_equirect.png"), overlay)
        save_image(os.path.join(step_dir, "07_risk_distance.png"), record["risk_maps"]["distance"] * 255.0)
        save_image(os.path.join(step_dir, "08_risk_edge.png"), record["risk_maps"]["edge"] * 255.0)
        save_image(os.path.join(step_dir, "09_risk_color.png"), record["risk_maps"]["color"] * 255.0)
        save_image(os.path.join(step_dir, "10_risk_smoothness.png"), record["risk_maps"]["smoothness"] * 255.0)
        save_image(os.path.join(step_dir, "11_risk_combined.png"), record["risk_maps"]["combined"] * 255.0)

        if record["kind"] == "initial":
            continue

        save_image(os.path.join(step_dir, "12_base_missing_mask.png"), record["base_missing_mask"])
        save_image(os.path.join(step_dir, "13_warped_combined_risk.png"), record["warped_combined_risk"] * 255.0)
        save_image(os.path.join(step_dir, "14_risk_selected_mask.png"), record["risk_selected_mask"])
        save_image(os.path.join(step_dir, "15_remasked_mask.png"), record["remasked_mask"])
        save_image(os.path.join(step_dir, "16_smoothed_mask.png"), record["smoothed_mask"])

        if record["kind"] == "step":
            save_image(os.path.join(step_dir, "17_warped.png"), record["warped"])
            save_image(os.path.join(step_dir, "18_known_mask.png"), record["known_mask"])
        else:
            save_image(os.path.join(step_dir, "17_left_warped.png"), record["left_warped"])
            save_image(os.path.join(step_dir, "18_right_warped.png"), record["right_warped"])
            save_image(os.path.join(step_dir, "19_left_known_mask.png"), record["left_known_mask"])
            save_image(os.path.join(step_dir, "20_right_known_mask.png"), record["right_known_mask"])
            save_image(os.path.join(step_dir, "21_left_warped_risk.png"), record["left_warped_risk"] * 255.0)
            save_image(os.path.join(step_dir, "22_right_warped_risk.png"), record["right_warped_risk"] * 255.0)
            save_image(os.path.join(step_dir, "23_merge_composite.png"), record["merge_composite"])


def save_phase4_group_artifacts(group_dir, records, pano_width, pano_height):
    ensure_dir(group_dir)

    contact_images = []
    contact_labels = []
    for record in records:
        contact_images.append(Image.fromarray(record["image"], mode="RGB"))
        contact_labels.append(record["name"])
    build_contact_sheet(contact_images, contact_labels).save(
        os.path.join(group_dir, "00_contact_sheet.png")
    )

    for index, record in enumerate(records):
        step_dir = os.path.join(group_dir, "{:02d}_{}".format(index, record["name"]))
        ensure_dir(step_dir)
        save_json(
            os.path.join(step_dir, "00_meta.json"),
            {
                "name": record["name"],
                "kind": record["kind"],
                "yaw_deg": record["view"]["yaw_deg"],
                "pitch_deg": record["view"]["pitch_deg"],
                "source_name": record.get("source_name", ""),
                "guidance_name": record.get("guidance_name", ""),
            },
        )
        save_image(os.path.join(step_dir, "01_output.png"), record["image"])
        if "guidance_image" in record:
            save_image(os.path.join(step_dir, "02_guidance.png"), record["guidance_image"])
        if "guided_input" in record:
            save_image(os.path.join(step_dir, "03_guided_input.png"), record["guided_input"])
        if "stitched_valid_mask" in record:
            save_image(os.path.join(step_dir, "04_stitched_valid_mask.png"), record["stitched_valid_mask"])
        if "stitched_weight_map" in record:
            save_image(
                os.path.join(step_dir, "05_stitched_weight_map.png"),
                np.clip(record["stitched_weight_map"], 0.0, 255.0).astype(np.uint8),
            )
        overlay, _ = build_equirectangular_debug(
            record["image"],
            record["view"],
            pano_width,
            pano_height,
            (80, 220, 255),
        )
        save_image(os.path.join(step_dir, "06_on_equirect.png"), overlay)

        if "risk_maps" in record:
            save_image(os.path.join(step_dir, "07_risk_distance.png"), record["risk_maps"]["distance"] * 255.0)
            save_image(os.path.join(step_dir, "08_risk_edge.png"), record["risk_maps"]["edge"] * 255.0)
            save_image(os.path.join(step_dir, "09_risk_color.png"), record["risk_maps"]["color"] * 255.0)
            save_image(os.path.join(step_dir, "10_risk_smoothness.png"), record["risk_maps"]["smoothness"] * 255.0)
            save_image(os.path.join(step_dir, "11_risk_combined.png"), record["risk_maps"]["combined"] * 255.0)

        if "base_missing_mask" in record:
            save_image(os.path.join(step_dir, "12_base_missing_mask.png"), record["base_missing_mask"])
        if "warped_combined_risk" in record:
            save_image(os.path.join(step_dir, "13_warped_combined_risk.png"), record["warped_combined_risk"] * 255.0)
        if "risk_selected_mask" in record:
            save_image(os.path.join(step_dir, "14_risk_selected_mask.png"), record["risk_selected_mask"])
        if "remasked_mask" in record:
            save_image(os.path.join(step_dir, "15_remasked_mask.png"), record["remasked_mask"])
        if "smoothed_mask" in record:
            save_image(os.path.join(step_dir, "16_smoothed_mask.png"), record["smoothed_mask"])

        if "warped" in record:
            save_image(os.path.join(step_dir, "17_warped.png"), record["warped"])
        if "known_mask" in record:
            save_image(os.path.join(step_dir, "18_known_mask.png"), record["known_mask"])

        for source_index, source in enumerate(record.get("overlap_sources", [])):
            prefix = "{:02d}_{}".format(source_index, source["name"])
            save_image(os.path.join(step_dir, "24_{}_warped.png".format(prefix)), source["warped"])
            save_image(os.path.join(step_dir, "25_{}_known_mask.png".format(prefix)), source["known_mask"])
            save_image(
                os.path.join(step_dir, "26_{}_weight.png".format(prefix)),
                np.clip(source["weight"], 0.0, 255.0).astype(np.uint8),
            )


def save_phase4_debug_artifacts(run_dir, artifacts):
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "00_config.json"), artifacts["config"])
    save_text(os.path.join(run_dir, "01_prompt.txt"), artifacts["prompt"])
    save_json(os.path.join(run_dir, "02_central_manifest.json"), artifacts["central_manifest"])
    save_json(os.path.join(run_dir, "03_upward_manifest.json"), artifacts["upward_manifest"])
    save_json(os.path.join(run_dir, "04_downward_manifest.json"), artifacts["downward_manifest"])
    save_image(os.path.join(run_dir, "05_central_360_equirect.png"), artifacts["central_panorama"])
    save_image(os.path.join(run_dir, "06_upward_partial_equirect.png"), artifacts["upward_panorama"])
    save_image(os.path.join(run_dir, "07_downward_partial_equirect.png"), artifacts["downward_panorama"])
    save_image(os.path.join(run_dir, "08_full_sphere_without_poles_equirect.png"), artifacts["full_panorama"])
    save_image(os.path.join(run_dir, "09_full_sphere_coverage.png"), artifacts["full_coverage"])

    contact_images = []
    contact_labels = []
    for record in artifacts["central_records"] + artifacts["upward_records"] + artifacts["downward_records"]:
        contact_images.append(Image.fromarray(record["image"], mode="RGB"))
        contact_labels.append(record["name"])
    build_contact_sheet(contact_images, contact_labels).save(
        os.path.join(run_dir, "10_all_view_contact_sheet.png")
    )

    save_phase4_group_artifacts(
        os.path.join(run_dir, "central"),
        artifacts["central_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )
    save_phase4_group_artifacts(
        os.path.join(run_dir, "upward"),
        artifacts["upward_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )
    save_phase4_group_artifacts(
        os.path.join(run_dir, "downward"),
        artifacts["downward_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )


def save_phase5_debug_artifacts(run_dir, artifacts):
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "00_config.json"), artifacts["config"])
    save_text(os.path.join(run_dir, "01_prompt.txt"), artifacts["prompt"])
    save_json(os.path.join(run_dir, "02_central_manifest.json"), artifacts["central_manifest"])
    save_json(os.path.join(run_dir, "03_upward_manifest.json"), artifacts["upward_manifest"])
    save_json(os.path.join(run_dir, "04_downward_manifest.json"), artifacts["downward_manifest"])
    save_json(os.path.join(run_dir, "05_pole_manifest.json"), artifacts["pole_manifest"])
    save_image(os.path.join(run_dir, "06_central_360_equirect.png"), artifacts["central_panorama"])
    save_image(os.path.join(run_dir, "07_upward_partial_equirect.png"), artifacts["upward_panorama"])
    save_image(os.path.join(run_dir, "08_downward_partial_equirect.png"), artifacts["downward_panorama"])
    save_image(os.path.join(run_dir, "09_full_sphere_without_poles_equirect.png"), artifacts["pre_pole_panorama"])
    save_image(os.path.join(run_dir, "10_top_pole_partial_equirect.png"), artifacts["top_pole_panorama"])
    save_image(os.path.join(run_dir, "11_bottom_pole_partial_equirect.png"), artifacts["bottom_pole_panorama"])
    save_image(os.path.join(run_dir, "12_full_sphere_equirect.png"), artifacts["full_panorama"])
    save_image(os.path.join(run_dir, "13_full_sphere_coverage.png"), artifacts["full_coverage"])

    contact_images = []
    contact_labels = []
    for record in artifacts["central_records"] + artifacts["upward_records"] + artifacts["downward_records"] + artifacts["pole_records"]:
        contact_images.append(Image.fromarray(record["image"], mode="RGB"))
        contact_labels.append(record["name"])
    build_contact_sheet(contact_images, contact_labels).save(
        os.path.join(run_dir, "14_all_view_contact_sheet.png")
    )
    save_text(os.path.join(run_dir, "15_phase_note.txt"), artifacts["phase_note"])

    save_phase4_group_artifacts(
        os.path.join(run_dir, "central"),
        artifacts["central_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )
    save_phase4_group_artifacts(
        os.path.join(run_dir, "upward"),
        artifacts["upward_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )
    save_phase4_group_artifacts(
        os.path.join(run_dir, "downward"),
        artifacts["downward_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )
    save_phase4_group_artifacts(
        os.path.join(run_dir, "poles"),
        artifacts["pole_records"],
        artifacts["pano_width"],
        artifacts["pano_height"],
    )
