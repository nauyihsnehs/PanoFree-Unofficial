import argparse
import json
import os
from datetime import datetime

from pipeline_helper import Config
from pipeline_helper import Flow
from pipeline_helper import Geometry
from pipeline_helper import Output
from pipeline_helper import Runtime


def build_timestamped_run_dir(base_dir):
    base_dir = os.path.abspath(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_dir, "{}_{:02d}".format(timestamp, suffix))
        suffix += 1
    return run_dir


def run_pipeline(config_path):
    config = Config.load_config(config_path)
    config["output"]["run_dir"] = build_timestamped_run_dir(config["output"]["run_dir"])

    generator = Runtime.create_generator(config.get("seed"))
    initial_view, source_image_used = Flow.generate_or_load_initial_view(config, generator)
    initial_record = Flow.create_initial_record(config, initial_view)
    initial_center = Flow.compute_initial_risks(config, initial_record)

    central_records = [initial_record]
    for step_spec in Flow.build_central_schedule(config):
        central_records.append(Flow.run_central_step(config, step_spec, central_records, generator, initial_center))
    central_records.append(Flow.run_central_merge(config, central_records, generator, initial_center))

    central_panorama, _ = Geometry.stitch_equirectangular_views(
        central_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        config["central_band"]["stitch_pitch_min_deg"],
        config["central_band"]["stitch_pitch_max_deg"],
    )

    upward_records = Flow.run_expansion_ring(
        config,
        "u",
        config["expansion"]["pitch_offset_deg"],
        central_records,
        [],
        [],
        generator,
        initial_center,
        True,
    )
    downward_records = Flow.run_expansion_ring(
        config,
        "d",
        -config["expansion"]["pitch_offset_deg"],
        central_records,
        upward_records,
        [],
        generator,
        initial_center,
        False,
    )

    upward_panorama, _ = Geometry.stitch_equirectangular_views(
        upward_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    downward_panorama, _ = Geometry.stitch_equirectangular_views(
        downward_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )

    pre_pole_records = central_records + upward_records + downward_records
    pre_pole_panorama, _ = Geometry.stitch_equirectangular_views(
        pre_pole_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )

    top_pole_record = Flow.run_pole_target(
        config,
        "top_pole",
        90.0,
        pre_pole_records,
        upward_records,
        upward_records,
        generator,
        initial_center,
        True,
    )
    bottom_pole_record = Flow.run_pole_target(
        config,
        "bottom_pole",
        -90.0,
        pre_pole_records,
        downward_records,
        downward_records,
        generator,
        initial_center,
        False,
    )
    pole_records = [top_pole_record, bottom_pole_record]

    top_pole_panorama, _ = Geometry.stitch_equirectangular_views(
        [top_pole_record],
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    bottom_pole_panorama, _ = Geometry.stitch_equirectangular_views(
        [bottom_pole_record],
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    full_panorama, full_coverage = Geometry.stitch_equirectangular_views(
        pre_pole_records + pole_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )

    uncovered_pixels = int((full_coverage <= 0).sum())
    expected_result_reached = uncovered_pixels == 0
    pipeline_note = Flow.build_pipeline_note(expected_result_reached, uncovered_pixels)

    artifacts = {
        "config": config,
        "prompt": config["prompt"],
        "initial_view": initial_view,
        "central_manifest": Flow.build_steps_manifest(central_records),
        "upward_manifest": Flow.build_steps_manifest(upward_records),
        "downward_manifest": Flow.build_steps_manifest(downward_records),
        "pole_manifest": Flow.build_steps_manifest(pole_records),
        "central_records": central_records,
        "upward_records": upward_records,
        "downward_records": downward_records,
        "pole_records": pole_records,
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

    if config["output"]["debug"]:
        Output.save_pipeline_outputs_full(config["output"]["run_dir"], artifacts)
    else:
        Output.save_pipeline_outputs_minimal(config["output"]["run_dir"], artifacts)

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
