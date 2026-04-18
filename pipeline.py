import argparse
import json
import os
import tomli

import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, DiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import DDIMScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import EulerAncestralDiscreteScheduler

from pipeline_helper import build_yaw_schedule
from pipeline_helper import dump_run_manifest
from pipeline_helper import merge_dict
from pipeline_helper import prepare_output_dirs
from pipeline_helper import project_view_to_panorama
from pipeline_helper import refine_inpaint_mask
from pipeline_helper import save_image
from pipeline_helper import save_mask
from pipeline_helper import set_seed
from pipeline_helper import warp_view_to_view
from pipeline_helper import mask_to_pil
from pipeline_helper import numpy_to_pil
from pipeline_helper import pil_to_numpy

DEFAULT_CONFIG = {
    "run": {
        "name": "m1_360_run",
        "output_root": "outputs",
        "seed": 1234,
        "device": "cuda",
        "dtype": "float16",
    },
    "prompt": {
        "text": "",
        "negative": "",
    },
    "models": {
        "text2img": "",
        "inpaint": "",
        "scheduler": "default",
        "local_files_only": False,
    },
    "generation": {
        "view_size": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "init_num_inference_steps": 35,
        "init_guidance_scale": 8.0,
        "strength": 0.85,
    },
    "panorama": {
        "width": 2048,
        "height": 1024,
        "hfov_deg": 90.0,
        "pitch_deg": 0.0,
        "yaw_start_deg": 0.0,
        "yaw_step_deg": 45.0,
        "num_views": 8,
    },
    "mask": {
        "dilate_px": 21,
        "blur_px": 13,
        "min_fill_ratio": 0.02,
    },
}

SCHEDULERS = {
    "default": None,
    "ddim": DDIMScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
}


def resolve_dtype(dtype_name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError("Unsupported dtype: %s" % dtype_name)
    return mapping[dtype_name]


def apply_scheduler(pipe, scheduler_name):
    scheduler_cls = SCHEDULERS.get(scheduler_name)
    if scheduler_cls is None:
        return pipe
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    return pipe


def build_generator(seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def load_config(config_path):
    with open(config_path, "rb") as handle:
        user_config = tomli.load(handle)
    config = merge_dict(DEFAULT_CONFIG, user_config)

    if not config["prompt"]["text"].strip():
        raise ValueError("prompt.text must not be empty")
    if not config["models"]["text2img"].strip():
        raise ValueError("models.text2img must not be empty")
    if not config["models"]["inpaint"].strip():
        raise ValueError("models.inpaint must not be empty")

    config["config_path"] = os.path.abspath(config_path)
    config["run"]["output_root"] = os.path.abspath(config["run"]["output_root"])
    config["generation"]["view_size"] = int(config["generation"]["view_size"])
    config["panorama"]["width"] = int(config["panorama"]["width"])
    config["panorama"]["height"] = int(config["panorama"]["height"])
    config["panorama"]["num_views"] = int(config["panorama"]["num_views"])
    return config


class DiffusionBackend:
    def __init__(self, config):
        run = config["run"]
        models = config["models"]
        generation = config["generation"]

        self.device = run["device"]
        self.dtype = resolve_dtype(run["dtype"])
        self.local_files_only = models["local_files_only"]
        self.generation = generation
        self.prompt = config["prompt"]["text"]
        self.negative_prompt = config["prompt"]["negative"]
        self.seed = run["seed"]

        variant = 'fp16'  # None if '1-5' in model_name else 'fp16'
        # self.text2img = AutoPipelineForText2Image.from_pretrained(
        self.text2img = DiffusionPipeline.from_pretrained(
            models["text2img"], variant=variant,
            torch_dtype=self.dtype, safety_checker=None,
            # local_files_only=self.local_files_only,
        )
        self.inpaint = AutoPipelineForInpainting.from_pretrained(
            models["inpaint"], variant=variant,
            torch_dtype=self.dtype, safety_checker=None,
            # local_files_only=self.local_files_only,
        )

        apply_scheduler(self.text2img, models["scheduler"])
        apply_scheduler(self.inpaint, models["scheduler"])

        self.text2img = self.text2img.to(self.device)
        self.inpaint = self.inpaint.to(self.device)

        if hasattr(self.text2img, "enable_attention_slicing"):
            self.text2img.enable_attention_slicing()
        if hasattr(self.inpaint, "enable_attention_slicing"):
            self.inpaint.enable_attention_slicing()

    def generate_initial_view(self):
        generator = build_generator(self.seed, self.device)
        image = self.text2img(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=self.generation["view_size"],
            height=self.generation["view_size"],
            num_inference_steps=self.generation["init_num_inference_steps"],
            guidance_scale=self.generation["init_guidance_scale"],
            generator=generator,
        ).images[0]
        return pil_to_numpy(image)

    def inpaint_view(self, image, mask, step_index):
        generator = build_generator(self.seed + step_index + 1, self.device)
        image_pil = numpy_to_pil(image)
        mask_pil = mask_to_pil(mask)
        result = self.inpaint(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            width=self.generation["view_size"],
            height=self.generation["view_size"],
            num_inference_steps=self.generation["num_inference_steps"],
            guidance_scale=self.generation["guidance_scale"],
            strength=self.generation["strength"],
            generator=generator,
        ).images[0]
        return pil_to_numpy(result)


def compose_panorama(canvas, known_mask, contribution, contribution_mask):
    write_mask = contribution_mask > 0.0
    canvas[write_mask] = contribution[write_mask]
    known_mask[write_mask] = 1.0
    return canvas, known_mask


def make_step_name(index, yaw_deg):
    return "%02d_yaw_%03d" % (index, int(round(yaw_deg)) % 360)


def save_panorama_snapshot(output_dirs, step_name, panorama_canvas):
    path = os.path.join(output_dirs["panorama_steps"], "%s.png" % step_name)
    save_image(path, panorama_canvas)


def run_pipeline(config):
    set_seed(config["run"]["seed"])
    output_dirs = prepare_output_dirs(config)
    dump_run_manifest(config, output_dirs)

    backend = DiffusionBackend(config)
    yaw_schedule = build_yaw_schedule(config)

    panorama_width = config["panorama"]["width"]
    panorama_height = config["panorama"]["height"]
    hfov_deg = config["panorama"]["hfov_deg"]
    pitch_deg = config["panorama"]["pitch_deg"]

    panorama_canvas = np.zeros((panorama_height, panorama_width, 3), dtype=np.float32)
    panorama_known = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    current_view = backend.generate_initial_view()
    current_yaw = yaw_schedule[0]

    step_name = make_step_name(0, current_yaw)
    save_image(os.path.join(output_dirs["views"], "%s.png" % step_name), current_view)

    contribution, contribution_mask = project_view_to_panorama(
        current_view,
        current_yaw,
        pitch_deg,
        panorama_width,
        panorama_height,
        hfov_deg,
    )
    panorama_canvas, panorama_known = compose_panorama(
        panorama_canvas,
        panorama_known,
        contribution,
        contribution_mask,
    )
    save_panorama_snapshot(output_dirs, step_name, panorama_canvas)

    mask_config = config["mask"]

    for index, next_yaw in enumerate(yaw_schedule[1:], start=1):
        step_name = make_step_name(index, next_yaw)
        warped, valid_mask = warp_view_to_view(
            current_view,
            current_yaw,
            pitch_deg,
            next_yaw,
            pitch_deg,
            hfov_deg,
        )

        missing_mask = 1.0 - valid_mask
        refined_mask = refine_inpaint_mask(
            missing_mask,
            mask_config["dilate_px"],
            mask_config["blur_px"],
        )

        save_image(os.path.join(output_dirs["warps"], "%s.png" % step_name), warped)
        save_mask(os.path.join(output_dirs["masks"], "%s.png" % step_name), refined_mask)

        if refined_mask.mean() < float(mask_config["min_fill_ratio"]):
            generated_view = warped
        else:
            generated_view = backend.inpaint_view(warped, refined_mask, index)

        save_image(os.path.join(output_dirs["views"], "%s.png" % step_name), generated_view)

        contribution, contribution_mask = project_view_to_panorama(
            generated_view,
            next_yaw,
            pitch_deg,
            panorama_width,
            panorama_height,
            hfov_deg,
        )
        panorama_canvas, panorama_known = compose_panorama(
            panorama_canvas,
            panorama_known,
            contribution,
            contribution_mask,
        )
        save_panorama_snapshot(output_dirs, step_name, panorama_canvas)

        current_view = generated_view
        current_yaw = next_yaw

    final_panorama_path = os.path.join(output_dirs["run_dir"], "panorama_final.png")
    final_mask_path = os.path.join(output_dirs["run_dir"], "panorama_known_mask.png")
    save_image(final_panorama_path, panorama_canvas)
    save_mask(final_mask_path, panorama_known)

    return {
        "run_dir": output_dirs["run_dir"],
        "final_panorama_path": final_panorama_path,
        "final_mask_path": final_mask_path,
        "num_views": len(yaw_schedule),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="PanoFree M1 baseline runner")
    parser.add_argument("--config", default="config.toml", help="Path to the TOML config file")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
