import copy
import json
import math
import os

import cv2
import diffusers
import torch
import tomli
import numpy as np
from PIL import Image
from PIL import ImageDraw

PIPELINE_CACHE = {}

_PIPELINE_DEFAULTS = {
    "prompt": "a realistic indoor panorama",
    "seed": 1234,
    "models": {
        "base_model": "stabilityai/stable-diffusion-2-1-base",
        "inpaint_model": "stabilityai/stable-diffusion-2-inpainting",
    },
    "input": {
        "source_image": "",
    },
    "source_view": {
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "fov_deg": 80.0,
        "width": 512,
        "height": 512,
    },
    "generation": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    },
    "inpaint": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "strength": 0.98,
    },
    "central_band": {
        "fov_deg": 80.0,
        "pitch_deg": 0.0,
        "yaw_stride_deg": 40.0,
        "steps_each_direction": 3,
        "merge_yaw_deg": 180.0,
        "sdedit_t0": 0.98,
        "stitch_pitch_min_deg": -40.0,
        "stitch_pitch_max_deg": 40.0,
        "risk_weights": [0.8, 0.2, 0.0, 0.0],
        "erase_ratio": 0.05,
        "risk_gaussian_kernel": 9,
        "risk_gaussian_sigma": 2.0,
        "mask_median_kernel": 5,
        "mask_dilate_kernel": 3,
        "mask_dilate_iterations": 1,
        "risk_fallback_threshold": 0.5,
    },
    "expansion": {
        "pitch_offset_deg": 25.0,
        "fov_deg": 110.0,
        "yaw_stride_deg": 80.0,
        "steps_per_direction": 3,
        "sdedit_t0": 0.84,
        "guidance_scale": 4.5,
        "noise_variance_multiplier": 1.00,
        "risk_weights": [0.6, 0.2, 0.1, 0.1],
        "erase_ratio": 0.10,
        "prior_crop_ratio": 0.3333333333,
        "risk_gaussian_kernel": 9,
        "risk_gaussian_sigma": 2.0,
        "mask_median_kernel": 5,
        "mask_dilate_kernel": 3,
        "mask_dilate_iterations": 1,
        "risk_fallback_threshold": 0.5,
    },
    "pole_closure": {
        "fov_deg": 90.0,
        "sdedit_t0": 0.82,
        "guidance_scale": 3.5,
        "noise_variance_multiplier": 1.00,
        "risk_weights": [0.6, 0.2, 0.1, 0.1],
        "erase_ratio": 0.20,
        "prior_crop_ratio": 0.3333333333,
        "risk_gaussian_kernel": 9,
        "risk_gaussian_sigma": 2.0,
        "mask_median_kernel": 5,
        "mask_dilate_kernel": 3,
        "mask_dilate_iterations": 1,
        "risk_fallback_threshold": 0.5,
    },
    "output": {
        "run_dir": "outputs/pipeline/example_run",
        "pano_width": 4096,
        "pano_height": 2048,
        "debug": False,
    },
}


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_pipeline_config(config_path):
    with open(config_path, "rb") as handle:
        user_config = tomli.load(handle)
    config = deep_update(copy.deepcopy(_PIPELINE_DEFAULTS), user_config)
    if not config.get("prompt"):
        raise RuntimeError("`prompt` is required.")
    if not config["models"].get("base_model"):
        raise RuntimeError("`models.base_model` is required.")
    if not config["models"].get("inpaint_model"):
        raise RuntimeError("`models.inpaint_model` is required.")
    if config["central_band"]["steps_each_direction"] != 3:
        raise RuntimeError("`central_band.steps_each_direction` must be 3.")
    if config["expansion"]["steps_per_direction"] != 3:
        raise RuntimeError("`expansion.steps_per_direction` must be 3.")
    if len(config["central_band"]["risk_weights"]) != 4:
        raise RuntimeError("`central_band.risk_weights` must contain four values.")
    if len(config["expansion"]["risk_weights"]) != 4:
        raise RuntimeError("`expansion.risk_weights` must contain four values.")
    if len(config["pole_closure"]["risk_weights"]) != 4:
        raise RuntimeError("`pole_closure.risk_weights` must contain four values.")
    return config


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


def make_torch_generator(torch, device, seed):
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))


def should_use_fp16_variant(model_name):
    lowered = model_name.lower()
    legacy_markers = [
        "stable-diffusion-v1",
        "stable-diffusion-1",
        "stable-diffusion-v1-5",
        "stable-diffusion-1-5",
        "sd-v1",
        "sd1.5",
    ]
    for marker in legacy_markers:
        if marker in lowered:
            return False
    return True


def load_pipeline(cache_key, pipeline_cls, model_name):
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    load_kwargs = {
        "torch_dtype": dtype,
    }
    if should_use_fp16_variant(model_name):
        load_kwargs["variant"] = "fp16"
    pipe = pipeline_cls.from_pretrained(model_name, **load_kwargs)
    pipe = pipe.to(device)
    PIPELINE_CACHE[cache_key] = (pipe, torch, device)
    return PIPELINE_CACHE[cache_key]


def image_to_pil(image):
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image.astype(np.uint8))


def build_noise_latents(pipe, torch, image_shape, generator, device, dtype, variance_multiplier):
    if variance_multiplier is None or abs(float(variance_multiplier) - 1.0) <= 1e-8:
        return None
    height = image_shape[0]
    width = image_shape[1]
    scale_factor = getattr(pipe, "vae_scale_factor", 8)
    latent_channels = getattr(pipe.vae.config, "latent_channels", 4)
    shape = (1, latent_channels, height // scale_factor, width // scale_factor)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return latents * float(variance_multiplier) ** 0.5


def generate_initial_view(config, generator=None):
    prompt = config["prompt"]
    generation_config = config["generation"]
    source_view = config["source_view"]
    pipe, torch, device = load_pipeline(("base", config["models"]["base_model"]), diffusers.StableDiffusionPipeline, config["models"]["base_model"])
    if generator is None:
        generator = make_torch_generator(torch, device, config.get("seed"))
    result = pipe(
        prompt=prompt,
        height=source_view["height"],
        width=source_view["width"],
        num_inference_steps=generation_config.get("num_inference_steps", 50),
        guidance_scale=generation_config.get("guidance_scale", 7.5),
        generator=generator,
    )
    return np.array(result.images[0].convert("RGB"), dtype=np.uint8)


def blend_guidance_image(init_image, mask_image, guidance_image):
    result = init_image.copy()
    if guidance_image.shape[:2] != init_image.shape[:2]:
        guidance_image = np.array(
            Image.fromarray(guidance_image.astype(np.uint8)).resize(
                (init_image.shape[1], init_image.shape[0]),
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
    mask = (mask_image > 0)[..., None].repeat(3, axis=-1)
    result[mask] = guidance_image[mask]
    return result


def run_guided_inpaint(prompt, init_image, mask_image, guidance_image, config, generator=None, strength=None, guidance_scale=None, noise_variance_multiplier=None):
    inpaint_config = config["inpaint"]
    pipe, torch, device = load_pipeline(("inpaint", config["models"]["inpaint_model"]), diffusers.StableDiffusionInpaintPipeline, config["models"]["inpaint_model"])
    if generator is None:
        generator = make_torch_generator(torch, device, config.get("seed"))

    image = init_image
    if guidance_image is not None:
        image = blend_guidance_image(init_image, mask_image, guidance_image)

    if strength is None:
        strength = inpaint_config.get("strength", 1.0)
    if guidance_scale is None:
        guidance_scale = inpaint_config.get("guidance_scale", 7.5)

    latents = build_noise_latents(
        pipe,
        torch,
        image.shape,
        generator,
        device,
        pipe.unet.dtype,
        noise_variance_multiplier,
    )

    result = pipe(
        prompt=prompt,
        image=image_to_pil(image),
        mask_image=image_to_pil(mask_image),
        num_inference_steps=inpaint_config.get("num_inference_steps", 50),
        guidance_scale=guidance_scale,
        generator=generator,
        strength=strength,
        latents=latents,
    )
    return image, np.array(result.images[0].convert("RGB"), dtype=np.uint8)


def load_source_image(path, expected_size):
    image = Image.open(path).convert("RGB")
    if image.size != expected_size:
        image = image.resize(expected_size, resample=Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def build_intrinsics(width, height, fov_deg):
    fov_rad = math.radians(fov_deg)
    focal = width / (2.0 * math.tan(fov_rad / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_view_rotation(view):
    yaw_rad = math.radians(view["yaw_deg"])
    pitch_rad = math.radians(view["pitch_deg"])
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    cx = math.cos(-pitch_rad)
    sx = math.sin(-pitch_rad)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return rot_y @ rot_x


def pixels_to_camera_rays(width, height, intrinsics):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    ones = np.ones_like(xs)
    pixels = np.stack([xs, ys, ones], axis=-1)
    inv_intrinsics = np.linalg.inv(intrinsics)
    rays = pixels @ inv_intrinsics.T
    rays[..., 1] *= -1.0
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return rays / norms


def world_rays_to_equirectangular(world_rays, pano_width, pano_height):
    x = world_rays[..., 0]
    y = world_rays[..., 1]
    z = world_rays[..., 2]
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(np.clip(y, -1.0, 1.0))
    u = ((yaw / (2.0 * math.pi)) + 0.5) * (pano_width - 1)
    v = (0.5 - pitch / math.pi) * (pano_height - 1)
    return u, v


def equirectangular_to_world_rays(pano_width, pano_height):
    xs = np.arange(pano_width, dtype=np.float32)
    ys = np.arange(pano_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    yaw = ((grid_x / (pano_width - 1.0)) - 0.5) * (2.0 * math.pi)
    pitch = (0.5 - (grid_y / (pano_height - 1.0))) * math.pi
    cos_pitch = np.cos(pitch)
    x = cos_pitch * np.sin(yaw)
    y = np.sin(pitch)
    z = cos_pitch * np.cos(yaw)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def project_perspective_to_equirectangular(image, view, pano_width, pano_height):
    intrinsics = build_intrinsics(view["width"], view["height"], view["fov_deg"])
    rotation = build_view_rotation(view)
    rays = pixels_to_camera_rays(view["width"], view["height"], intrinsics)
    world_rays = rays @ rotation.T
    u, v = world_rays_to_equirectangular(world_rays, pano_width, pano_height)
    canvas = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    mask = np.zeros((pano_height, pano_width), dtype=np.uint8)
    ui = np.clip(np.rint(u).astype(np.int32), 0, pano_width - 1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, pano_height - 1)
    canvas[vi, ui] = image
    mask[vi, ui] = 255
    return canvas, mask


def view_center_on_equirectangular(view, pano_width, pano_height):
    rotation = build_view_rotation(view)
    center_ray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    world_ray = rotation @ center_ray
    u, v = world_rays_to_equirectangular(world_ray.reshape(1, 1, 3), pano_width, pano_height)
    return int(round(float(u[0, 0]))), int(round(float(v[0, 0])))


def build_view_homography(source_view, target_view):
    source_intrinsics = build_intrinsics(source_view["width"], source_view["height"], source_view["fov_deg"])
    target_intrinsics = build_intrinsics(target_view["width"], target_view["height"], target_view["fov_deg"])
    source_rotation = build_view_rotation(source_view)
    target_rotation = build_view_rotation(target_view)
    homography = source_intrinsics @ source_rotation.T @ target_rotation @ np.linalg.inv(target_intrinsics)
    homography /= homography[2, 2]
    return homography.astype(np.float32)


def warp_image_and_mask(image, homography, out_size):
    width, height = out_size
    warped = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
    known_mask = cv2.warpPerspective(
        valid,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    known_mask = np.where(known_mask > 0, 255, 0).astype(np.uint8)
    missing_mask = np.where(known_mask > 0, 0, 255).astype(np.uint8)
    return warped, known_mask, missing_mask


def build_view_to_view_remap(source_view, target_view):
    source_intrinsics = build_intrinsics(source_view["width"], source_view["height"], source_view["fov_deg"])
    target_intrinsics = build_intrinsics(target_view["width"], target_view["height"], target_view["fov_deg"])
    source_rotation = build_view_rotation(source_view)
    target_rotation = build_view_rotation(target_view)
    target_camera_rays = pixels_to_camera_rays(target_view["width"], target_view["height"], target_intrinsics)
    world_rays = target_camera_rays @ target_rotation.T
    source_camera_rays = world_rays @ source_rotation
    z = source_camera_rays[..., 2]
    projected = source_camera_rays @ source_intrinsics.T
    map_x = projected[..., 0] / np.clip(z, 1e-8, None)
    map_y = projected[..., 1] / np.clip(z, 1e-8, None)
    map_y = (source_view["height"] - 1.0) - map_y
    valid = (
            (z > 0.0)
            & (map_x >= 0.0)
            & (map_x <= source_view["width"] - 1.0)
            & (map_y >= 0.0)
            & (map_y <= source_view["height"] - 1.0)
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def remap_with_visibility(values, map_x, map_y, valid, interpolation):
    remapped = cv2.remap(
        values,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if remapped.ndim == 3:
        return np.where(valid[..., None], remapped, 0)
    return np.where(valid, remapped, 0)


def build_boundary_weight_map(width, height):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    weight = np.minimum.reduce([grid_x + 1.0, grid_y + 1.0, width - grid_x, height - grid_y])
    weight = np.maximum(weight, 0.0)
    return weight


def sample_view_to_equirectangular(image, view, pano_width, pano_height):
    world_rays = equirectangular_to_world_rays(pano_width, pano_height)
    rotation = build_view_rotation(view)
    camera_rays = world_rays @ rotation
    z = camera_rays[..., 2]
    intrinsics = build_intrinsics(view["width"], view["height"], view["fov_deg"])
    projected = camera_rays @ intrinsics.T
    x = projected[..., 0] / np.clip(z, 1e-8, None)
    y = projected[..., 1] / np.clip(z, 1e-8, None)
    y = (view["height"] - 1.0) - y
    valid = (
            (z > 0.0)
            & (x >= 0.0)
            & (x <= view["width"] - 1.0)
            & (y >= 0.0)
            & (y <= view["height"] - 1.0)
    )
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)
    sampled = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    boundary_weight = build_boundary_weight_map(view["width"], view["height"])
    sampled_weight = cv2.remap(
        boundary_weight,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_weight = np.where(valid, sampled_weight, 0.0).astype(np.float32)
    return sampled, valid.astype(np.uint8) * 255, sampled_weight


def stitch_equirectangular_views(view_records, pano_width, pano_height, pitch_min_deg, pitch_max_deg):
    canvas_sum = np.zeros((pano_height, pano_width, 3), dtype=np.float32)
    weight_sum = np.zeros((pano_height, pano_width), dtype=np.float32)
    coverage = np.zeros((pano_height, pano_width), dtype=np.uint8)
    row_mask = np.zeros((pano_height, pano_width), dtype=bool)
    if pitch_min_deg is None or pitch_max_deg is None:
        row_mask[:, :] = True
    else:
        pitch_min_v = int(round((0.5 - pitch_max_deg / 180.0) * (pano_height - 1)))
        pitch_max_v = int(round((0.5 - pitch_min_deg / 180.0) * (pano_height - 1)))
        row_mask[max(pitch_min_v, 0):min(pitch_max_v + 1, pano_height), :] = True
    for record in view_records:
        sampled, valid_mask, sampled_weight = sample_view_to_equirectangular(
            record["image"],
            record["view"],
            pano_width,
            pano_height,
        )
        valid = (valid_mask > 0) & row_mask & (sampled_weight > 0.0)
        coverage = np.where(valid, 255, coverage).astype(np.uint8)
        weight = np.where(valid, sampled_weight, 0.0)
        canvas_sum += sampled.astype(np.float32) * weight[..., None]
        weight_sum += weight
        record["stitched_valid_mask"] = np.where(valid, 255, 0).astype(np.uint8)
        record["stitched_weight_map"] = sampled_weight
    canvas = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    nonzero = weight_sum > 0.0
    canvas[nonzero] = np.clip(canvas_sum[nonzero] / weight_sum[nonzero, None], 0.0, 255.0).astype(np.uint8)
    return canvas, coverage


def normalize_map(values):
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def compute_view_panorama_coords(view, pano_width, pano_height):
    intrinsics = build_intrinsics(view["width"], view["height"], view["fov_deg"])
    rotation = build_view_rotation(view)
    rays = pixels_to_camera_rays(view["width"], view["height"], intrinsics)
    world_rays = rays @ rotation.T
    return world_rays_to_equirectangular(world_rays, pano_width, pano_height)


def compute_distance_risk(view, pano_width, pano_height, initial_center):
    u, v = compute_view_panorama_coords(view, pano_width, pano_height)
    du = np.abs(u - float(initial_center[0]))
    du = np.minimum(du, pano_width - du)
    dv = np.abs(v - float(initial_center[1]))
    dx = du / max(float(pano_width - 1), 1.0)
    dy = dv / max(float(pano_height - 1), 1.0)
    distance = np.sqrt(dx * dx + dy * dy)
    return normalize_map(distance)


def compute_edge_risk(view, kernel, sigma):
    xs = np.arange(view["width"], dtype=np.float32)
    ys = np.arange(view["height"], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance_to_edge = np.minimum.reduce(
        [
            grid_x,
            grid_y,
            (view["width"] - 1.0) - grid_x,
            (view["height"] - 1.0) - grid_y,
        ]
    )
    distance_to_edge = normalize_map(distance_to_edge)
    edge_risk = 1.0 - distance_to_edge
    return cv2.GaussianBlur(edge_risk.astype(np.float32), (kernel, kernel), sigma)


def compute_gradient_magnitude(image):
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def compute_view_risk_maps(record, context_records, config, initial_center, risk_config):
    kernel = risk_config["risk_gaussian_kernel"]
    sigma = risk_config["risk_gaussian_sigma"]
    pano_width = config["output"]["pano_width"]
    pano_height = config["output"]["pano_height"]
    context_images = [item["image"] for item in context_records]
    distance = compute_distance_risk(record["view"], pano_width, pano_height, initial_center)
    edge = compute_edge_risk(record["view"], kernel, sigma)
    if len(context_images) <= 1:
        color = np.zeros((record["image"].shape[0], record["image"].shape[1]), dtype=np.float32)
        smoothness = np.zeros((record["image"].shape[0], record["image"].shape[1]), dtype=np.float32)
    else:
        row_mean_color = np.stack([image.astype(np.float32) for image in context_images], axis=0).mean(axis=0).mean(axis=1)
        color = record["image"].astype(np.float32) - row_mean_color[:, None, :]
        color = normalize_map(np.sqrt((color * color).sum(axis=2)))
        color = cv2.GaussianBlur(color.astype(np.float32), (kernel, kernel), sigma)

        current_gradient = compute_gradient_magnitude(record["image"])
        row_mean_gradient = np.stack([compute_gradient_magnitude(image) for image in context_images], axis=0).mean(axis=0).mean(axis=1)
        smoothness = normalize_map(np.abs(current_gradient - row_mean_gradient[:, None]))
        smoothness = cv2.GaussianBlur(smoothness.astype(np.float32), (kernel, kernel), sigma)
    combined = normalize_map(
        distance * float(risk_config["risk_weights"][0])
        + edge * float(risk_config["risk_weights"][1])
        + color * float(risk_config["risk_weights"][2])
        + smoothness * float(risk_config["risk_weights"][3])
    )
    risk_maps = {
        "distance": distance,
        "edge": edge,
        "color": color,
        "smoothness": smoothness,
        "combined": combined,
    }
    return risk_maps


def warp_risk_map(risk_map, homography, out_size):
    width, height = out_size
    return cv2.warpPerspective(
        risk_map.astype(np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def merge_warped_risks(left_risk, right_risk, left_valid_mask, right_valid_mask):
    left_valid = left_valid_mask > 0
    right_valid = right_valid_mask > 0
    merged = np.zeros_like(left_risk, dtype=np.float32)
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    both = left_valid & right_valid
    merged[left_only] = left_risk[left_only]
    merged[right_only] = right_risk[right_only]
    merged[both] = np.maximum(left_risk[both], right_risk[both])
    return merged


def select_risky_known_pixels(warped_risk, known_mask, erase_ratio, fallback_threshold):
    valid = known_mask > 0
    selected = np.zeros_like(known_mask, dtype=np.uint8)
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return selected
    erase_count = int(math.ceil(valid_count * float(erase_ratio)))
    if erase_count > 0:
        values = warped_risk[valid]
        erase_count = min(erase_count, values.size)
        if erase_count > 0:
            threshold = np.partition(values, values.size - erase_count)[values.size - erase_count]
            selected[(valid) & (warped_risk >= threshold)] = 255
            return selected
    selected[(valid) & (warped_risk >= float(fallback_threshold))] = 255
    return selected


def smooth_remask(mask, gaussian_kernel, gaussian_sigma, median_kernel, dilate_kernel, dilate_iterations):

    median = cv2.medianBlur(mask.astype(np.uint8), median_kernel)
    blurred = cv2.GaussianBlur(median.astype(np.float32), (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    binary = np.where(blurred >= 127.5, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    return cv2.dilate(binary, kernel, iterations=dilate_iterations)


def build_equirectangular_debug(image, view, pano_width, pano_height, point_color):
    canvas, _ = project_perspective_to_equirectangular(image, view, pano_width, pano_height)
    overlay = Image.fromarray(canvas, mode="RGB")
    drawer = ImageDraw.Draw(overlay)
    center_x, center_y = view_center_on_equirectangular(view, pano_width, pano_height)
    radius = 10
    drawer.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=point_color,
        width=3,
    )
    return np.array(overlay, dtype=np.uint8)


def save_optional_image(record, step_dir, filename, key, scale=1.0):
    if key not in record:
        return
    image = record[key]
    if abs(float(scale) - 1.0) > 1e-8:
        image = image * float(scale)
    save_image(os.path.join(step_dir, filename), image)


def _save_group_artifacts(group_dir, records, pano_width, pano_height):
    os.makedirs(group_dir, exist_ok=True)
    contact_images = [Image.fromarray(record["image"], mode="RGB") for record in records]
    contact_labels = [record["name"] for record in records]
    build_contact_sheet(contact_images, contact_labels).save(os.path.join(group_dir, "00_contact_sheet.png"))
    for index, record in enumerate(records):
        step_dir = os.path.join(group_dir, "{:02d}_{}".format(index, record["name"]))
        os.makedirs(step_dir, exist_ok=True)
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
        save_optional_image(record, step_dir, "02_guidance.png", "guidance_image")
        save_optional_image(record, step_dir, "03_guided_input.png", "guided_input")
        save_optional_image(record, step_dir, "04_stitched_valid_mask.png", "stitched_valid_mask")
        if "stitched_weight_map" in record:
            save_image(
                os.path.join(step_dir, "05_stitched_weight_map.png"),
                np.clip(record["stitched_weight_map"], 0.0, 255.0).astype(np.uint8),
            )
        save_image(
            os.path.join(step_dir, "06_on_equirect.png"),
            build_equirectangular_debug(record["image"], record["view"], pano_width, pano_height, (80, 220, 255)),
        )
        if "risk_maps" in record:
            for index_name, key in [
                ("07", "distance"),
                ("08", "edge"),
                ("09", "color"),
                ("10", "smoothness"),
                ("11", "combined"),
            ]:
                save_image(
                    os.path.join(step_dir, "{}_risk_{}.png".format(index_name, key)),
                    record["risk_maps"][key] * 255.0,
                )
        for filename, key, scale in [
            ("12_base_missing_mask.png", "base_missing_mask", 1.0),
            ("13_warped_combined_risk.png", "warped_combined_risk", 255.0),
            ("14_risk_selected_mask.png", "risk_selected_mask", 1.0),
            ("15_remasked_mask.png", "remasked_mask", 1.0),
            ("16_smoothed_mask.png", "smoothed_mask", 1.0),
            ("17_warped.png", "warped", 1.0),
            ("18_known_mask.png", "known_mask", 1.0),
            ("19_left_warped.png", "left_warped", 1.0),
            ("20_right_warped.png", "right_warped", 1.0),
            ("21_left_known_mask.png", "left_known_mask", 1.0),
            ("22_right_known_mask.png", "right_known_mask", 1.0),
            ("23_merge_composite.png", "merge_composite", 1.0),
        ]:
            save_optional_image(record, step_dir, filename, key, scale=scale)
        for source_index, source in enumerate(record.get("overlap_sources", [])):
            prefix = "{:02d}_{}".format(source_index, source["name"])
            save_image(os.path.join(step_dir, "24_{}_warped.png".format(prefix)), source["warped"])
            save_image(os.path.join(step_dir, "25_{}_known_mask.png".format(prefix)), source["known_mask"])
            save_image(
                os.path.join(step_dir, "26_{}_weight.png".format(prefix)),
                np.clip(source["weight"], 0.0, 255.0).astype(np.uint8),
            )


def _save_pipeline_outputs_minimal(run_dir, artifacts):
    os.makedirs(run_dir, exist_ok=True)
    save_image(os.path.join(run_dir, "02_initial_view.png"), artifacts["initial_view"])
    save_image(os.path.join(run_dir, "06_central_360_equirect.png"), artifacts["central_panorama"])
    save_image(os.path.join(run_dir, "12_full_sphere_equirect.png"), artifacts["full_panorama"])


def _save_pipeline_outputs_full(run_dir, artifacts):
    os.makedirs(run_dir, exist_ok=True)
    for filename, key in [
        ("00_config.json", "config"),
        ("03_central_manifest.json", "central_manifest"),
        ("04_upward_manifest.json", "upward_manifest"),
        ("05_downward_manifest.json", "downward_manifest"),
        ("14_pole_manifest.json", "pole_manifest"),
    ]:
        save_json(os.path.join(run_dir, filename), artifacts[key])
    for filename, key in [
        ("01_prompt.txt", "prompt"),
        ("15_pipeline_note.txt", "pipeline_note"),
    ]:
        save_text(os.path.join(run_dir, filename), artifacts[key])
    for filename, key in [
        ("02_initial_view.png", "initial_view"),
        ("06_central_360_equirect.png", "central_panorama"),
        ("07_upward_partial_equirect.png", "upward_panorama"),
        ("08_downward_partial_equirect.png", "downward_panorama"),
        ("09_full_sphere_without_poles_equirect.png", "pre_pole_panorama"),
        ("10_top_pole_partial_equirect.png", "top_pole_panorama"),
        ("11_bottom_pole_partial_equirect.png", "bottom_pole_panorama"),
        ("12_full_sphere_equirect.png", "full_panorama"),
        ("13_full_sphere_coverage.png", "full_coverage"),
    ]:
        save_image(os.path.join(run_dir, filename), artifacts[key])

    contact_images = []
    contact_labels = []
    for record in artifacts["central_records"] + artifacts["upward_records"] + artifacts["downward_records"] + artifacts["pole_records"]:
        contact_images.append(Image.fromarray(record["image"], mode="RGB"))
        contact_labels.append(record["name"])
    build_contact_sheet(contact_images, contact_labels).save(os.path.join(run_dir, "16_all_view_contact_sheet.png"))

    for folder, key in [
        ("central", "central_records"),
        ("upward", "upward_records"),
        ("downward", "downward_records"),
        ("poles", "pole_records"),
    ]:
        _save_group_artifacts(
            os.path.join(run_dir, folder),
            artifacts[key],
            artifacts["pano_width"],
            artifacts["pano_height"],
        )


def save_pipeline_outputs(run_dir, artifacts, debug):
    if debug:
        _save_pipeline_outputs_full(run_dir, artifacts)
        return
    _save_pipeline_outputs_minimal(run_dir, artifacts)


