import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image


def merge_dict(base, override):
    merged = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = merge_dict(value, {})
        else:
            merged[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def pil_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    return image


def numpy_to_pil(image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def mask_to_pil(mask):
    mask = np.clip(mask, 0.0, 1.0)
    return Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")


def ensure_odd(value):
    value = int(value)
    if value < 1:
        return 1
    if value % 2 == 0:
        value += 1
    return value


def refine_inpaint_mask(mask, dilate_px, blur_px):
    mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    dilate_px = ensure_odd(dilate_px)
    blur_px = ensure_odd(blur_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    blurred = cv2.GaussianBlur(dilated, (blur_px, blur_px), 0)
    return np.clip(blurred.astype(np.float32) / 255.0, 0.0, 1.0)


def save_image(path, image):
    numpy_to_pil(pil_to_numpy(image)).save(path)


def save_mask(path, mask):
    mask_to_pil(mask).save(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def prepare_output_dirs(config):
    run = config["run"]
    root = ensure_dir(run["output_root"])
    run_dir = ensure_dir(os.path.join(root, run["name"]))
    return {
        "run_dir": run_dir,
        "views": ensure_dir(os.path.join(run_dir, "views")),
        "warps": ensure_dir(os.path.join(run_dir, "warps")),
        "masks": ensure_dir(os.path.join(run_dir, "masks")),
        "panorama_steps": ensure_dir(os.path.join(run_dir, "panorama_steps")),
    }


def dump_run_manifest(config, output_dirs):
    manifest = dict(config)
    manifest_path = os.path.join(output_dirs["run_dir"], "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_yaw_schedule(config):
    panorama = config["panorama"]
    start = panorama["yaw_start_deg"]
    step = panorama["yaw_step_deg"]
    count = panorama["num_views"]
    return [start + index * step for index in range(count)]


def make_camera_rays(size, hfov_deg):
    half = math.tan(math.radians(hfov_deg) * 0.5)
    axis = np.linspace(-half, half, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, -axis)
    zz = np.ones_like(xx)
    rays = np.stack([xx, yy, zz], axis=-1)
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / np.maximum(norm, 1e-8)


def rotation_matrix(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    yaw_matrix = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    return yaw_matrix @ pitch_matrix


def rays_world(rays, yaw_deg, pitch_deg):
    rotation = rotation_matrix(yaw_deg, pitch_deg)
    return np.einsum("ij,hwj->hwi", rotation, rays)


def world_to_equirectangular(rays_world_value, width, height):
    x = rays_world_value[..., 0]
    y = np.clip(rays_world_value[..., 1], -1.0, 1.0)
    z = rays_world_value[..., 2]
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(y)

    u = ((yaw / (2.0 * math.pi)) + 0.5) * width
    v = (0.5 - pitch / math.pi) * height

    u = np.mod(u, width).astype(np.float32)
    v = np.clip(v, 0.0, height - 1.0).astype(np.float32)
    return u, v


def project_view_to_panorama(view, yaw_deg, pitch_deg, panorama_width, panorama_height, hfov_deg):
    size = view.shape[0]
    rays = make_camera_rays(size, hfov_deg)
    world_rays = rays_world(rays, yaw_deg, pitch_deg)
    map_x, map_y = world_to_equirectangular(world_rays, panorama_width, panorama_height)

    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.float32)
    weight = np.zeros((panorama_height, panorama_width), dtype=np.float32)

    flat_x = np.rint(map_x.reshape(-1)).astype(np.int32) % panorama_width
    flat_y = np.rint(map_y.reshape(-1)).astype(np.int32)
    flat_pixels = view.reshape(-1, 3).astype(np.float32)

    np.add.at(panorama, (flat_y, flat_x), flat_pixels)
    np.add.at(weight, (flat_y, flat_x), 1.0)

    valid = weight > 0
    panorama[valid] = panorama[valid] / weight[valid, None]
    return panorama, valid.astype(np.float32)


def warp_view_to_view(source_view, source_yaw_deg, source_pitch_deg, target_yaw_deg, target_pitch_deg, hfov_deg):
    size = source_view.shape[0]
    rays = make_camera_rays(size, hfov_deg)

    target_world = rays_world(rays, target_yaw_deg, target_pitch_deg)
    source_rotation = rotation_matrix(source_yaw_deg, source_pitch_deg)
    source_local = np.einsum("ji,hwj->hwi", source_rotation, target_world)

    z = source_local[..., 2]
    half = math.tan(math.radians(hfov_deg) * 0.5)

    x = source_local[..., 0] / np.maximum(z, 1e-8)
    y = source_local[..., 1] / np.maximum(z, 1e-8)

    valid = (z > 0.0) & (np.abs(x) <= half) & (np.abs(y) <= half)

    map_x = ((x / half) + 1.0) * 0.5 * (size - 1)
    map_y = ((-y / half) + 1.0) * 0.5 * (size - 1)

    warped = cv2.remap(
        source_view,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, valid.astype(np.float32)
