import numpy as np

from .camera import build_intrinsics, build_view_rotation, equirectangular_to_world_rays


def build_boundary_weight_map(width, height):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    weight = np.minimum.reduce(
        [
            grid_x + 1.0,
            grid_y + 1.0,
            width - grid_x,
            height - grid_y,
        ]
    )
    weight = np.maximum(weight, 0.0)
    return weight


def sample_view_to_equirectangular(image, view, pano_width, pano_height):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for stitching. Install `opencv-python` before running Phase 2."
        ) from exc

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

    pitch_min_v = int(round((0.5 - pitch_max_deg / 180.0) * (pano_height - 1)))
    pitch_max_v = int(round((0.5 - pitch_min_deg / 180.0) * (pano_height - 1)))
    row_mask = np.zeros((pano_height, pano_width), dtype=bool)
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
    canvas[nonzero] = np.clip(
        canvas_sum[nonzero] / weight_sum[nonzero, None],
        0.0,
        255.0,
    ).astype(np.uint8)
    return canvas, coverage

