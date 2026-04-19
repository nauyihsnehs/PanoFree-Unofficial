import math

import numpy as np

from .camera import build_intrinsics, build_view_rotation, camera_rays_to_world, pixels_to_camera_rays, world_rays_to_equirectangular


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for risky-area estimation. Install `opencv-python` before running Phase 3."
        ) from exc
    return cv2


def normalize_map(values):
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def smooth_risk_map(values, kernel, sigma):
    cv2 = require_cv2()
    return cv2.GaussianBlur(values.astype(np.float32), (kernel, kernel), sigma)


def compute_view_panorama_coords(view, pano_width, pano_height):
    intrinsics = build_intrinsics(view["width"], view["height"], view["fov_deg"])
    rotation = build_view_rotation(view)
    rays = pixels_to_camera_rays(view["width"], view["height"], intrinsics)
    world_rays = camera_rays_to_world(rays, rotation)
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
    return smooth_risk_map(edge_risk, kernel, sigma)


def compute_gradient_magnitude(image):
    cv2 = require_cv2()
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def build_row_mean_color(context_images):
    stacked = np.stack([image.astype(np.float32) for image in context_images], axis=0)
    return stacked.mean(axis=0).mean(axis=1)


def build_row_mean_gradient(context_images):
    gradients = np.stack([compute_gradient_magnitude(image) for image in context_images], axis=0)
    return gradients.mean(axis=0).mean(axis=1)


def compute_color_risk(image, context_images, kernel, sigma):
    if len(context_images) <= 1:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    row_mean = build_row_mean_color(context_images)
    diff = image.astype(np.float32) - row_mean[:, None, :]
    abruptness = np.sqrt((diff * diff).sum(axis=2))
    abruptness = normalize_map(abruptness)
    return smooth_risk_map(abruptness, kernel, sigma)


def compute_smoothness_risk(image, context_images, kernel, sigma):
    if len(context_images) <= 1:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    current_gradient = compute_gradient_magnitude(image)
    row_mean = build_row_mean_gradient(context_images)
    abruptness = np.abs(current_gradient - row_mean[:, None])
    abruptness = normalize_map(abruptness)
    return smooth_risk_map(abruptness, kernel, sigma)


def compute_combined_risk(risk_maps, weights):
    combined = (
        risk_maps["distance"] * float(weights[0])
        + risk_maps["edge"] * float(weights[1])
        + risk_maps["color"] * float(weights[2])
        + risk_maps["smoothness"] * float(weights[3])
    )
    return normalize_map(combined)


def compute_view_risk_maps(record, context_records, config, initial_center, risk_config=None):
    if risk_config is None:
        risk_config = config["central_band"]
    kernel = risk_config["risk_gaussian_kernel"]
    sigma = risk_config["risk_gaussian_sigma"]
    pano_width = config["output"]["pano_width"]
    pano_height = config["output"]["pano_height"]
    context_images = [item["image"] for item in context_records]

    risk_maps = {
        "distance": compute_distance_risk(record["view"], pano_width, pano_height, initial_center),
        "edge": compute_edge_risk(record["view"], kernel, sigma),
        "color": compute_color_risk(record["image"], context_images, kernel, sigma),
        "smoothness": compute_smoothness_risk(record["image"], context_images, kernel, sigma),
    }
    risk_maps["combined"] = compute_combined_risk(
        risk_maps,
        risk_config["risk_weights"],
    )
    return risk_maps


def warp_risk_map(risk_map, homography, out_size):
    cv2 = require_cv2()
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
    cv2 = require_cv2()
    median = cv2.medianBlur(mask.astype(np.uint8), median_kernel)
    blurred = cv2.GaussianBlur(median.astype(np.float32), (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    binary = np.where(blurred >= 127.5, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    return cv2.dilate(binary, kernel, iterations=dilate_iterations)
