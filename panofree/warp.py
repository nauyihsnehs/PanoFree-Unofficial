import numpy as np

from .camera import build_intrinsics, build_view_rotation, camera_rays_to_world, pixels_to_camera_rays


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for warping. Install `opencv-python` before running Phase 1."
        ) from exc
    return cv2


def build_view_homography(source_view, target_view):
    source_intrinsics = build_intrinsics(
        source_view["width"],
        source_view["height"],
        source_view["fov_deg"],
    )
    target_intrinsics = build_intrinsics(
        target_view["width"],
        target_view["height"],
        target_view["fov_deg"],
    )
    source_rotation = build_view_rotation(source_view)
    target_rotation = build_view_rotation(target_view)

    homography = (
        source_intrinsics
        @ source_rotation.T
        @ target_rotation
        @ np.linalg.inv(target_intrinsics)
    )
    homography /= homography[2, 2]
    return homography.astype(np.float32)


def warp_image_and_mask(image, homography, out_size):
    cv2 = require_cv2()

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
    source_intrinsics = build_intrinsics(
        source_view["width"],
        source_view["height"],
        source_view["fov_deg"],
    )
    target_intrinsics = build_intrinsics(
        target_view["width"],
        target_view["height"],
        target_view["fov_deg"],
    )
    source_rotation = build_view_rotation(source_view)
    target_rotation = build_view_rotation(target_view)

    target_camera_rays = pixels_to_camera_rays(
        target_view["width"],
        target_view["height"],
        target_intrinsics,
    )
    world_rays = camera_rays_to_world(target_camera_rays, target_rotation)
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
    cv2 = require_cv2()
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

