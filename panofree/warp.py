import numpy as np

from .camera import build_intrinsics, build_view_rotation


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
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for warping. Install `opencv-python` before running Phase 1."
        ) from exc

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

