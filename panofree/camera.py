import math

import numpy as np


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


def rotation_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def rotation_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def build_view_rotation(view):
    yaw_rad = math.radians(view["yaw_deg"])
    pitch_rad = math.radians(view["pitch_deg"])
    return rotation_y(yaw_rad) @ rotation_x(-pitch_rad)


def pixel_grid(width, height):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    return np.meshgrid(xs, ys)


def pixels_to_camera_rays(width, height, intrinsics):
    xs, ys = pixel_grid(width, height)
    ones = np.ones_like(xs)
    pixels = np.stack([xs, ys, ones], axis=-1)
    inv_intrinsics = np.linalg.inv(intrinsics)
    rays = pixels @ inv_intrinsics.T
    rays[..., 1] *= -1.0
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return rays / norms


def camera_rays_to_world(rays, rotation):
    return rays @ rotation.T


def world_rays_to_equirectangular(world_rays, pano_width, pano_height):
    x = world_rays[..., 0]
    y = world_rays[..., 1]
    z = world_rays[..., 2]

    yaw = np.arctan2(x, z)
    pitch = np.arcsin(np.clip(y, -1.0, 1.0))

    u = ((yaw / (2.0 * math.pi)) + 0.5) * (pano_width - 1)
    v = (0.5 - pitch / math.pi) * (pano_height - 1)
    return u, v


def project_perspective_to_equirectangular(image, view, pano_width, pano_height):
    intrinsics = build_intrinsics(view["width"], view["height"], view["fov_deg"])
    rotation = build_view_rotation(view)
    rays = pixels_to_camera_rays(view["width"], view["height"], intrinsics)
    world_rays = camera_rays_to_world(rays, rotation)
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
    u, v = world_rays_to_equirectangular(
        world_ray.reshape(1, 1, 3),
        pano_width,
        pano_height,
    )
    return int(round(float(u[0, 0]))), int(round(float(v[0, 0])))

