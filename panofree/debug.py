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

