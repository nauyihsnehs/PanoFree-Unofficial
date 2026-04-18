import numpy as np
from PIL import Image


PIPELINE_CACHE = {}


def require_module(name, install_hint):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(install_hint) from exc


def resolve_device():
    torch = require_module(
        "torch",
        "PyTorch is required for Phase 1. Install `torch` before running the pipeline.",
    )
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


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

    torch = require_module(
        "torch",
        "PyTorch is required for Phase 1. Install `torch` before running the pipeline.",
    )
    diffusers = require_module(
        "diffusers",
        "Diffusers is required for Phase 1. Install `diffusers` before running the pipeline.",
    )

    device, dtype = resolve_device()
    load_kwargs = {
        "torch_dtype": dtype,
    }
    if should_use_fp16_variant(model_name):
        load_kwargs["variant"] = "fp16"
    pipe = pipeline_cls.from_pretrained(
        model_name,
        **load_kwargs,
    )
    pipe = pipe.to(device)
    PIPELINE_CACHE[cache_key] = (pipe, torch, device)
    return PIPELINE_CACHE[cache_key]


def load_generation_pipeline(model_name):
    if not model_name:
        raise RuntimeError("`models.base_model` is required.")
    diffusers = require_module(
        "diffusers",
        "Diffusers is required for Phase 1. Install `diffusers` before running the pipeline.",
    )
    return load_pipeline(
        ("base", model_name),
        diffusers.StableDiffusionPipeline,
        model_name,
    )


def load_inpaint_pipeline(model_name):
    if not model_name:
        raise RuntimeError("`models.inpaint_model` is required.")
    diffusers = require_module(
        "diffusers",
        "Diffusers is required for Phase 1. Install `diffusers` before running the pipeline.",
    )
    return load_pipeline(
        ("inpaint", model_name),
        diffusers.StableDiffusionInpaintPipeline,
        model_name,
    )


def image_to_pil(image):
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image.astype(np.uint8))


def pil_to_array(image):
    return np.array(image.convert("RGB"), dtype=np.uint8)


def create_generator(seed, device=None):
    torch = require_module(
        "torch",
        "PyTorch is required for Phase 1. Install `torch` before running the pipeline.",
    )
    if device is None:
        device, _ = resolve_device()
    return make_torch_generator(torch, device, seed)


def generate_initial_view(config, generator=None):
    prompt = config["prompt"]
    generation_config = config["generation"]
    source_view = config["source_view"]
    pipe, torch, device = load_generation_pipeline(config["models"]["base_model"])
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
    return pil_to_array(result.images[0])


def run_inpaint(prompt, init_image, mask_image, config):
    return run_guided_inpaint(
        prompt,
        init_image,
        mask_image,
        None,
        config,
    )


def run_guided_inpaint(prompt, init_image, mask_image, guidance_image, config, generator=None, strength=None):
    inpaint_config = config["inpaint"]
    pipe, torch, device = load_inpaint_pipeline(config["models"]["inpaint_model"])
    if generator is None:
        generator = make_torch_generator(torch, device, config.get("seed"))

    image = init_image
    if guidance_image is not None:
        image = blend_guidance_image(init_image, mask_image, guidance_image)

    if strength is None:
        strength = inpaint_config.get("strength", 1.0)

    result = pipe(
        prompt=prompt,
        image=image_to_pil(image),
        mask_image=image_to_pil(mask_image),
        num_inference_steps=inpaint_config.get("num_inference_steps", 50),
        guidance_scale=inpaint_config.get("guidance_scale", 7.5),
        generator=generator,
        strength=strength,
    )
    return image, pil_to_array(result.images[0])


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
