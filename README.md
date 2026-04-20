# PanoFree-Unofficial

Unofficial implementation of

**PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance**

- Paper: [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3887_ECCV_2024_paper.php)
- Project page: [panofree.github.io](https://panofree.github.io/)
- Authors: Aoming Liu, Zhong Li, Zhang Chen, Nannan Li, Yi Xu, Bryan A. Plummer

This repo currently implements the **full spherical panorama** pipeline.

Known differences:

- **Planar** panorama generation is not implemented yet.
- No experiments are implemented yet.
- A more paper-aligned reference config is provided in [`pipeline-paper.toml`](pipeline-paper.toml).
- Some low-level details are conservative engineering completions where the paper is underspecified, such as exact smoothing and mask post-processing operators.

### Example Outputs

| Initial View                                                   | Central Band                                                           | Pre-Pole Sphere                                                                           | Final Sphere                                                            |
|----------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| ![Initial view](./outputs/20260420_110318/02_initial_view.png) | ![Central band](./outputs/20260420_110318/06_central_360_equirect.png) | ![Pre-pole panorama](./outputs/20260420_110318/09_full_sphere_without_poles_equirect.png) | ![Full panorama](./outputs/20260420_110318/12_full_sphere_equirect.png) |

## File And Directory Overview

```text
.
├── outputs/                        # Saved runs and debug artifacts
├── pipeline.py                     # Main orchestration for the full panorama pipeline
├── pipeline_helper.py              # Model loading, geometry, warping, risk maps, stitching, saving
├── pipeline.toml                   # Default working config
├── pipeline-paper.toml            # More paper-faithful config reference
```

Inside each debug run under `outputs/<timestamp>/`, the main files are:

- `00_config.json`: resolved runtime config
- `01_prompt.txt`: text prompt
- `02_initial_view.png`: initial generated perspective view
- `03_central_manifest.json`: scheduled/generated metadata for central-band views
- `04_upward_manifest.json`: scheduled/generated metadata for upward expansion views
- `05_downward_manifest.json`: scheduled/generated metadata for downward expansion views
- `06` to `13`: stitched panorama outputs
- `14_pole_manifest.json`: scheduled/generated metadata for pole-closure views
- `15_pipeline_note.txt`: success/failure note
- `16_all_view_contact_sheet.png`: all view thumbnails
- `central/`, `upward/`, `downward/`, `poles/`: per-step debug artifacts, each containing numbered step subfolders (e.g., `00_x0/`, `01_x1/`)

## Requirements

```text
python==3.10.20
numpy==1.26.4
Pillow==9.5.0
opencv-python==4.10.0.84
torch==2.5.1+cu118
diffusers==0.35.1
tomli==2.4.1
transformers==4.47.1
accelerate==1.4.0
safetensors==0.5.3
```

## Inference

```bash
python pipeline.py [--config pipeline.toml]
```

## Configs

The pipeline reads a TOML config file. By default it uses `pipeline.toml`.

Top-level fields:

- `prompt`: the single global prompt used for the whole pipeline
- `seed`: random seed passed to the diffusion generator
- `models`: checkpoint selection for generation and inpainting
- `input`: optional source image input
- `source_view`: base camera definition shared by generated views
- `generation`: settings for the initial text-to-image view
- `inpaint`: shared default settings for inpainting steps
- `central_band`: schedule and remasking settings for the 360-degree center ring
- `expansion`: schedule and guidance settings for upward and downward expansion
- `pole_closure`: settings for top and bottom pole completion
- `output`: panorama size, output directory, and debug export flag

Field details:

- `models.base_model`: text-to-image checkpoint used for the initial center view
- `models.inpaint_model`: inpainting checkpoint used for all subsequent guided steps
- `input.source_image`: if non-empty, the pipeline loads and resizes this image as `x0` instead of sampling the initial view from text
- `source_view.fov_deg`: base field of view of the perspective view representation
- `source_view.width`, `source_view.height`: per-view resolution, currently expected to be `512x512` in the reproduction setup
- `generation.num_inference_steps`: diffusion steps for the initial center view
- `generation.guidance_scale`: CFG scale for the initial center view
- `inpaint.num_inference_steps`: diffusion steps for inpainting calls
- `inpaint.guidance_scale`: fallback CFG scale for inpainting if a stage does not override it
- `inpaint.strength`: fallback inpaint strength if a stage does not override it

Central band:

- `central_band.fov_deg`: FoV for center-band views
- `central_band.pitch_deg`: pitch of the central ring, normally `0.0`
- `central_band.yaw_stride_deg`: yaw increment between neighboring center-band views
- `central_band.steps_each_direction`: number of left/right steps; the current implementation requires `3`
- `central_band.merge_yaw_deg`: yaw of the loop-closure view, normally `180.0`
- `central_band.sdedit_t0`: effective SDEdit-style denoising strength used in central-band guided inpainting
- `central_band.stitch_pitch_min_deg`, `central_band.stitch_pitch_max_deg`: pitch range used when saving the stitched central panorama
- `central_band.risk_weights`: linear weights for distance, edge, color, and smoothness risks
- `central_band.erase_ratio`: fraction of currently known pixels to remask each step

Expansion:

- `expansion.pitch_offset_deg`: vertical offset from the central ring
- `expansion.fov_deg`: wider FoV for upper and lower views
- `expansion.yaw_stride_deg`: yaw spacing for expansion views
- `expansion.steps_per_direction`: currently required to be `3` by the loader
- `expansion.sdedit_t0`: denoising strength for upward/downward inpainting
- `expansion.guidance_scale`: stage-specific CFG scale override
- `expansion.noise_variance_multiplier`: optional extra latent variance injected before denoising
- `expansion.prior_crop_ratio`: crop ratio for the upper or lower prior image guidance
- `expansion.risk_weights`, `expansion.erase_ratio`: remasking policy for expansion stages

Pole closure:

- `pole_closure.fov_deg`: FoV for top and bottom pole views
- `pole_closure.sdedit_t0`: denoising strength for pole completion
- `pole_closure.guidance_scale`: stage-specific CFG scale override
- `pole_closure.noise_variance_multiplier`: optional latent variance multiplier for pole generation
- `pole_closure.prior_crop_ratio`: crop ratio used to build pole guidance images
- `pole_closure.risk_weights`, `pole_closure.erase_ratio`: remasking policy for pole stages

Shared risk-mask controls:

- `risk_gaussian_kernel`, `risk_gaussian_sigma`: smoothing used for risk-map computation
- `mask_median_kernel`: cleanup before final remask smoothing
- `mask_dilate_kernel`, `mask_dilate_iterations`: dilation applied to the binary remask
- `risk_fallback_threshold`: fallback threshold when percentile-based risky-pixel selection is not enough

Output:

- `output.run_dir`: parent directory for saved runs; the script appends a timestamped subdirectory automatically
- `output.pano_width`, `output.pano_height`: final equirectangular canvas size
- `output.debug`: when `true`, save full manifests, risk maps, masks, warped inputs, and per-step debug folders

Config choice:

- `pipeline.toml`: current working config used by default
- `pipeline-paper.toml`: more paper-aligned reference config

## Citation

```bibtex
@inproceedings{liu2024panofree,
  title={PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidance},
  author={Liu, Aoming and Li, Zhong and Chen, Zhang and Li, Nannan and Xu, Yi and Plummer, Bryan A.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
