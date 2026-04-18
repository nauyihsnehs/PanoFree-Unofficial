# PanoFree Unofficial

This repo is intentionally minimal for `M1` debugging.

## Files

- `pipeline.py`: entrypoint, config loading, model setup, main generation loop
- `pipeline_helper.py`: geometry, mask, image, and output helper functions
- `config.toml`: all runtime options in one place
- `environment.yml`: conda environment

## Current M1 scope

- text-only initialization
- vanilla sequential warping and inpainting
- center-band 360 panorama generation only

Not included yet:

- bidirectional generation
- cross-view guidance
- risky area erasing
- vertical expansion
- pole closure

## Environment

```powershell
conda env create -f environment.yml
conda activate panofree
```

## Run

Edit `config.toml`, then run:

```powershell
python pipeline.py --config config.toml
```

Outputs are written under `outputs/<run_name>/`.
