# CAPTCHA Lab

CAPTCHA Lab is a working repository for studying and operating a 3x3 image-based CAPTCHA solver.

The codebase contains two closely related layers:

1. A production-style recognition pipeline that detects a CAPTCHA grid, runs OCR across several preprocessing variants, fuses the candidate predictions, and selects the tiles that match the requested target number.
2. A larger automation and operations layer that wraps the solver in flows, logging, Django admin models, and scraper-related services.

This repository currently reflects a frozen production baseline. The trained fusion model is treated as a required artifact, not an optional convenience. The goal of this README is to explain what lives in the repository, how the solver works, how to run it, and where the important outputs are written.

## What This Project Does

At a high level, the solver:

1. loads a CAPTCHA screenshot;
2. finds the 3x3 tile grid in the image;
3. crops each tile;
4. applies a fixed set of preprocessing transformations to each tile;
5. runs EasyOCR over each transformed variant;
6. combines the OCR candidates using a trained fusion selector;
7. determines which tiles match the requested 3-digit target;
8. writes debug artifacts so the detection and OCR process can be inspected.

There are two target-extraction modes:

1. automatic mode, where the solver OCRs the instruction strip above the grid;
2. manual fallback mode, where the target is supplied explicitly on the command line.

## Repository Layout

The most relevant files and directories are:

- `captcha_solver.py`: the main CLI solver for a screenshot or image path
- `solver.py`: tile-level recognition logic and OCR/fusion helpers
- `target_extractor.py`: target-number extraction utilities
- `ocr.py`: EasyOCR reader setup
- `fusion.py`: fusion selector model loading and candidate aggregation
- `preprocess.py`: preprocessing helpers used by the recognition pipeline
- `extract_tiles.py`: grid detection, box selection, tile cropping, and debug overlays
- `main.py`: an earlier entry point that operates on `images/captcha.png`
- `BASELINE.md`: production baseline and frozen decision record
- `models/fusion_model.joblib`: the required trained fusion model
- `dataset/`: example CAPTCHA screenshots used for local testing
- `experiments/`: archived CNN and synthetic-data work
- `flows/`: browser and automation flows
- `operations/`: Django app code, admin templates, tasks, services, and migrations
- `scraper/`: scraper models and service code
- `control_panel/`: Django project settings, URLs, WSGI, ASGI, and Celery wiring

## Recommended Entry Point

For the current solver, the primary CLI entry point is `captcha_solver.py`.

Example:

```bash
python captcha_solver.py dataset/Screenshot-00023.png
```

This runs in automatic target mode. The solver will attempt to read the instruction strip above the detected grid and extract the 3-digit target from the screenshot itself.

If automatic target extraction is unreliable for a specific image, you can provide the target manually:

```bash
python captcha_solver.py dataset/Screenshot-00023.png 909
```

The manual target path is the safer fallback when the OCR text in the instruction strip is noisy, partially occluded, or otherwise ambiguous.

## What The Solver Writes

The solver writes its artifacts to `output/live_solver/`.

Typical files include:

- `output/live_solver/screenshot.png`
- `output/live_solver/instruction.png`
- `output/live_solver/detected.png`
- `output/live_solver/tile_1.png` through `output/live_solver/tile_9.png`
- `output/live_solver/decision.json`

These artifacts are intentionally verbose. They are meant to make the decision process inspectable, not just produce a binary answer.

`instruction.png` is especially useful when automatic target extraction fails. If the target cannot be read reliably, inspect that crop first.

## Typical Output

When automatic extraction succeeds, the CLI prints a line similar to:

```text
Detected target: 909 confidence=0.998 variant=upscale
```

The exact confidence and variant name depend on the OCR and preprocessing path selected for that image.

The output JSON records the final decision, the selected tiles, any uncertain tiles, and the per-tile recognition details.

## Recognition Pipeline

The production recognition path is intentionally conservative and model-driven.

### 1. Grid detection

The solver locates the square CAPTCHA grid in the screenshot and crops the 3x3 tile region.

### 2. Tile extraction

The grid is split into nine individual tile images.

### 3. Preprocessing

Each tile is transformed into multiple OCR-friendly variants. The current frozen model expects these variant names:

- `raw`
- `gray_clahe`
- `upscale_2x`
- `upscale_3x`
- `sharpened`
- `otsu`
- `adaptive_threshold`
- `saturation_mask`
- `lab_color_distance`

These variant names and transformations are part of the production baseline. They should not be changed casually, because the trained fusion model expects them.

### 4. OCR

EasyOCR is run across the preprocessing variants.

### 5. Fusion

The OCR candidates are passed into the trained fusion selector in `models/fusion_model.joblib`.

### 6. Tile decision

The solver determines whether each tile matches the requested 3-digit target and marks uncertain tiles separately.

## Baseline Policy

This repository includes a frozen baseline record in `BASELINE.md`.

Important baseline rules:

- the fusion model is required at runtime;
- there is no silent fallback to a weaker recognition strategy;
- preprocessing variants should not be renamed without retraining and validating the model;
- the archived CNN and synthetic-data experiments are preserved for reference only;
- future changes should be measured against the current frozen baseline rather than replacing it implicitly.

If you are changing recognition behavior, read `BASELINE.md` before modifying the solver.

## Dependencies

The runtime dependencies listed in this repository are:

- `easyocr`
- `opencv-python`
- `numpy`
- `gunicorn`

There is also a more complete frozen dependency set in `requirements-lock.txt`.

## Installation

The repository is set up as a Python project. A typical local workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are working on the frozen production path, make sure `models/fusion_model.joblib` is present before running the solver.

## Example Workflow

To test a screenshot locally:

```bash
python captcha_solver.py dataset/Screenshot-00023.png
```

If the target text is not recovered automatically:

```bash
python captcha_solver.py dataset/Screenshot-00023.png 909
```

Then inspect:

- `output/live_solver/instruction.png` to see the prompt crop
- `output/live_solver/detected.png` to inspect grid detection
- `output/live_solver/tile_*.png` to inspect the per-tile crops
- `output/live_solver/decision.json` to review the final structured decision

## Project History

This repository contains earlier experiments and support code that are no longer the main production path.

### Archived CNN work

The `experiments/cnn/` directory contains older CNN training and dataset code.

### Archived synthetic-data work

The `experiments/synthetic/` directory contains synthetic CAPTCHA generation, renderer experiments, glyph and style analysis, and prototype-building scripts.

### Automation and operations

The `flows/`, `operations/`, `scraper/`, and `control_panel/` directories support higher-level automation, orchestration, and admin tooling around the solver.

## Notes On `main.py`

`main.py` is an older image-processing entry point that works with `images/captcha.png` and writes intermediate artifacts to `output/`.

It remains in the repository as part of the broader codebase, but the current README focuses on `captcha_solver.py` because that is the clearest CLI for the frozen production pipeline.

## Practical Guidance

If you are extending this repository, keep these constraints in mind:

- prefer inspecting the actual image crops before changing OCR logic;
- do not change preprocessing names or shapes without checking the fusion model contract;
- preserve the distinction between automatic target extraction and manual fallback;
- keep debug artifacts verbose enough to diagnose failures without rerunning the full pipeline;
- treat the baseline document as the source of truth for what is frozen.

## Related Files

- [BASELINE.md](/Users/kiasmacbookair/Projects/captcha-lab/BASELINE.md)
- [captcha_solver.py](/Users/kiasmacbookair/Projects/captcha-lab/captcha_solver.py)
- [main.py](/Users/kiasmacbookair/Projects/captcha-lab/main.py)
- [solver.py](/Users/kiasmacbookair/Projects/captcha-lab/solver.py)
- [fusion.py](/Users/kiasmacbookair/Projects/captcha-lab/fusion.py)
- [ocr.py](/Users/kiasmacbookair/Projects/captcha-lab/ocr.py)
- [extract_tiles.py](/Users/kiasmacbookair/Projects/captcha-lab/extract_tiles.py)

## Status

This is a frozen production baseline for CAPTCHA recognition. The repository is expected to evolve carefully, with changes documented against the current behavior rather than replacing it silently.
