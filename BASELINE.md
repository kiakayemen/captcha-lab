# Frozen CAPTCHA OCR Baseline

## Status

This document records the frozen production baseline for the CAPTCHA recognition pipeline.

The current recognizer is considered sufficiently accurate for integration. The CNN and synthetic-data experiments are archived and are no longer part of the production recognition path.

## Production pipeline

The production recognizer follows this sequence:

1. Load the CAPTCHA image.
2. Detect and crop the 3×3 tile grid.
3. Generate the configured preprocessing variants for each tile.
4. Run the fine-tuned PARSeq model on every preprocessing variant.
5. Clean and normalize the OCR predictions.
6. Pass all OCR candidates to the trained fusion selector.
7. Return one final number prediction and fusion score for each tile.
8. Select tiles whose predicted number matches the target number.

## Production source files

The production pipeline currently depends on:

* `captcha_solver.py`
* `main.py`
* `solver.py`
* `target_extractor.py`
* `fusion.py`
* `ocr.py`
* `preprocess.py`
* `models/fusion_model.joblib`

## Production model

* Model file: `models/fusion_model.joblib`
* Model role: combine OCR candidates from multiple preprocessing variants
* Model type: trained fusion selector
* Required at runtime: yes
* Missing-model behavior: fatal error
* Silent fallback behavior: disabled

The production solver must not silently switch to `confidence_sum_fallback` when the trained model is unavailable.

## OCR preprocessing variants

The frozen fusion model expects these preprocessing variant names:

* `raw`
* `gray_clahe`
* `upscale_2x`
* `upscale_3x`
* `sharpened`
* `otsu`
* `adaptive_threshold`
* `saturation_mask`
* `lab_color_distance`

These names and their associated transformations must not be changed without retraining and validating a new fusion model.

## Benchmark results

Current labeled-tile benchmark:

* Total labeled tiles: 513
* Correct production fusion predictions: 466
* Production tile accuracy: 90.84% (466/513)
* No-prediction tiles: 8/513
* Correct all-variant oracle predictions: 483
* All-variant oracle accuracy: 94.15%

The oracle result means that at least one preprocessing variant produced the correct value for 483 of the 513 tiles. It is not the production accuracy.

## Frozen dependency versions

* Fine-tuned PARSeq checkpoint: `experiments/parseq_finetune/run_001/best_model.pt`
* Python package `opencv-python`: 5.0.0
* Python package `numpy`: 2.5.1
* Python package `pandas`: 3.0.3
* Python package `scikit-learn`: 1.9.0
* Python package `joblib`: 1.5.3
* Python package `torch`: 2.13.0

The complete dependency list is stored in `requirements-lock.txt`.

## Archived experiments

The following work is preserved for reference but is not part of production:

### CNN experiments

Stored under:

`experiments/cnn/`

This includes:

* CNN source code
* CNN training scripts
* CNN checkpoints
* training-history files

### Synthetic-data experiments

Stored under:

`experiments/synthetic/`

This includes:

* synthetic renderers
* synthetic compositors
* generated previews
* glyph and style analysis
* prototype-generation scripts
* synthetic datasets and reports

## Frozen decisions

The following decisions apply to this baseline:

* Do not perform additional CNN training.
* Do not create additional synthetic training data.
* Do not modify preprocessing transformations.
* Do not rename preprocessing variants.
* Do not retrain the fusion model automatically.
* Do not use confidence-sum fallback in production.
* Do not continue instruction-number OCR as the primary extraction method.
* Treat the fusion model as a required production artifact.
* Measure future improvements against this frozen baseline.

## Future work

Changes after this baseline should focus on:

1. complete-CAPTCHA benchmarking;
2. false-positive and false-negative measurement;
3. uncertainty and rejection rules;
4. standardized machine-readable solver output;
5. one stable production entry point;
6. authorized browser or scraping integration.

Any recognition change that affects benchmark results should be recorded as a new baseline rather than silently replacing this one.
