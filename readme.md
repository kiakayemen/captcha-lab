# CAPTCHA OCR refactor v3

This version adds automatic extraction of the three-digit target from the instruction strip above the detected 3×3 grid.

## Automatic target mode

```bash
python captcha_solver.py dataset/Screenshot-00023.png
```

## Manual fallback

```bash
python captcha_solver.py dataset/Screenshot-00023.png 909
```

The solver writes:

```text
output/live_solver/instruction.png
output/live_solver/detected.png
output/live_solver/tile_1.png ... tile_9.png
output/live_solver/decision.json
```

Expected automatic-mode output begins with something like:

```text
Detected target: 909 confidence=0.998 variant=upscale
```

If automatic extraction fails, inspect `instruction.png`. You can still pass the target manually.
