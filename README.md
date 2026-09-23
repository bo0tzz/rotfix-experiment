# Rotation detection for Immich

C4-equivariant CNN that detects wrongly-rotated photos and predicts the correction. Built to decide
whether this is worth building in.

`v2-big12`: 757k params, ~3MB, 28ms/img on a Xeon Gold and 208ms on a Celeron N5105 under
immich-ml's CPU defaults (ORT, batch 1, `intra_op=2`) — ~half a CLIP pass, +4–7% on Immich's
existing per-image ML.

## Results

13,352 real library images, 693 rotated (5.2%), human-verified. Scored in **stored** orientation;
a hit requires the **exact angle**.

| model | licence | ms | R@P≥.93 | R@P≥.97 | AP | one-press |
|---|---|---:|---:|---:|---:|---:|
| v2-224full 6ep | ours | 9 | .846 | .771 | .9329 | 574 |
| v2-224full12 12ep | ours | 9 | .867 | .824 | .9492 | 592 |
| v2-384real 6ep | ours | 27 | .870 | .811 | .9521 | 593 |
| v2-bigfull 6ep | ours | 28 | .898 | .860 | .9593 | 616 |
| **v2-big12 12ep** | ours | 28 | **.922** | **.890** | **.9699** | **631** |
| mnv4-v2 6ep | ours | 62* | .905 | .889 | .9498 | 619 |
| check_orientation | **CC-BY-NC** | 324* | .945 | .478 | .9457 | 628 |

\* PyTorch eager, not re-measured under ORT.

`check_orientation` (ternaus 2020) is the off-the-shelf baseline: unshippable licence, and it
collapses at the high-precision end (R@P≥.97 .478) so it can't back a pre-rotated band anyway.

`v2-big12` PR: thr .95 → 657 flagged, P .956, R .906. Argmax → 1015 flagged, P .671, R .983.
12 of 693 never surface. Intended as two bands (pre-rotated one-press / surfaced-only), not
autonomous rotation.

## Architecture

The logit vector *is* the group axis, so rotating the input cyclically permutes the logits — one
pass is exactly equivariant and 4-rotation TTA is provably a no-op. That's the 4x over the
non-equivariant baselines, not network size. Head is a single `Linear(w, 1, bias=False)` shared
across the four group slices, so there's no per-class parameter and it can't learn a class prior.

`p4_scratch` is the same net at half width (189,712 params).

## Levers, vs a 0.0015 AP noise floor (identical-config seed repeat)

| lever | gain |
|---|---|
| capacity (half → full width) | **+.0265** |
| training steps (6 → 12ep) | +.0106 large, +.0164 small |
| data volume (quarter → full) | +.0149 / +.0081 |
| resolution 224 → 384 | +.0192, confounded with 3x steps (bs32 vs bs96); 2.6x CPU |
| 2nd model ensemble | +.0048 (2.4x CPU) |
| 3rd model ensemble | +.0002 |
| rotation augmentation | exactly 0 (no-op for C4 nets) |
| EXIF orientation as gate | negative — `orientation=1` is 5.36% rotated vs 5.19% base; missing tags 6.6x safer |

## Reusing CLIP doesn't work

Tested on `immich-app/ViT-B-32__openai`, all 13,352 images at 4 rotations (`src/clip_baseline*.py`):

| approach | 4-way | upright-kept | product P/R @argmax |
|---|---:|---:|---|
| zero-shot vs directional prompts (best of 3) | .667 | .687 | — |
| "which rotation looks most upright?" | .070 | .070 | — |
| linear probe on pooled vector | .909 | .931 | P .220 / R .488 |
| v2-big12 | .974 | .998 @.95 | P .671 / R .983 |

`cos(stored, rotated 90°) = 0.88` vs 0.59 for a random different image — pooling discards most of
the orientation signal. The probe is the ceiling for any linear readout and still misses over half.
Rotate-and-score is *below* the .25 chance floor. Spatial pooling recovers it (2x2 → ~89%) but then
you're re-running CLIP, which costs more than the dedicated model.

## Data

Open Images, 1,094,589 images, CC-BY 2.0 with per-image author — picked for redistributable
weights, which is the reason we didn't just use check_orientation. Labels are Google's own
`Rotation` column (13% blank, dropped), **not** an assumed-upright prior. Their labelled rotated
rate is 1.3% vs the 5.2% we measure, so label noise is reduced, not gone.

Derivation: original → JPEG q80 4:4:4 → shortest-side 256 → q95 4:4:4. The q80 deliberately bakes
in Immich's preview artefacts. Test set is separate: library previews pulled verbatim, no re-encode.

## Layout

```
src/models.py       P4Net ships; PlainNet / P4MobileNet are comparison arms
src/train.py        arms + hyperparams via env (ARCH, RES, EPOCHS, BS, ...)
src/derive.py       builds the training set
src/bake.py         required before ONNX export, see below
src/dump_probs.py   per-image posteriors → .npz
src/score.py        PR curves + AP from those .npz     `python score.py v2-big12 ...`
logs/               per-epoch histories, raw + rendered
results/            posteriors (p, y, ids) for every run
NOTES.md            ~3,000-line lab notebook, everything timestamped
```

## Gotchas

- `materialise()` runs inside `forward()`, so `aten::rot90` blocks ONNX export. `bake.py` folds the
  group convs into plain convs — exact (`max|diff| = 0`), and 2.4x faster than eager.
- Per-epoch `lib_*` in the logs is a 3,500-image subset. It misled us three times; judge on
  `score.py` only.
- 180° is the weak class in every model and config tested.
- Ground truth is one family's library; the 5.2% base rate will vary.
