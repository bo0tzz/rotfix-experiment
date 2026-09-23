# Rotation-detection experiment on the Immich library

Started 2026-09-19. Working dir: `~/rotfix-experiment` (script `rotfix.py`, venv `.venv`).
Goal: find wrongly rotated images in the family account (13 353 images, server 3.2.2), fix them via the
non-destructive edits API, and collect enough data to judge whether this is worth building into Immich.

## Setup

- Machine: i9-12900H (6P+8E, 20 threads), 31 GB RAM, Intel UHD iGPU (Alder Lake), no dGPU.
- Input: Immich `thumbnail` size (webp, ~250 px long edge, EXIF orientation already applied, i.e. exactly what
  the UI shows). 13 356 listed via `POST /search/metadata` (filter tree, cursor paging, 1000/page);
  13 352 downloaded, 4 gave HTTP 404 on the thumbnail endpoint. 234 MB total, 16 concurrent, ~1 min.
- Two detectors, same label convention: class k = "needs k*90 degrees clockwise to be upright".
  1. **Probe**: frozen DINOv2 ViT-L/14 (`facebook/dinov2-large`, 224x224 squash, CLS + mean-patch = 2048-d)
     + StandardScaler + LogisticRegression, trained on 3000 random library images x 4 synthetic rotations.
     Assumes the library is mostly upright (label noise = the true rotated rate). 20% of the 3000 held out.
  2. **check_orientation** (ternaus, 2020): ResNeXt50-32x4d SWSL, 4-class, trained on OpenImages.
     Class k = `np.rot90` applied k times (CCW), so same convention as ours. 23 M params.
- Inference: 4-rotation test-time augmentation, aggregated as p_agg[c] = sum_j p_j[(c + j) mod 4].
  For the probe, TTA only for images with p_upright < 0.95 on the plain pass (budget).

## Throughput (DINOv2-L, 224x224)

| backend | device | batch | img/s |
|---|---|---|---|
| PyTorch 2.14 CPU fp32 | CPU (20 thr) | 32 | 1.6 |
| OpenVINO CPU, THROUGHPUT hint | CPU | 16 | 1.6 |
| OpenVINO GPU (fp16 default) | iGPU | 16 | 7.8 (sync) / 11.7 (async queue, 4 requests) |
| OpenVINO GPU | iGPU | 32 | 9.3 (sync) |
| OpenVINO MULTI:GPU,CPU | both | 16 | 8.6 (worse than GPU alone) |

DINOv2-base on PyTorch CPU: 5.8 img/s. PyTorch XPU does not support Alder Lake iGPUs; OpenVINO does
(needs `intel-compute-runtime`, `/dev/dri/renderD128` world-rw here). Convert once with
`ov.convert_model` (~40 s), cache the IR under `emb/`.

## Runs

See `runs.jsonl` (one line per stage with timings) and the `*.log` files.

## Results

(filled in as they arrive)

## Dataset files

- `assets.json`: id, owner, visibility, livePhotoVideoId, originalFileName, originalPath for every image.
- `thumbs/<id>.webp`: the inputs.
- `emb/feats.npz`: DINOv2 features keyed `<id>:<k>` (float16); `emb/train_ids.json`; `emb/clf.pkl`.
- `predictions.json`: probe output for TTA candidates (id, fix_cw, conf, p_upright).
- `predictions_co.json`: check_orientation softmax for all 4 rotations of every image.
- `applied.json`: every edit applied (id, previous edits, angle, conf) — also the revert log.
- Human review outcome: album "Rotation review"; whatever gets reverted afterwards = false positives.

## Open questions / things to verify

- Direction of Immich's `rotate` edit angle: sharp `.rotate(angle)` is clockwise, but edits go through
  `createAffineMatrix` which uses `rotate(-angle)` then `sharp.affine`. Verify on one asset before bulk apply.
- `PUT /assets/:id/edits` **replaces** all edits, so existing edits must be fetched and re-sent.
- Edits are refused for live photos, panoramas, GIF, SVG.

## Incidents

- 13:10 GPU HANG (i915 "Fence expiration time out", context reset) running DINOv2-L on the iGPU with
  batch 32 and 4 async infer requests, while the CPU was also saturated by the check_orientation run.
  OpenVINO surfaced it as `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`. Restarted with batch 16 and
  2 requests; embed made resumable (checkpoint every 2048 embeddings). Lesson for a built-in version:
  iGPU inference needs small batches and must tolerate device resets.
- check_orientation progress line in the first run under-reported by 16x (counted flushes, not images);
  real rate ~4 img/s (16 passes/s) on 12 CPU threads while contending with the OpenVINO conversion.

## Early comparison (13:35, after first checkpoint)

Probe trained on only 215 images x4 (held-out 53 x4): 87.3% accuracy without TTA. Overlap with
check_orientation: 1244 images. Probe flagged 52, check_orientation 51, both agreeing 14, probe-only 27,
co-only 26, both flag but disagree on angle 11. Eyeballed contact sheets (`sheet_*.png`, `quick_compare.json`):

- **both agree**: mostly real hits (sideways harbour/band photos, newspaper/book pages shot sideways).
- **co-only**: nearly all false positives, including at conf >= 0.9 (upright barn, painting, people lying on
  a floor, top-down shots, a cow). No convincing hit the probe missed.
- **probe-only**: mixed; the undertrained probe has real hits at >= 0.85 (sideways cow, sideways newspaper text)
  and false positives on people lying down, dogs on grass, portraits.
- **disagree**: all text pages/documents, 90 vs 270 ambiguity; the two receipts look upright to me, so both
  detectors are wrong there. Documents are the hard class for both.

Decision: check_orientation adds ~no recall and halved the iGPU feeder's throughput (11.7 -> ~4-6 img/s)
through CPU contention. Killed it at ~2400 images done (results kept in `predictions_co.json`). Plan: re-run
it later only on the probe's candidate set (a few hundred images, minutes) so the agreement signal is still
available as a precision filter for auto-apply.

## Full probe on pooled features: too weak (14:15)

Full embedding run finished (22 352 vectors, 2301 s, 7.0 img/s on the iGPU with batch 16 x 4 requests;
restart at 13:50 from a checkpoint after the 2-request run averaged only ~4.3 img/s).

Probe on CLS + mean-patch features, 2400 train x4 / 600 val x4:

| C | 4-way acc | upright kept (TTA) |
|---|---|---|
| 0.01 | 83.5% | 91.7% |
| 0.1 | 80.0% | 91.2% |
| 1 | 77.7% | 90.8% |

Not usable: ~8% of upright images would be flagged (~1000 photos). Diagnosis: mean-pooling the patch
tokens is rotation-invariant by construction, and the CLS token carries little layout information, so the
feature threw away exactly what orientation depends on. Lesson: for orientation, keep spatial layout
(coarse grid pooling), do not use globally pooled embeddings. Immich's existing CLIP embeddings would have
the same problem, so "reuse the smart-search vectors" is not a shortcut for this feature.

## Dev-set iteration (14:20)

Per feedback, iterate on a small set before more long runs. 500 training images x 4 rotations, saving
CLS + 4x4-pooled patch grid (17 x 1024 per image) so feature variants can be compared without re-embedding.
Same 500 images run through check_orientation for a like-for-like 4-way comparison on synthetic rotations.
Split 375 train / 125 val by image. Metrics: plain 4-way accuracy, TTA accuracy, and "upright kept"
(fraction of stored images not flagged), which is the number that matters for false positives.

## Dev-set results (15:55)

375 train / 125 val images x4 synthetic rotations. All probe variants and check_orientation plateau at
89-90% "upright kept" under 4-rotation TTA (with full TTA the prediction is exactly rotation-equivariant, so
TTA accuracy == upright-kept by construction):

| features | plain 4-way acc | upright kept (TTA) |
|---|---|---|
| cls | 74.8% | 89.6% |
| cls+mean (what the full run saved) | 79.0% | 89.6% |
| 2x2 grid | 89.2% | 90.4% |
| cls+2x2 | 88.8% | 90.4% |
| 4x4 grid | 89.6% | 89.6% |
| check_orientation (ResNeXt50) | 89.0% | 89.6% |

Eyeballing the 14 val images flagged by either detector (`sheet_dev_val.png`, `dev_val_flags.json`):
the 9 where both agree on the angle are ALL genuinely rotated in the library. Probe-only (1) and co-only (2)
are false positives; the 2 angle-disagreements are a book page and a map (ambiguous).

So the plateau is the library, not the models: ~7% (9/125) of a random sample is stored rotated, which is
exactly the label noise both "upright kept" numbers bottom out at. Corrects the 14:15 diagnosis: pooled
features lose plain accuracy (79 vs 89%) but TTA recovers it; the "8% false positives" on the full probe
were mostly real detections. Expect on the order of 900 rotated images library-wide.

Policy for auto-apply: both detectors agree on a non-zero angle (9/9 precision on dev). Singletons and
disagreements go to the review album only.

## Direction verification (16:05)

Asset DSCF1128.jpg (girl on a road, stored sideways; both detectors: needs 270 CW). `PUT /assets/:id/edits`
with `{"action":"rotate","parameters":{"angle":270}}` -> upright. So Immich's rotate angle == our
"clockwise degrees to fix"; no sign flip needed. Edited thumbnail was regenerated within seconds, but
`GET /assets/:id/thumbnail` keeps serving the unedited render unless `edited=true` is passed (shared links
force it). Logged in `applied.json` with note "direction verification".

check_orientation converted to OpenVINO and run on the iGPU for full-library 4-rotation TTA (see runs.jsonl).

## Full-library baseline on the iGPU (16:07)

check_orientation via OpenVINO on the iGPU: 9771 remaining images x 4 rotations in 362 s (27 img/s,
~108 passes/s), versus ~4 img/s on 12 CPU threads with PyTorch. Small CNNs are cheap on the iGPU;
ViT-L was 7-10 img/s.

Probe prediction: plain pass on pooled features leaves 4209 of 13352 images below p_upright 0.95 (the
weak plain accuracy of pooled features shows here), plus 25 more from check_orientation flags not already
included; 9948 extra embeddings for TTA. With 2x2 grid features this candidate set would be roughly a
third the size, so grid pooling is the right choice for a built-in version even though TTA equalises accuracy.

Machine suspended during one of the long runs (lost a chained step); now running under
`systemd-inhibit --what=sleep:idle` for the rest of the session.

## Final detection results (17:00)

Probe TTA prediction: 4234 candidates, 9948 extra embeddings, 1224 s on the iGPU. 854 probe flags.
check_orientation full TTA: 919 flags. Both agree on a non-zero angle: 677. Probe-only 121, co-only 186,
both flag but disagree on angle 56. Union of anything flagged: 1040 of 13 352 (7.8%), consistent with the
dev-set estimate of ~7%.

Probe confidence is poorly calibrated in absolute terms (heavily regularised 4-way softmax averaged over 4
rotations): max 0.95, so thresholds were set from contact sheets rather than from the numbers.

Contact-sheet review (`sheet_final_*.png`):
- Random sample of agreements: ~44/48 correct. Errors concentrate in 180-degree flags (poppy, gravel,
  top-down shots) and in images with no defined "up".
- Weakest agreements: ~50-60% precision; false positives are top-down object shots, textures, sky, black
  frames; nearly all have check_orientation conf < 0.8. True hits in that tail mostly have co conf >= 0.83.
- co-only, top by confidence: many REAL hits the probe missed (kids sideways, costumes, ponies, newspaper
  pages) => the pooled-feature probe has a recall problem, not just a plain-pass problem. check_orientation
  at >= 0.95 on 90/270 looked ~85% precise; its 180 flags are unreliable.
- probe-only: mostly false positives, especially 180 (ceilings, signs, shelves).
- angle disagreements: almost all text pages/documents (90 vs 270 ambiguity).

Tiers (`tier` step, stored in combined.json):
- A (auto-apply): agree AND co_conf >= 0.8; for 180 additionally probe_conf >= 0.5 AND co_conf >= 0.9.
  605 images (90: 387, 270: 209, 180: 9). Weakest-32 sheet (`sheet_tierA_weak.png`): ~24/32 clearly right,
  rest top-down/ambiguous. Estimated precision ~95%.
- B (likely, not applied): weak agreements + co-only >= 0.95 on 90/270. 89 images.
- C (unsure): everything else flagged. 346 images.
Albums: "Rotation: auto-rotated" (A), "Rotation: likely, not rotated" (B), "Rotation: unsure" (C).

Ground truth for the dataset comes from the user's pass through the albums: reverts in A = false
positives; manual rotations in B/C = false negatives of the auto policy. `applied.json` has, per asset:
previous edits, angle applied, both detectors' outputs, thumbhash before, and the face-refresh timestamp.

## Takeaways for a built-in feature

1. The rotated rate in a real family library can be several percent (7-8% here, ~1000 images), which is
   far too many to fix by hand and makes a detector genuinely valuable.
2. A small dedicated CNN (check_orientation, 23 M params, 2020) is as accurate as a DINOv2-L probe on this
   task once 4-rotation TTA is used, and ~10x cheaper; on an Intel iGPU it does ~100 passes/s. The
   expensive backbone bought nothing here. A modern small model fine-tuned on rotation would be the pick.
3. Globally pooled embeddings (CLIP/DINO CLS or mean) are a poor basis: plain accuracy ~79%. Keep a coarse
   spatial grid (2x2 was enough: 89%). Reusing the existing smart-search CLIP vectors would not work.
4. 4-rotation TTA is essential (makes the prediction rotation-equivariant) and 180-degree predictions are
   the least reliable class for every model; treat them more conservatively.
5. Two models agreeing is a strong precision filter (9/9 on dev, ~95% at scale with a confidence floor);
   documents, top-down shots, textures and sky are the systematic failure classes and could be
   down-weighted with cheap heuristics (OCR present, no faces, low edge orientation entropy).
6. Operationally: the edits API (non-destructive rotate) is the right sink; it regenerates thumbnails
   but not faces/CLIP/OCR, so a feature would need to re-queue those. Queue status is admin-only.
   Face re-detection on a rotated asset replaces all ML faces (old boxes no longer overlap).
7. Input: 250 px thumbnails were sufficient for both models.

## Apply (17:10)

600 of 605 tier-A rotations applied via the edits API (~5 min, 3 calls per asset). 5 were refused with
"Duplicate edit actions are not allowed": they already carried a manual 90-degree rotate edit. Because the
pipeline fetched the *unedited* thumbnail render, it re-detected them; the server's duplicate check
prevented a double rotation. Bonus ground truth: on all 5 human-fixed assets both detectors predicted 90,
matching the human's edit (5/5). Removed them from the auto-rotated album. No applied asset had prior edits.

Albums: auto-rotated 600, likely-not-rotated 89, unsure 346.
Face refresh deferred until after the human review (per user); `faces` step now skips assets whose rotate
edit was removed in the UI since. Lesson for a built-in version: analyse the edited render, not the original.

## Evaluation setup (17:06)

Baseline snapshot `edits_snapshot_20260919T170608.json`: 605 edited assets library-wide = the 600
auto-applied + the 5 pre-existing manual 90-degree rotations. Nothing else in the library had edits.
After the human review, `rotfix.py snapshot` again and `rotfix.py evaluate` diffs the two:
tier-A reverts/angle changes = auto-apply false positives; rotations added in B/C = misses of the auto
policy (with which detector had the right angle); rotations added on unflagged assets = detector-union
false negatives (bounded by what the human actually finds).

## Evaluation against the human review (17:30)

Human pass complete over all three albums. Final snapshot: 688 rotated assets library-wide (5.2% of 13 356;
90: 427, 180: 18, 270: 243) = 595 kept auto-rotations + 1 angle correction + 38 from album B + 49 from
album C + 5 pre-existing. Recall numbers below treat every unflagged image as upright, since the human only
reviewed flagged ones, so they are relative to what was found (detector-union recall is 1.0 by construction).

Tier outcomes: A 595/600 correct (99.2%). B: 38/89 were rotated (check_orientation had the right angle on
37/38, the probe on 22/38). C: 49/346 rotated (check_orientation right on 40/49, the probe on 3/49).

Per-detector, angle-exact precision / recall (`eval_detectors.txt`):

| detector | thr | flagged | P | R |
|---|---|---|---|---|
| check_orientation alone | 0.8 | 690 | 0.935 | 0.938 |
| check_orientation alone | 0.9 | 580 | 0.967 | 0.815 |
| DINOv2 probe alone | 0.5 | 537 | 0.940 | 0.734 |
| DINOv2 probe alone | 0.0 | 854 | 0.732 | 0.908 |
| both agree, co thr | 0.8 | 615 | 0.980 | 0.876 |
| both agree, co thr | 0.9 | 533 | 0.991 | 0.767 |

Conclusions:
- The 2020 off-the-shelf ResNeXt50 is the better single model by a wide margin at equal recall, and it
  nearly matched the ensemble on its own. The DINOv2-L probe (pooled features, 3000 noisy training images)
  mainly contributed precision as a veto; it cost ~10x the compute and was the one that found nothing alone.
- Ensemble agreement was worth ~4.5 points of precision at 0.8 over check_orientation alone, at a cost of
  ~6 points of recall. For an auto-apply feature that trade is right; for a "review these" feature, a single
  small model at ~0.8 is enough.
- check_orientation's remaining errors at >= 0.9: 8 false 180s, 8 false 90s, 2 false 270s on upright
  images, 1 angle mix-up. Its misses among rotated images (11) are dominated by 180 confusions.
  180 degrees is the weak class in every direction.
- The user's 125-image dev sample estimated 7% rotated; the true rate was 5.2%. Small-sample noise.

## OCR-box orientation heuristic (17:40, `eval_ocr.txt`, `ocr_sample.json`)

Idea (from asset IMG_1574.JPG): Immich's OCR boxes reveal text-line orientation. Tall boxes (h >= 1.5 w,
the same rule the recogniser uses to stand a crop up) => 90/270; wide => 0/180. Direction: the recogniser
applies one fixed 90-degree CCW turn to tall crops and has no angle classifier, so readable high-score text
in tall boxes implies the image needs 270 CW.

Gotcha: `GET /assets/:id/ocr` returns boxes transformed into the *edited* frame (transformOcrBoundingBox),
so for assets carrying a 90/270 rotate edit the dims swap and tall/wide flips. Evaluated on the 1040 flagged
assets + 1500 random unflagged ones.

- Coverage is the limit: only 100/1040 flagged assets have any OCR text, 52 have >= 3 boxes (tier A 34,
  B 11, C 55 with text). Of the 56 angle-disagreement cases (visually mostly newspapers) only 7 have OCR
  text at all, so either OCR has not run on most of the library or detection struggles on those images.
- Where it votes (>= 3 boxes, coverage >= 2%): 41/41 correct on flagged assets (18 sideways calls all truly
  rotated with the exact angle, 23 upright calls all upright); unflagged sample 24/25 (one false sideways).
  Looser (>= 2 boxes): 50/55 and 94/95.
- Direction by recogniser score is NOT validated: 19 of the 20 sideways-text assets were truly 270, so
  "tall => 270" alone explains the exact-angle hits; only one 90 case (score 0.89 vs 0.89-0.93 for 270s).
  A built-in version should re-run recognition on the flipped crop instead of inferring from score.

Verdict: a near-free, high-precision, low-coverage vote. Worth adding as a tie-breaker for document-like
images in a built-in feature, not a detector on its own.

## Benchmark: DuarteBarbosa EfficientNetV2-S vs check_orientation (18:02, `eval_duarte.txt`)

Model: `DuarteBarbosa/deep-image-orientation-detection` v2, ONNX (80 MB, EfficientNetV2-S @ 384px,
resize 416 + center-crop 384, ImageNet norm, raw logits out). Run via OpenVINO on the iGPU: 13 352 images
x 4 rotations in 1231 s = 10.9 img/s (~44 passes/s), vs 27 img/s for check_orientation at 224px.
Class semantics resolved empirically against ground truth: class k == needs k*90 CW (its config's
CW/CCW comment for class 3 is wrong). Scored against the 688 human-verified rotated images.

Angle-exact precision/recall at matched operating points:

| detector | thr | flagged | P | R |
|---|---|---|---|---|
| Duarte TTA | 0.90 | 687 | 0.949 | 0.948 |
| Duarte TTA | 0.80 | 728 | 0.911 | 0.964 |
| Duarte plain (no TTA) | 0.90 | 831 | 0.806 | 0.974 |
| check_orientation TTA | 0.80 | 690 | 0.935 | 0.938 |
| check_orientation TTA | 0.90 | 580 | 0.967 | 0.815 |
| Duarte AND check_orientation agree | 0.80 | 638 | 0.983 | 0.911 |
| Duarte AND check_orientation agree | 0.90 | 548 | 0.993 | 0.791 |

- Duarte is the better single model: at its best point it beats check_orientation's best by ~1.5 points of
  precision and ~13 points of recall (0.949/0.948 vs 0.967/0.815 or 0.935/0.938). It misses only 5 of 688.
- Its 98.82% claim does NOT hold on unseen data (it is ~95% here at best), consistent with the suspected
  sample-level rather than image-level train/val split. Still the strongest off-the-shelf option found.
- TTA is worth ~14 points of precision for it (0.806 -> 0.949 at 0.90).
- Confidence is not peaked: plain-pass max ~0.93, TTA max ~0.95. Thresholds are model-specific.
- Its remaining errors at >=0.9 are all false positives on upright images (90: 18, 180: 10, 270: 7);
  same systematic classes as everything else (bark, sky, blurry night shots, brick, top-down objects).
- The two-model ensemble is still the precision king (0.983 @ 0.911 recall) and would have been a better
  auto-apply policy than the one actually used (0.98 precision at 0.876 recall).

**Did it find anything the review missed? No.** 286 disagreements with ground truth, all false positives:
- 98 where Duarte AND check_orientation agree the image is rotated but truth says upright: every single
  one was already shown in the review albums and judged upright by the human. Zero unseen.
- 119 Duarte-only flags on never-reviewed images: check_orientation says upright on all 119; visual check
  of the top 48 shows tree bark, textures, sky, blurry night shots, brick walls, top-down flowers.
- Remainder are angle disagreements on already-rotated assets.
So the library is clean and the earlier two-model union had effectively 100% recall on what exists.

## Licensing for shipping in Immich

Immich re-hosts its models under the `immich-app/` HF org and downloads via `snapshot_download`, so any
candidate has to be redistributable commercially (AGPL project, paid Immich offering).

- **check_orientation**: code MIT; weights trained on **OpenImages**, whose images are CC-BY 2.0 and
  annotations CC-BY 4.0. Commercial use permitted with attribution. Cleanest provenance of the three.
- **Duarte v2**: repo/model **MIT**, but trained on COCO (CC-BY 4.0), Kaggle TextOCR (CC-BY 4.0,
  images themselves sourced from OpenImages), a Kaggle "AI-Generated vs Real Images" set (license
  unverified, and AI-generated imagery carries its own provenance questions), and the author's personal
  photographs (no stated license/consent terms). The unverified Kaggle set and the personal photos are the
  risk; the MIT grant covers the author's own contribution but cannot launder upstream terms.
- **PP-LCNet_x1_0_doc_ori**: Apache 2.0, from the PaddleOCR family Immich already uses. Documents only.
- Legal posture: whether weights are a derivative of training data is unsettled, and CC-BY's practical
  ask (attribution to dataset sources) is cheap to satisfy in a NOTICE/model card. The real blocker is
  a dataset with *unknown* terms, not a CC-BY one.

Implication: Duarte's model is the best performer but the weakest provenance. The safe path is to train
Immich's own model on datasets with auditable commercial-use terms (OpenImages / COCO / Unsplash Lite,
all CC-BY or explicitly commercial-permissive), using Duarte's published recipe, which cost ~5 GPU-hours
on one H100. That also buys an image-level split, an abstain class for no-defined-up images, and a
backbone sized for Immich's deployment targets.

## Inference cost vs Immich's existing models (19:30, `bench_models.py`)

All measured on the same Alder Lake iGPU via OpenVINO fp16, THROUGHPUT hint, 4 async requests.
Defaults confirmed in `server/src/dtos/config.dto.ts`: clip `ViT-B-32__openai`, face `buffalo_l`.

| model | role | px | ms/pass | passes/img | ms/img |
|---|---|---|---|---|---|
| CLIP ViT-B-32 visual | smart search, every image | 224 | 31.5 | 1 | 31.5 |
| buffalo_l detection | faces, every image | 640 | 22.7 | 1 | 22.7 |
| buffalo_l recognition | per detected face | 112 | 5.8 | ~2 | ~12 |
| **existing per-image total** | | | | | **~66** |
| check_orientation + TTA | candidate | 224 | 10.3 | 4 | 41 |
| Duarte EffNetV2-S + TTA | candidate | 384 | 32.4 | 4 | 129 |
| Duarte EffNetV2-S + TTA | candidate | 224 | 10.7 | 4 | 43 |

The CLIP ONNX has batch baked to 1 (reshape to bs>1 fails on the position-embedding add), so it is
measured at bs=1; the others at bs=8/32. Duarte at 384 with TTA costs ~2x Immich's entire existing
per-image ML pipeline. File sizes: CLIP visual 335 MB, buffalo rec 166 MB, Duarte 77 MB, buffalo det 16 MB.

Duarte at 224px (trained at 384, run at 224) on the full library, angle-exact:

| thr | 384px P/R | 224px P/R |
|---|---|---|
| 0.80 | 0.911 / 0.964 | 0.909 / 0.940 |
| 0.90 | 0.949 / 0.948 | 0.958 / 0.892 |

So 3x cheaper for ~2-5 points of recall. The model is more resolution-robust than expected.
Cost drivers, in order: TTA (4x), input resolution (384 vs 224 = 3x), backbone size.

## Data scaling (`learning_curve.txt`)

Linear probe on frozen DINOv2 2x2-grid features, 125 held-out images, mean of 3 seeds:

| train images | plain 4-way acc | upright kept (TTA) |
|---|---|---|
| 25 | 0.865 | 0.904 |
| 50 | 0.879 | 0.909 |
| 100 | 0.879 | 0.904 |
| 200 | 0.881 | 0.901 |
| 375 | 0.892 | 0.904 |

Flat from 50 images on. Caveat: this fits a linear head on already-strong features and does not bound
from-scratch training. But combined with check_orientation (OpenImages, 91% val) and Duarte (189k unique
images, ~95% here), it says the task is not data-hungry: the signal is easy, the residual errors are
concentrated in a few semantic classes, and label noise (any "upright" corpus is 1-5% actually rotated,
5.2% in this library) becomes the accuracy ceiling well before data volume does. Duarte's author manually
corrected 1300+ TextOCR labels, which is the same problem.

## Architecture research (20:00)

### Killing the 4x TTA cost

Why TTA is needed at all: Duarte was trained on all four rotations but nothing tied the four predictions
together, so its single-pass output is inconsistent (P 0.806 plain vs 0.949 with TTA on this library).
TTA is just averaging away that inconsistency at 4x the cost. Two ways to get it in one pass:

1. **C4-equivariant network (principled).** The label space here *is* the cyclic group C4, which is the
   one case where equivariance is exactly the property wanted. In a p4 G-CNN (Cohen & Welling 2016) the
   final feature map carries a group dimension of size 4; rotating the input cyclically permutes it. Read
   out that dimension and the prediction is *exactly* equivariant, one forward pass, with a guarantee
   rather than an average. 90-degree rotations are exact pixel permutations, so no interpolation and no
   steerable basis is needed; filter rotation is a transpose+flip.
   - **Export is a non-issue**, contrary to my earlier caution: escnn/e2cnn (QUVA-Lab, BSD Clear) provide
     `.export()`, which converts a trained equivariant net into a pure PyTorch model with, in their words,
     "no additional computational overhead in comparison to conventional CNNs". The tied filters are
     materialised into ordinary conv weights, so the exported graph is a plain CNN with no custom ops and
     ONNX/ArmNN/RKNN export behaves normally. Caveat: only "a few commonly used modules" implement
     `export()`, so the architecture must be built from the supported set.
   - Cost: general G-CNNs are quoted at up to 3.78x slower / 4.63x MACs, but that is for continuous or
     large groups. For planar cyclic/dihedral groups the literature describes the overhead as negligible,
     since group convolution reduces to stacking transformed filter banks and indexing. Needs measuring,
     but even 2x beats TTA's 4x and is exact.
2. **Rotation-consistency loss (pragmatic).** Keep a plain CNN, add a term penalising disagreement between
   the four rotated views during training. No guarantee, trivially exportable, no inference overhead.

### Woehrer 2026, "Image Rotation Angle Estimation: Comparing Circular-Aware Methods" (arXiv 2603.25351)

The systematic study that did not exist when this experiment started: 16 backbones x 5 output heads,
5 seeds, on DRC-D and COCO. It targets continuous 360-degree estimation, not our 4-class task, but the
backbone and head findings transfer. MIT code + weights at github.com/maxwoe/image-rotation-angle-estimation.

- Best single config: EfficientViT-B3 + classification head, 1.23 deg MAE. MambaOut Base + Circular
  Gaussian Distribution 1.24 deg.
- **Small backbones are competitive**: ConvNeXt V2 Atto (3.7M) reaches 2.69 deg with a unit-vector head;
  EdgeNeXt XX-Small reaches 2.10 deg with a phase-shifting coder. Explicit conclusion: "a well-matched
  smaller backbone can outperform a mismatched larger one" and scaling the backbone does not reliably help.
- **CGD is the most robust head** (wins 9 of 16 backbones, consistent convergence); plain classification
  gets the best peak but is unstable on several backbones; direct angle regression collapses everywhere.
- **CGD outputs a distribution over angles, so its spread is a built-in uncertainty measure.** That is the
  abstain mechanism for no-canonical-orientation images, without needing a fifth class.
- Their error analysis independently matches ours: failures "cluster near cardinal rotations (90, 180)
  where the model identifies a plausible orientation axis but selects the wrong polarity or perpendicular
  direction... on images with weak gravitational cues, such as close-up views or symmetric scenes".
- DRC-D is deliberately curated to contain "clear upright indicators, avoiding ambiguous orientations such
  as aerial views or abstract scenes". Its license is not documented in the repo README.
- They attribute remaining error to dataset scale: COCO 2017 (117k) beat COCO 2014 (83k), 2.84 vs 3.71 deg.

### Other candidates

- **CPUBone** (arXiv 2603.26425, Mar 2026, CC BY 4.0, github.com/altair199797/CPUBone): backbone family
  designed for CPUs specifically, on the premise that mobile backbones assume parallelism CPUs lack.
  B0 is 10.4M params / 77.6% top-1 / 24.2 ms on a Raspberry Pi 5. Directly relevant to Immich's
  no-GPU self-hosters, though B0 is still oversized for a 4-class head.
- **RotBench** (arXiv 2508.13968): multimodal LLMs including GPT-5 and Gemini 2.5 are poor at identifying
  image rotation, worse than small specialised models. Rules out the VLM shortcut.

### Licensing landscape for backbones (the deciding constraint)

Checked the actual weight licenses rather than the repo licenses:

| backbone | weights license | commercial? |
|---|---|---|
| `swsl_resnext50_32x4d` (check_orientation's backbone) | **CC-BY-NC-4.0** | **no** |
| `convnextv2_atto.fcmae_ft_in1k` | **CC-BY-NC-4.0** | **no** |
| `mobilenetv4_conv_small.*_in1k` (timm, Wightman-trained) | Apache-2.0 | yes, modulo ImageNet caveat |
| CPUBone | CC BY 4.0 | yes, modulo ImageNet caveat |

**This is the most consequential finding of the session: check_orientation, the model that has been the
baseline throughout, is built on Facebook's semi-weakly-supervised Instagram-1B weights and is CC-BY-NC.
It cannot ship in Immich at all.** ConvNeXt V2, the best tiny backbone in the Woehrer study, is the same.

The generic caveat on everything else: timm's own model cards note ImageNet-1k was released for
non-commercial research and that the implications for pretrained weights are unsettled. That ambiguity is
industry-wide and is the same one Immich already lives with (worth confirming separately, but InsightFace's
buffalo models, Immich's face default, are to my knowledge non-commercial research licensed).

## Pilot bug: equivariance broken by the norm layer's affine (21:20)

First p4 run showed single-pass 0.7768 vs TTA 0.7865 at epoch 2. For an exactly equivariant net those
must be identical (if y(rot_r x) = roll(y(x), r) then the TTA aggregate is just 4x the single-pass
softmax, same argmax), so the gap was a bug, not noise.

Cause: `nn.GroupNorm(G, G*C)` has an independent learnable gamma/beta for every (rotation, channel) pair.
At initialisation they are all 1/0, so the equivariance unit test passed vacuously; once training makes
them differ across the rotation axis, equivariance is destroyed. Measured: 1.5e-7 error at init,
5.9e-1 with randomised affine.

Fix: `P4Norm` = GroupNorm(affine=False) over the 4 rotation blocks + an affine with only C parameters,
broadcast across the rotation axis. Per-block statistics are fine because blocks permute under rotation
and so do their statistics. Verified 7.2e-7 with *every* parameter randomised, so it holds for any
trained state, not just at init.

**Lesson for any real implementation: every learnable per-channel parameter must be tied across the
group axis, and equivariance tests must randomise the weights or they prove nothing.**
Both runs restarted from scratch at 21:20. P4Net 78,912 params, PlainNet 316,228 (4.0x).

## Open Images as the training corpus (21:30)

Metadata files (`train-images-boxable-with-rotation.csv`, 638 MB; `validation-images-with-rotation.csv`)
carry per-image `License`, `Author`, `Rotation` and `Thumbnail300KURL`.

Validation split, as a proxy for the whole set:

| | count |
|---|---|
| rows | 41,620 |
| License CC-BY 2.0 | 41,620 (100%) |
| Rotation 0 | 35,826 |
| Rotation 90 / 180 / 270 | 113 / 30 / 333 |
| Rotation blank | 5,318 (13%) |
| usable after filtering | 36,302 |

Two things this buys: uniform CC-BY 2.0 with per-image author for mechanical attribution, and **real
rotation labels instead of an assumed-upright prior**. Google's rotated rate is 1.3%, well below the 5.2%
measured in the real library, so either Flickr uploads are cleaner or the annotation is incomplete; treat
it as reduced, not eliminated, label noise. Drop the 13% with a blank Rotation.

**The `Thumbnail300KURL` column is dead.** It points at Flickr's static CDN and 0 of 24 sampled URLs
returned 200 (all 502), for both upright and rotated rows. The 2018 metadata has rotted.
Use the maintainer-hosted mirror instead: `https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg`,
which returned 8/8 at a mean of 352 KB. No credentials needed.

Sizing: a 100k-image subset is ~35 GB of download, resized once to 256 px it is ~2 GB on disk. Deferred
until the pilot finishes so it does not steal CPU from training. `oid_prepare.py` builds the filtered,
shuffled pool; validated on the validation split (36,302 rows kept).

## Export validation (21:50, `pilot_export.py`)

Baked a trained-state P4Net into stock `nn.Conv2d` layers (materialising the rotated/shifted weights once)
and exported to ONNX opset 17.

- `max |P4Net - ExportedP4| = 5.4e-7` - identical computation.
- Exported model equivariance error `8.9e-7` - the property survives the bake.
- ONNX ops: `Add, AveragePool, Conv, InstanceNormalization, MatMul, Mul, ReduceMean, Relu, Reshape,
  Shape, Squeeze`. **No custom operators.** The export concern was unfounded; a p4 net is a plain CNN
  once the weights are materialised.
- Stored parameters go 78,912 (trainable) -> 315,504 (materialised), i.e. the same as PlainNet's 316,228.
  Worth being precise about: weight sharing is a *training-time* property. It buys sample efficiency and
  exactness, **not** a smaller deployed file. Compute and file size at inference match a plain net of the
  same width; what you save is the 4x of TTA.

Deployment note: GroupNorm lowers to `InstanceNormalization`, which some ARM NPU toolchains (notably RKNN)
handle poorly. A production version should use BatchNorm with its affine tied across the rotation axis,
which folds into the preceding conv at export and leaves a pure Conv/Relu/Pool/MatMul graph.

## PILOT RESULT (2026-09-20 00:40) - equivariance beats TTA outright

18 epochs each, 64 px, 10,682 train / 2,670 test images by split, every test image evaluated in all four
stored orientations. Single seed.

| arch | params | best single-pass | best TTA | TTA gain | train wall-clock |
|---|---|---|---|---|---|
| PlainNet | 316,228 | 0.8330 | 0.8491 | +0.0161 | 115 min |
| P4Net | 78,912 | **0.8689** | 0.8689 | **+0.0000** | 139 min |

**The question was: does one equivariant pass match a plain net's four-pass TTA? It beats it.**

- p4 single pass 0.8689 (1 forward pass) vs plain+TTA 0.8491 (4 forward passes). +2.0 points of accuracy
  at a quarter of the inference cost.
- `max |single - tta|` across all 18 epochs: p4 **0.000000** exactly, plain 0.039. The exactness is not
  approximate and does not drift during training.
- Sample efficiency: p4 reaches 0.83 by epoch 5; plain needs epoch 16. Roughly 3x, consistent with the
  4x parameter reduction from weight sharing.
- Training is ~20% slower for p4 (materialising the rotated/shifted weights every step). Inference is not,
  since the weights are baked once at export.

Caveats: 64 px and ~10k images, so the absolute numbers are far below production; trained on this library
so nothing is claimed about generalisation; one seed, no error bars. Only the *gap* transfers.

Untested third arm: a plain net with a rotation-consistency loss. That would close some of the plain net's
0.016 single-vs-TTA gap, but cannot reach exactness, and on this evidence would still be starting 2 points
of accuracy and 4x the parameters behind.

**Recommendation for a built-in feature is now evidence-backed:** C4-equivariant backbone, single forward
pass, no TTA, softmax entropy over the four logits as the abstain signal, BatchNorm with group-tied affine
so it folds away at export, trained on the CC-BY Open Images pool with Google's Rotation labels.

## Open Images fetch (2026-09-20 01:20)

**Pool: 1,495,772 CC-BY train images that carry a rotation label.** Far more than needed; taking 150,000.
Each is pulled from `open-images-dataset.s3.amazonaws.com` (the Flickr thumbnail URLs in the metadata are
dead), resized to longest-side 256 and stored as webp q88, with a manifest line recording id, rotation,
author and license so attribution is mechanical. Resumable: it skips ids already on disk.

Storage: aspect ratio is preserved rather than square-cropped. The model input must be square for C4
equivariance to hold (rot90 changes the shape of a non-square image), but **pad-to-square is the right
operation, not centre-crop**: padding the short side symmetrically commutes with rot90, so the whole frame
is kept and the guarantee survives. Deferring that choice to the data loader keeps it changeable without
re-downloading.

## Storage format: webp was the wrong default (measured)

Chose webp by inheritance from the Immich thumbnails, not by reasoning. Measured at 256 px, 300 images:

| format | mean size | decode | per core |
|---|---|---|---|
| WEBP q88 | 11.1 KB | 0.65 ms | 1,537 img/s |
| JPEG q90 | 14.9 KB | 0.20 ms | 5,048 img/s |
| JPEG q85 | 12.0 KB | 0.18 ms | 5,667 img/s |

JPEG decodes 3.3x faster (libjpeg-turbo SIMD vs libwebp) and at q85 is the same size as webp q88, so the
compression argument does not survive measurement. It also keeps the nvJPEG GPU-decode path open, which
webp cannot use at all.

Not changing the current fetch: even webp gives ~12,000 img/s on 8 cores, far more than a small model on
an RTX 3060 will consume, so the training loop will not be dataloader-starved either way. **Use JPEG q85
for any future fetch.**

## Assumption audit (2026-09-20 10:30) - two real bugs found

### 1. BROKEN: Open Images `Rotation` direction was backwards
Assumed it meant degrees clockwise to reach upright (`PIL rotate(-r)`). Verified visually against 8
rotated validation images (`oid_check/rotation_semantics.png`): a barrel, a standing performer, a gold
statue and a disc with readable "COMMUNITY" text are all upright under `rotate(+r)`, i.e. **degrees
counter-clockwise**. Fixed in train.py and sweep.py.

Impact: 1,702 of 149,988 training images (1.1%) were canonicalised 180 degrees off instead of upright,
*systematically* rather than randomly, which is worse. It would specifically poison the 180 class, which
is already the weakest for every model tested.

Note the two datasets use OPPOSITE conventions: Open Images is CCW-to-upright, the Immich library truth
(from the rotate-edit angle, verified visually at 16:05 yesterday) is CW-to-fix. Do not unify them.

### 2. BROKEN: pad-to-square does not commute with rot90
Earlier claim ("padding the short side symmetrically commutes with rot90, so the guarantee survives
preprocessing and the whole frame is kept") is wrong. Centring uses integer division, so an odd size
difference leaves the image 1 px off-centre, and after rotation that offset is on the other axis.
Measured max pixel diff 223/255 between the two orders. Cropping to fix parity is not equivariant either,
because it has to choose an edge.

**Squash-to-square is the operation that commutes**: it resamples the two axes independently, so rotating
merely swaps two 1-D resamplings. Measured 2.4e-5 in float torch (exact), and 10/255 max, 0.106/255 mean
in PIL uint8 on real photos (implementation rounding only). Switched both scripts to squashing.

Why the observed "single == TTA" was still real: evaluation rotates the *tensor* after preprocessing, so
that path was always exactly equivariant. The preprocessing asymmetry only bites on a photo that was
genuinely stored rotated. So measured results stand, but the design advice was wrong.

### 3. HOLDS: "Rotation == 0" really does mean upright
Ran check_orientation with TTA over 250 OID Rotation=0 validation images and 250 human-verified-upright
library images at matched thresholds:

| threshold | OID flagged | library flagged |
|---|---|---|
| 0.80 | 2.0% | 0.8% |
| 0.90 | 0.4% | 0.8% |
| 0.95 | 0.0% | 0.8% |

Open Images upright labels are as clean as our human-verified set, arguably cleaner. **The label-noise
ceiling I worried about is not binding.** (The library's 0.8% is either genuine misses in the human pass
or check_orientation false positives; small either way.)

### 4. HOLDS: network equivariance
Verified repeatedly with *every parameter randomised*, not just at init: 7e-7 for P4Net, 9.5e-7 for
P4MobileNet/V2, and 8.9e-7 after the ONNX bake.

### Still open
- **Sweep noise**: library eval is 1,200 images x4 = 4,800 samples, SE ~0.55 pp. Differences under ~1 pp
  in the sweep table are not meaningful; res128 / res160 / stempool2 are a three-way tie, not a ranking.
- **GMACs vs wall-clock**: partially disproven for full-resolution group convs. Needs real latency
  measurement on GPU and CPU before any efficiency claim is made.
- **Metric mismatch**: 4-way accuracy weights all orientations equally; the product metric is
  precision/recall on genuinely rotated images, benchmarked against check_orientation's 0.935/0.938.
- **Sweep proxy fidelity**: whether 3 epochs on 25k ranks configs the same way 10 epochs on 145k does.
  Unverified, and cheap proxies are known to mis-rank.

## Run 2 relaunch (2026-09-20 11:15) - four corrections

Killed run1 mid-flight. It was measuring my learning-rate schedule, not the architectures:

```
mnv4_pretrained  ep0  loss=0.6491  library=0.8904/0.9337   <- matched the equivariant net's FINAL score
mnv4_pretrained  ep1  loss=0.6613  library=0.5192/0.5886   <- collapsed
mnv4_pretrained  ep9  loss=0.6890  library=0.7791/0.7837   <- never recovered
```

One LR (3e-3, tuned for from-scratch) applied to every arm. The OneCycle peak around epoch 2 destroyed
the pretrained features. The apparent 17-point win for p4_mobile (0.9387 vs 0.7702) is an artefact.
**Pretraining looks valuable**, which was the flagged risk to the pilot conclusion all along.

Corrections in run2:
1. per-arm LR: 3e-4 for pretrained arms, 3e-3 for from-scratch
2. Open Images canonicalisation rotates CCW (was CW) - 1,702 images stop being 180 degrees wrong
3. squash-to-square instead of pad (padding does not commute with rot90)
4. ground truth 688 -> 693, including the 5 the full manual sweep found

Added arms: `p4_deep6` (within noise of best at 73% cost in the sweep) and `p4_res128`.
Added metric: **precision/recall on the library in its STORED orientation**, which is what the feature
is judged on, benchmarked against check_orientation's 0.935/0.938. 4-way accuracy weights all four
orientations equally and flatters a model on a library that is 95% upright: p4_mobile's 0.9387 4-way
implies ~6% false positives at raw argmax vs check_orientation's ~0.4%.


## Overnight plan, 2026-09-20 21:15 CEST -> user back ~10:00

Scaling to 600k (4x) results so far:

| model | 145k library | 600k library | 145k P/R @.9 | 600k P/R @.9 |
|---|---|---|---|---|
| mnv4_consistency | 0.9610/0.9649 | 0.9652/0.9692 | 0.916/0.848 | 0.943/0.834 |
| p4_deep6 | 0.9425 | 0.9514 (ep4/6) | 0.868/0.701 | pending |
| p4_scratch | 0.9477 | 0.9603 (ep3/6) | 0.883/0.788 | pending |

**Key read: 4x data bought precision (0.916 -> 0.943) but NOT recall (0.848 -> 0.834).**
check_orientation's advantage is entirely recall (0.938). So more data is the wrong lever for the
remaining gap; stop assuming scale fixes it.

Note p4_scratch is gaining fastest from data (+1.3 by epoch 3 of 6) and at 190k params / 1 pass is
closing on mnv4_consistency's 2.5M params / 4 passes.

Plan:
1. (running) fetch the rest of the pool to 1.495M on the idle GTX 1660, so the option exists.
2. When run3a frees the 3060 (~22:00): **diagnose the recall gap** - dump per-image predictions and
   check whether the images we miss are the ones check_orientation gets, or whether both miss the same
   ambiguous set and check_orientation simply guesses luckier. This is the cheapest decisive experiment.
3. Depending on (2): either a bigger equivariant model at 600k with more epochs, or a recipe change
   aimed at recall specifically.
4. Keep every result in this file with timestamps.

## Recall post-mortem, 22:00 - the gap is real, not ambiguity

At each model's own threshold giving ~0.93 precision, how many of the 693 rotated it finds:

| model | finds | recall |
|---|---|---|
| check_orientation | 648 | 0.935 |
| mnv4_consistency | 602 | 0.869 |
| p4_scratch | 556 | 0.802 |
| p4_deep6 | 516 | 0.745 |
| p4_mobile | 415 | 0.599 |
| plain_scratch | 383 | 0.553 |
| mnv4_scratch | 59 | 0.085 |

Overlap against check_orientation on the rotated set:

| model | both | only co | only ours | NEITHER |
|---|---|---|---|---|
| p4_scratch | 548 | 100 | 8 | 37 |
| mnv4_consistency | 596 | 52 | 6 | 39 |
| p4_deep6 | 513 | 135 | 3 | 42 |

**Two conclusions.** check_orientation's set is very nearly a strict superset of ours (it finds 100 that
p4_scratch misses; p4_scratch finds 8 it misses), so this is a genuine deficiency, not luck on ambiguous
images. And only 37-42 images are missed by everything, which puts the achievable recall ceiling near
0.94 - **check_orientation is already at it.**

**What explains the gap:** pretraining. Identical MobileNetV4 finds 602 images pretrained vs 59 from
scratch. Meanwhile p4_scratch, from scratch with 13x fewer parameters, finds 556. Equivariance is buying
roughly what pretraining buys, and the two have never been combined because no pretrained equivariant
backbone exists.

## 22:12 - distillation experiment

Rather than pretrain an equivariant backbone (expensive), distil: freeze mnv4_consistency as teacher,
train the equivariant students against its TTA-aggregated soft targets plus the hard label
(alpha 0.3 hard / 0.7 soft). The teacher's soft targets encode what pretraining taught it.

Teacher choice is deliberate: mnv4_consistency is ours and built on Apache-2.0 weights. check_orientation
would be a better teacher (0.935 recall) but its weights are CC-BY-NC, and distilling from it would carry
that restriction into the student.

Running: kd-deep6 on the 3060, kd-scratch on the 5060, both N_TRAIN pinned to 594,609 to match the
existing baselines exactly (the fetch job had grown the pool to 1.09M mid-launch, which would have
confounded distillation with extra data). ETA ~01:15.

## 2026-09-21 08:45 - distillation: negative result

Students trained against a frozen mnv4_consistency teacher (alpha 0.3 hard / 0.7 soft, TTA-aggregated
targets), same 594,609 images as the baselines.

| model | no KD | with KD |
|---|---|---|
| p4_deep6 library | 0.9509 | 0.9531 |
| p4_deep6 @ P~0.90 | R=0.779 | R~0.79 |
| p4_scratch library | 0.9610 | 0.9590 |
| p4_scratch @ P~0.92 | R=0.824 | R~0.81 |

All within noise. **Distillation does not help here.** Diagnosis: the teacher is only moderately better
than the students (602 vs 556 of 693 found), so there is little margin to transfer, and the hard labels
already carry most of that signal. KD pays when the teacher is far ahead; ours was not.

Dataset now 1,094,757 images. Nothing has trained on more than 595k.

### Standing summary

| model | params | passes | recall @ P~0.93 | shippable |
|---|---|---|---|---|
| check_orientation | 23M | 4 | 0.935 | **no** (CC-BY-NC weights) |
| mnv4_consistency | 2.5M | 4 | 0.869 | yes (Apache-2.0 + CC-BY data) |
| p4_scratch | 190k | 1 | 0.802 | yes |
| p4_deep6 | 44k | 1 | 0.745 | yes |

Achievable recall ceiling ~0.94 (37-42 of 693 defeat every model). check_orientation is at it but cannot
ship. Our best shippable model is 6.6 points behind it and needs 4 passes.

### Untried levers, in order of expected value
1. **More data on the best models** - 1.09M is now available, 1.8x what anything has seen. Weak prior:
   145k->600k improved precision, not recall.
2. **A bigger equivariant student.** Every equivariant model tested is <=190k params against
   check_orientation's 23M. Capacity is the most obvious untested axis.
3. **Pretrain the equivariant backbone** on generic classification, then fine-tune. Expensive but it is
   the one thing that would combine the two effects that each independently work.
4. Recipe: label smoothing, EMA, longer cosine schedule, stronger augmentation.

## 2026-09-21 10:55 - detection vs confidence: the gap is calibration, not sight

Splitting argmax recall (does the model pick the right angle at all) from usable recall (at P>=0.93):

| model | argmax recall | R@P=0.93 | lost to confidence |
|---|---|---|---|
| p4_deep6 | 0.925 | 0.745 | 0.180 |
| p4_scratch | 0.957 | 0.808 | 0.149 |
| mnv4_pretrained | 0.947 | 0.854 | 0.092 |
| mnv4_consistency | 0.961 | 0.872 | 0.089 |
| ensemble p4_scratch+mnv4_consistency | 0.965 | 0.887 | 0.078 |
| check_orientation | 0.978 | 0.938 | 0.040 |

**The detection gap to check_orientation is 2.1 points, not 13.** Nearly all of the usable-recall gap is
confidence. Of the 78 rotated images the ensemble misses, 41 are ones where it picked the CORRECT angle
at mean confidence 0.646 while check_orientation was at 0.929. Those misses are overwhelmingly printed
text (newspaper, book and magazine pages) - see sheet_gettable.png.

**This corrects the 22:00 post-mortem.** "check_orientation finds things we don't" was wrong; it is
confident about things we are merely right about. It also flips a ranking: on raw detection the
equivariant p4_scratch (0.957) BEATS pretrained MobileNetV4 (0.947); it only loses on usable recall
(0.808 vs 0.854). Pretraining's contribution here is calibration, not discrimination.

4-way accuracy numbers throughout are argmax-based and were never distorted by this.

Temperature scaling cannot help: it is monotonic, so the precision/recall curve is unchanged. The
confidence *ranking across images* is what is wrong.

**Free win already measured:** ensembling p4_scratch + mnv4_consistency gives R=0.887 at P=0.936, up from
0.872 for the best single model, with no retraining.

**Now-prioritised ideas** (all target confidence ranking, not features):
1. OCR fusion - independent, near-perfect signal on exactly the failing class (41/41 correct where it
   voted, measured 2026-09-19). Low coverage but that is fine as a confidence booster.
2. Higher input resolution (320/384) - never tested above 224; text needs pixels.
3. Focal loss or similar - down-weights easy examples, raising relative confidence on hard ones.
4. Ensemble (already +1.5 points, free).

## 2026-09-21 11:15 - THE HEADLINE CORRECTION: the curves cross, AP is tied

Benchmarking at a single operating point (P>=0.93) was misleading. Full precision/recall curves:

| model | AP | R@P0.85 | R@P0.90 | R@P0.93 | R@P0.95 | R@P0.98 |
|---|---|---|---|---|---|---|
| p4_deep6 | 0.880 | 0.811 | 0.779 | 0.747 | 0.696 | 0.622 |
| p4_scratch | 0.921 | 0.877 | 0.844 | 0.808 | 0.772 | 0.680 |
| mnv4_pretrained | 0.925 | 0.899 | 0.882 | 0.854 | 0.827 | 0.768 |
| mnv4_consistency | 0.943 | 0.912 | 0.896 | 0.873 | 0.860 | **0.818** |
| ensemble p4_scratch+mnv4_consistency | **0.949** | 0.938 | 0.913 | 0.890 | 0.866 | 0.810 |
| check_orientation | 0.950 | 0.955 | 0.951 | 0.938 | 0.899 | **0.387** |

**Average precision is tied: 0.949 vs 0.950.** The curves cross. check_orientation wins between P0.85 and
P0.93; we win above P0.95, and at P0.98 it collapses to 0.387 recall while ours hold above 0.81 - more
than twice as good.

**This overturns the "13 points behind on recall" framing used all through 2026-09-20/21.** That number
came from evaluating at P=0.93, which is check_orientation's strongest region and our weakest.

**High precision is the regime this feature needs.** The whole lesson from the 600-image auto-apply pass
was that false positives cost the user time. At P=0.98 our model finds 81% of rotated images with one
wrong call in fifty; check_orientation finds 39%.

Mechanism: check_orientation's confidence is high but contaminated - it is confidently wrong often
enough that its top bucket cannot reach high precision. Our models hedge more, which costs recall
mid-curve but makes their top-end confidence genuinely informative.

**Lesson: never compare detectors at one operating point. Plot the curve.**

## 2026-09-21 11:20 - OCR fusion: dead, for two independent reasons

1. **Coverage.** Only 24 of the 693 rotated library images have >=3 OCR boxes (428 of the ~12,659 upright
   do). Even a perfect signal reaches 3.5% of the target class. The newspaper pages that dominate the
   failing set mostly have no OCR results at all.
2. **Measurement is blocked.** `GET /assets/:id/ocr` returns boxes transformed into the EDITED frame, and
   every rotated image carries a rotate edit, so the orientation is normalised away: mean tall-box
   fraction is 0.147 for rotated vs 0.030 for upright, where it should be near 1.0 if the frame were
   original. Cannot be evaluated on this test set without re-running OCR on unedited images.

Dropping it. In production the transform issue would not arise (OCR runs before any edit), but the
coverage problem would remain.

## 2026-09-21 11:35 - data is the lever that works

p4_scratch (189,712 params, C4-equivariant, SINGLE forward pass):

| training set | library (full) | P/R @0.9 |
|---|---|---|
| 594,609 | 0.9610 | 0.922 / 0.824 |
| 1,089,589 | 0.9652 | **0.933 / 0.846** |

Nearly doubling the data improved precision AND recall. Distillation and 4x capacity both failed to move
either. For reference mnv4_consistency @600k is 0.9652 library with 2.5M params and 4 passes; p4_scratch
now matches it on one pass with 13x fewer parameters.

cap-test (p4_big, 757k params, 4x the channels of p4_scratch, same 595k data) tracked the p4_scratch
baseline almost exactly through epoch 2 (0.9531 vs 0.9537) - **capacity is a dead end**, consistent with
the finding that these models can already see the rotation and the gap is confidence.

## Correction: "capacity is a dead end" was premature

cap-test (p4_big, 757k params) tracked the p4_scratch baseline at epoch 2 (0.9531 vs 0.9537) but is
AHEAD at epoch 3 (0.9646 vs 0.9603). Capacity may pay off late rather than not at all. Not settled until
the run finishes; the 11:35 note above should be read with that caveat.

## 2026-09-21 12:30 - CORRECTION: capacity works after all

cap-test finished. p4_big (757,280 params, 4x the channels of p4_scratch), same 594,609 images:

| model @595k | params | library | P/R @0.9 | P/R @0.95 |
|---|---|---|---|---|
| p4_scratch | 189,712 | 0.9610 | 0.922 / 0.824 | - |
| **p4_big** | 757,280 | **0.9682** | 0.913 / **0.877** | 0.952 / 0.849 |

**+5.3 points of recall from 4x capacity.** My 11:35 note calling capacity "a dead end" was wrong - based
on an epoch-2 snapshot where the two were tied. The gain appears late in training. Third time today a
partial view produced a wrong conclusion (operating point, compute utilisation, and now this).

Also: p4_big @595k (library 0.9682) ~= p4_scratch @1.09M (0.9652), so **4x capacity is worth about 2x
data**, and both levers are live. p4_big @595k at P=0.952/R=0.849 also beats p4_scratch @1.09M
(P=0.956/R=0.809), so capacity is currently the stronger of the two.

Cost note: p4_big is ~3 GMACs @224, close to p4_scratch's ~2.25 GMACs @384 - so the running resolution
experiment is effectively capacity-vs-resolution at comparable budget.

## 2026-09-21 15:00 - resolution: 224 control done, 384 losing at matched epochs

Both arms on the SAME 594,700 images stored at 512 px, evaluated on library PREVIEWS downscaled to 512
(`testset/thumbs512`), so both ends of the pipeline have real resolution rather than upscaled thumbnails.

| epoch | 224px | 384px |
|---|---|---|
| 0 | 0.9080 | 0.9157 |
| 1 | 0.9411 | 0.9377 |
| 2 | 0.9514 | 0.9263 |
| 3 | 0.9577 | - |
| final | **0.9575** (full lib) | running |

384 led at epoch 0, fell behind by epoch 1, and dipped at epoch 2 - while costing 2.4x the wall clock
per epoch (2710 s vs 1140 s). Provisional but leaning clearly negative for resolution.

res224 product metric: thr0.8 P=0.841 R=0.883 | thr0.9 P=0.913 R=0.833 | thr0.95 P=0.945 R=0.775.

**Caveat: this control is NOT comparable to the earlier 0.9610/0.9652 p4_scratch numbers** - the test set
changed from 250 px thumbnails to 512 px previews. Only res224 vs res384 is a valid comparison.

## 15:02 - combining the two levers that work

Launched `big-full`: p4_big (757k params) on the full 1,089,589 images at 224 px, original 256 px source
and thumbnail test set so it IS comparable to p4_scratch @1.09M (0.9652) and p4_big @595k (0.9682).
Capacity gave +5.3 recall, data gave +2.2 recall; this asks whether they add.

## 2026-09-21 17:06 - RESOLUTION: 384 is worse, and the reason is a domain gap

Both arms: same 594,700 images stored at 512 px, same library-preview test set.

| | 224 px | 384 px |
|---|---|---|
| Open Images val (final) | 0.9270 | **0.9460** |
| library (full) | **0.9575** | 0.9142 |
| s / epoch | 1140 | 2710 |

**The 384 model is BETTER on its own validation set and 4.3 points WORSE on the real target.** Per-epoch
it is unambiguous: OID climbs every epoch (0.8588 -> 0.9460) while library peaks at epoch 1 (0.9377) and
then falls to 0.9177. Training loss drops throughout.

**This is not training instability** (my first read at 16:30 - wrong). It is a domain gap: at higher
resolution the model can see fine detail - JPEG artefacts, sharpening, sensor noise - that differs
between Open Images and Immich previews, and it latches onto those instead of structural cues. At 224 px
that detail is destroyed, forcing it to rely on horizon/face/text-line structure, which transfers.

**Conclusion: train at 224. Resolution is not the lever, and going higher is actively harmful for
cross-domain transfer.** This also retro-justifies the original 256 px storage decision that I criticised
this morning as a self-inflicted constraint - it was accidentally the right call.

Caveat: only tested at 224 vs 384 with one recipe. A lower LR or stronger augmentation might close some
of the domain gap, but the direction of the OID/library divergence says the problem is what the model is
learning, not how fast it is learning it.

## 2026-09-21 18:15 - the 384 result was CONFOUNDED by my own encode pipeline

Re-evaluated the trained 384 model on the library with only the test-image CODEC changed:

| library test images | 4-way accuracy |
|---|---|
| as stored (WebP q90, Immich preview -> PIL bicubic -> webp) | 0.9142 |
| re-encoded to JPEG q85 (matching the training codec) | **0.9369** |

**+2.3 points from changing nothing but the codec of the test images.** Same weights, same content.
That is over half the apparent gap to the 224 control (0.9575).

The two datasets went through different pipelines entirely, which I created:
- training: S3 original -> GPU bicubic to 512 -> nvJPEG q85 (.jpg)
- library test: Immich preview -> PIL bicubic to 512 -> WEBP q90 (.webp)

**So "train at 224, resolution is harmful" (17:06) is not supported.** Roughly half that penalty was my
data handling, not resolution. The remaining ~2 points may still be a real resolution effect, but the
experiment needs redoing with matched pipelines before anything is concluded.

**Principle (user's framing, and it is the right one): train on what you serve.** The training corpus
should be processed through Immich's own encode path - libvips, webp, quality 80 - not a different one.
A 190k-parameter model has little spare capacity, so low-level codec statistics are cheap signal for it
to latch onto.

## Two process failures exposed at the same time

1. **Output collision.** Every run wrote to `runs/<ARCH>/`, so runs of the same architecture
   overwrote each other. big-full destroyed cap-test's `hist.json`. Fixed: `RUN_ID` now defaults to the
   job name.
2. **cap-test is unverifiable.** Its job was deleted (no logs), its history was overwritten, AND it
   overlapped with the webp->jpg conversion that deleted the very files it was reading. **The capacity
   finding (+5.3 recall) should be treated as unconfirmed** until big-full completes; big-full is the
   same architecture on the full data and at epoch 1 is already at 0.9629, which is consistent.

## 2026-09-21 18:15 - rebuilt both datasets to match Immich exactly

**Immich's real encode path** (read from `media.repository.ts`, verified against live previews):
`sharp: autorotate -> pipelineColorspace srgb + ICC -> resize(size,size,{fit:'outside',
withoutEnlargement:true}) -> toFormat(fmt,{quality, chromaSubsampling: q>=80 ? '4:4:4':'4:2:0',
progressive:false})`. Defaults: thumbnail WebP/250/q80, **preview JPEG/1440/q80**.

Three things I had wrong by approximating:
1. **`fit:'outside'` sizes by the SHORTEST side**, not the longest. A 1440 preview of a 4:3 landscape is
   1920x1440. I had assumed longest-side throughout.
2. **chromaSubsampling 4:4:4** at q>=80. PIL defaults to 4:2:0, so every re-encode I did discarded chroma
   resolution Immich keeps.
3. Colourspace/ICC handling and progressive=false.

Verified the replication against 3 real previews: **dimensions exact**, pixels within ~1% (mean |diff|
2.4/255). The residual is identical across resize/thumbnail/strip variants, so it is JPEG encoder version
(sharp's bundled libvips vs system), not pipeline. Judged good enough - the codec effect we measured
(webp vs jpeg) was worth 2.3 accuracy points; 1% encoder noise is a different order.

**Test set** (`testset/prev_raw`): 13,352 previews downloaded VERBATIM, no re-encode at any stage.
4.7 GB, all JPEG shortest-side 1440. Exact by construction - no replication needed.

**Training set** (`oid/img_v2`): originals -> JPEG q80 4:4:4 (the preview step; OID originals are 1024 px
so `withoutEnlargement` means no resize) -> downscale shortest-side 256 -> store q95 4:4:4. The
preview-stage artefacts are baked in, which is what the model meets at inference.

**Originals are KEPT this time** in `oid/orig` (~248 GB).

Also fixed: `RUN_ID` so runs no longer overwrite each other's output directory.

### 2026-09-21 19:40 — matched-data (v2) set built, reruns launched

`derive-v2` completed in 23 min (10 parallel processes, ~360 img/s aggregate vs 59 img/s
single-process). Verified:

- `oid/orig` — 600,000 Open Images originals, 181 GB, **kept this time**
- `oid/img_v2` — 600,000 derived training images, 33 GB
- `oid/manifest_v2.jsonl` — 600,000 rows, matches the image count exactly

Derivation is the Immich-matched path (`scratchpad/derive.py`): q80 4:4:4 progressive=false
re-encode (the preview step; originals are 1024 px longest side so `withoutEnlargement` means no
resize happens), then LANCZOS downscale to shortest-side 256, stored at q95 4:4:4. Test set is
`testset/prev_raw` — the 13,352 Immich previews byte-for-byte, JPEG.

**Caveat worth writing down now:** v2 is 600k images, v1 (`oid/img`, webp) was 1,094,757. So v2
numbers are **not** directly comparable to any v1 run — the training set is 45% smaller as well as
differently encoded. Within v2 the comparisons are clean:

- `v2-224` (turing, p4_scratch, res 224, bs 96, 6 ep) — the v2 baseline
- `v2-384` (blackwell, p4_scratch, res 384, bs 32, 6 ep) — settles resolution on matched data
- `v2-big` (ampere, p4_big, res 224, bs 64) — queued behind big-full; settles capacity against v2-224

Both reruns launched 19:38. Source verified in sync with local
`train.py`/`models.py` before launch (per-arm LR fix and squash-to-square both present).

Ran three GPU jobs concurrently (big-full on ampere + these two on turing/blackwell) — three
distinct physical cards, nothing else contending.

### big-full progress (p4_big, v1 data, 1.09M)

```
ep 0 loss=0.4715 | oid 0.8842 | library 0.9303 | 4386s
ep 1 loss=0.2452 | oid 0.9282 | library 0.9629 | 4377s
ep 2 loss=0.1921 | oid 0.9360 | library 0.9666 | 4375s
```

At epoch 2 it is already past the p4_scratch baseline (library 0.9610). ETA ~22:40 for epoch 5.

### 2026-09-21 20:08 — v2-224 crashed on a truncated derived image

```
PIL.UnidentifiedImageError: cannot identify image file 'oid/img_v2/34ae8f7dad65a7c2.jpg'
```

Zero bytes on disk; the original (`oid/orig/34ae8f7dad65a7c2.jpg`, 870 KB) is intact, so this
is a truncated write from the 10-way parallel derivation, not a bad source. Re-derived the single
file through the same `derive.py` path (76 KB, 256x383). `find -size 0` over `img_v2` returns
exactly one hit, now fixed; a full first-two-bytes/last-two-bytes JPEG marker scan of all 600k is
running to confirm nothing else is silently truncated.

`v2-384` was already past dataset construction (train 595000, 18593 steps/epoch at bs 32) and is
still running — it will hit the same file later in epoch 0 if the set isn't clean, which is what the
scan is for.

big-full epoch 3: `loss=0.1610 | oid 0.9418 | library 0.9711 | 4377s`.

### 2026-09-21 20:25 — v2 image set verified clean, both reruns training

Full JPEG marker scan of `img_v2` (SOI at byte 0, EOI at the last two bytes, size >= 1 KB) over all
600,000 files: **0 bad**. The zero-byte `34ae8f7dad65a7c2.jpg` was the only casualty of the 10-way
parallel derivation and is repaired. `v2-224` relaunched and is past the batch that killed it.

First matched-data epoch:

```
v2-384  ep 0 loss=0.5401 | oid 0.8724 | library 0.9249 | 2716s
```

At 2716 s/epoch, `v2-384` finishes 6 epochs around 00:15. `v2-224` is on turing at 6197 steps/epoch
with no epoch line yet after 13 min.

### 2026-09-21 20:56 — pacing check

`v2-224` is on a GTX 1660 (turing claim), GPU at 100%, 3986 MiB — training fine, just slow: 44 min
in with no epoch line. Per-step compute is near-identical between the two reruns (bs 96 x 224^2 =
4.8 Mpx vs bs 32 x 384^2 = 4.7 Mpx), so the gap is purely card speed. Extrapolating from v2-384's
2716 s for 18593 steps, the 1660 looks to be running ~6x slower per step, which puts `v2-224` at
roughly 90 min/epoch and ~9 h for 6 epochs (ETA ~05:45).

Revised schedule: big-full ~22:40, `v2-big` on the freed ampere ~22:45-02:45 (p4_big at 224 on 600k
should be ~2400 s/epoch, scaled from big-full's 4377 s on 1.09M), `v2-384` ~00:15, `v2-224` ~05:45.
All four land before morning.

### 2026-09-21 21:25 — progress

```
big-full  ep 4 loss=0.1360 | oid 0.9488 | library 0.9734 | 4376s
v2-384    ep 1 loss=0.2948 | oid 0.9048 | library 0.9503 | 2715s
```

big-full has one epoch left (ETA ~22:40) and is still improving. `v2-224` at 75 min with no epoch
line yet, consistent with the ~90 min/epoch estimate for the 1660.

All three cards confirmed GPU-bound, not input-bound: RTX 3060 100% / 150 W of 170, RTX 5060 99% /
108 W of 145, GTX 1660 100% / 66 W of 130. Training uses bf16 autocast, so the 1660 (Turing TU116,
no tensor cores, no native bf16) is the one card getting nothing from it.

### 2026-09-21 22:06 — the GTX 1660 is a dead end for this workload

`v2-224` is 116 min in with still no epoch line, well past the 90 min/epoch estimate. The card is at
100% "utilization" and 1905 MHz but drawing only **46 W of a 130 W limit** — nvidia-smi utilization
only means a kernel was resident, and 46 W says the SMs are mostly idle inside tiny launch-bound
kernels. Two things compound here:

1. p4 group convs at these widths (8-64 channels x 4 group) are small kernels; the GPU is
   launch-bound rather than compute-bound.
2. Training runs `torch.amp.autocast(dtype=torch.bfloat16)`. Turing TU116 has no tensor cores *and*
   no native bf16 (bf16 arithmetic intrinsics start at sm_80), so on this card autocast is pure
   conversion overhead with nothing to gain — plausibly slower than plain fp32 would be.

Point 2 is worth carrying into the feature design: if a shipped model ever does bf16 inference,
pre-Ampere self-hosters get a pessimisation, not a speedup.

**Plan change.** Rather than let the 1660 grind ~12 h, migrate `v2-224` to the RTX 5060 when
`v2-384` frees it (~00:22); at 0.146 s/step x 6197 steps that is ~15 min/epoch, ~1.5 h for the full
run. Revised finish: `v2-big` on the 3060 from ~22:45 to ~02:45, `v2-224` on the 5060 from ~00:22 to
~02:00. Everything lands by ~02:45 instead of ~08:30.

```
v2-384  ep 2 loss=0.2318 | oid 0.9218 | library 0.9549 | 2712s
```

### 2026-09-21 22:27 — big-full FINISHED (and beat the drain by 3 minutes)

A host reboot landed at ~22:30. `big-full` completed at 22:27 CEST, three minutes ahead of it.
`v2-224` and `v2-384` were both killed mid-epoch and restarted from scratch, losing ~2.5 h and
~1.9 h respectively. They restart with the patched
`train.py` (per-epoch `model.pt`), so a second drain would cost less.

**big-full — p4_big, 757,280 params, res 224, v1 data (1.09M, webp), 6 epochs, RTX 3060, 7h18m**

| ep | loss | oid | library(3500) |
|----|------|-----|---------|
| 0 | 0.4715 | 0.8842 | 0.9303 |
| 1 | 0.2452 | 0.9282 | 0.9629 |
| 2 | 0.1921 | 0.9360 | 0.9666 |
| 3 | 0.1610 | 0.9418 | 0.9711 |
| 4 | 0.1360 | 0.9488 | 0.9734 |
| 5 | 0.1195 | 0.9524 | 0.9740 |

Full library (13,352 x4): single 0.9715, tta 0.9715 (identical, as required — the net is exactly
equivariant, so TTA is a no-op by construction).

Product metric (13,352 stored, 693 genuinely rotated):

| thr | P | R |
|-----|---|---|
| 0.5 | 0.700 | 0.971 |
| 0.8 | 0.863 | 0.935 |
| 0.9 | **0.914** | **0.905** |
| 0.95 | 0.944 | 0.870 |

Against the p4_scratch baseline (library 0.9610, R=0.824 at P~0.92) this is a large gain: **R 0.905
vs 0.824 at comparable precision**, and R=0.870 at P=0.944.

**It does not cleanly settle capacity, though.** big-full changed two things at once against that
baseline — 757k params *and* the full 1.09M training set. That is exactly the conflation `v2-big` vs
`v2-224` is designed to remove, since both will run on the same 600k v2 set. Treat this number as
"capacity + data together are worth ~8 points of recall", not as a capacity result.

Also note the curve is still rising at epoch 5 (0.9734 -> 0.9740 is flattening on the subset, but
loss is still dropping steadily). 6 epochs may be short for this model.

### 2026-09-22 00:05 — first matched-data epochs (post-drain restarts)

```
v2-384  ep 0 loss=0.5469 | oid 0.8792 | library 0.9263 | 2719s
v2-big  ep 0 loss=0.5727 | oid 0.8802 | library 0.9303 | 2482s
```

`v2-384` epoch 0 reproduced its pre-drain value almost exactly (0.9263 now vs 0.9249 before), which
is a useful free check that the restart is clean and run-to-run noise at this stage is ~0.15 points.

`v2-big` at 2482 s/epoch puts it at ~03:11, matching the estimate. Its epoch 0 library score (0.9303)
is identical to big-full's epoch 0 on the 1.09M v1 set — too early to mean much, but the trajectories
start together.

`v2-224` still has no epoch line at 95 min on the 1660.

### 2026-09-22 00:37 — epoch 1, and a caveat on "exact equivariance"

```
v2-384  ep 1 loss=0.2923 | oid 0.9098/0.9098 | library 0.9466 | 2714s
v2-big  ep 1 loss=0.2898 | oid 0.9090/0.9088 | library 0.9511 | 2481s
```

**`v2-big` epoch 1 is the first time single and TTA have disagreed: 0.9090 vs 0.9088.** On 5,000
validation images that is a single image flipping. The equivariance proof is exact in real
arithmetic, and the ONNX export check confirmed it, but evaluation runs under bf16 autocast, where an
image whose top-2 logits are nearly tied can have its argmax flipped by rounding. So the accurate
claim is "TTA is a no-op up to float precision", not "bit-identical" — worth stating precisely since
the whole architectural argument rests on it.

**Third job is not costing the other two anything.** The metrics API is still down post-upgrade, so
instead: `v2-384` epoch times are 2716 / 2715 / 2712 s pre-drain and 2719 / 2714 s post-drain, with
`v2-big` starting midway through that epoch 1. No measurable slowdown, so CPU contention between
three concurrent runs is not biting and `v2-224` can keep its insurance slot on the 1660.

### 2026-09-22 01:08 — the 1660's real epoch time, measured

Snapshotting `hist.json` turned up an epoch-0 record for `v2-224` that the job log did not show — it
is **left over from the pre-drain run** (19:38-22:30 on the 1660), not the current one, which
restarted at 22:57 and reaches its own epoch 0 around 01:11 and will overwrite it. Saved separately
as `hist/v2-224_turing_predrain.json`.

It answers the question I had been estimating:

```
p4_scratch res 224, GTX 1660: ep 0 loss=0.6633 | oid 0.8568 | library 0.9123 | 8038s
```

**8038 s = 134 min/epoch**, so a full 6-epoch run on the 1660 is 13.4 h. My earlier extrapolation from
the 5060 said ~90 min/epoch; the real figure is 50% worse, which makes the launch-bound/no-bf16
diagnosis look, if anything, understated. The migration to blackwell at ~03:30 is emphatically the
right call.

Progress:

```
v2-384  ep 1 loss=0.2923 | oid 0.9098 | library 0.9466 | 2714s
v2-big  ep 2 loss=0.2200 | oid 0.9254 | library 0.9603 | 2481s
```

Very early and not yet comparable (different epoch counts), but the epoch-0 library scores line up as
224 = 0.9123 < 384 = 0.9263 < p4_big@224 = 0.9303.

### 2026-09-22 01:40 — run-to-run noise is larger than I claimed

The restarted `v2-224` reached its own epoch 0: `loss=0.6246 | oid 0.8522 | library 0.9017 | 8040s`
(confirming 134 min/epoch on the 1660 to within 2 s).

Two identical `v2-224` runs now have epoch-0 library scores of **0.9123 and 0.9017 — a 1.06 point
spread**. Earlier I read `v2-384`'s two epoch-0s (0.9249 / 0.9263) as showing ~0.15 points of noise;
that was too optimistic a generalisation from one pair. Early-epoch scores for the small model are
roughly an order of magnitude noisier than that.

Practical consequence: **do not read anything into sub-point gaps**, and compare only final-epoch
numbers, ideally with the product-metric curve rather than a single accuracy. The intermediate epoch
tables in this file are pace checks, not results.

Matched epoch 2 on the v2 data so far:

| run | arch | res | library @ ep2 |
|---|---|---|---|
| `v2-384` | p4_scratch 190k | 384 | 0.9566 |
| `v2-big` | p4_big 757k | 224 | 0.9603 |

### 2026-09-22 03:13 — v2-big finished its 6 epochs (final eval still running)

```
v2-big  ep 5 loss=0.1305 | oid 0.9436 | library 0.9734 | 2472s
v2-384  ep 4 loss=0.1705 | oid 0.9378 | library 0.9631 | 2715s
```

`v2-big`'s epoch-5 library score on the 3,500 subset is **0.9734 — exactly big-full's 0.9734**, same
architecture on 1.09M v1 webp images vs 600k v2 codec-matched ones. Half the data, same score. That
is the strongest evidence yet that the codec match is doing real work: it appears to buy back roughly
the 45% of training data we lost. Holding judgement until the product metrics land, since the subset
accuracy is the coarsest of the three numbers.

Also note `v2-big` is slightly *behind* big-full at matched epochs 3 and 4 (0.9671 vs 0.9711, 0.9714
vs 0.9734) and only catches up at epoch 5 — so the data-size deficit is real early and closes late.

### 2026-09-22 03:34 — v2-big DONE, and it overturns what I said 20 minutes ago

**v2-big — p4_big, 757,280 params, res 224, v2 data (600k, codec-matched), 6 epochs, RTX 3060, 4h30m**

oid 0.9436 | library(3500) 0.9734 | library(full 13352) single 0.9718, tta 0.9718

| thr | P | R |
|-----|---|---|
| 0.5 | 0.698 | 0.958 |
| 0.8 | 0.864 | 0.918 |
| 0.9 | 0.911 | 0.876 |
| 0.95 | 0.952 | 0.836 |

**Correction to the 03:13 entry.** I wrote that v2-big matching big-full's 0.9734 subset accuracy was
"the strongest evidence yet that the codec match is doing real work... it buys back roughly the 45% of
training data we lost." The product metric says otherwise:

| thr | big-full R (1.09M, webp) | v2-big R (600k, matched) |
|-----|---|---|
| 0.5 | 0.971 | 0.958 |
| 0.8 | 0.935 | 0.918 |
| 0.9 | 0.905 | 0.876 |
| 0.95 | 0.870 | 0.836 |

big-full is **3 points better on recall at P~0.91** and ahead at every threshold; v2-big only edges it
on precision at thr0.95 (0.952 vs 0.944). The identical 0.9734 subset numbers were a coincidence of
the coarsest metric — exactly the trap the 01:40 noise entry warned about, which I then walked into.

So on the evidence so far, **more data beat the codec match**, and "train on what you serve" did not
pay for a 45% smaller training set. Two caveats before treating that as settled:

1. It is confounded — big-full differs from v2-big in *both* data volume and training codec.
2. The original codec finding was about the **test** set encoding (library as webp 0.9142 vs JPEG q85
   0.9369), i.e. evaluating on images encoded unlike the ones the model saw. That is a different
   mechanism from re-encoding the **training** set, and this result does not contradict it. What it
   does undercut is my extrapolation from one to the other.

Deriving the remaining ~490k originals to v2 and rerunning p4_big would disentangle this properly.

```
v2-384  ep 5 loss=0.1550 | oid 0.9402 | library 0.9671 | 2716s   (final eval running)
```

### 2026-09-22 03:45 — v2-384 DONE

**v2-384 — p4_scratch, 189,712 params, res 384, v2 data (600k), 6 epochs, RTX 5060, 5h15m**

oid 0.9402 | library(3500) 0.9671 | library(full 13352) single 0.9655, tta 0.9655

| thr | P | R |
|-----|---|---|
| 0.5 | 0.663 | 0.960 |
| 0.8 | 0.864 | 0.922 |
| 0.9 | 0.928 | 0.875 |
| 0.95 | 0.952 | 0.830 |

**Resolution and capacity look interchangeable.** Side by side on identical 600k v2 data and 6 epochs:

| | params | res | P@thr0.9 | R@thr0.9 | P@thr0.95 | R@thr0.95 | lib(full) |
|---|---|---|---|---|---|---|---|
| `v2-384` | 189,712 | 384 | **0.928** | 0.875 | 0.952 | 0.830 | 0.9655 |
| `v2-big` | 757,280 | 224 | 0.911 | 0.876 | 0.952 | 0.836 | 0.9718 |

Recall is a dead heat (0.875 vs 0.876 at thr0.9; 0.830 vs 0.836 at thr0.95) and precision at thr0.95
is identical to three decimals. `v2-384` is actually ahead on precision at thr0.9, `v2-big` ahead on
whole-library 4-way accuracy. **A 4x smaller model at 384 buys the same product-metric performance as
a 4x larger model at 224.**

That reframes the cost question: it is not "resolution or capacity", it is which of the two is cheaper
to run. p4_scratch at 384 processes (384/224)^2 = 2.94x the pixels with a quarter of the channels;
p4_big at 224 does the reverse. Needs an actual FLOP/latency measurement to call, which is the obvious
thing to do once `v2-224` lands.

Migrated `v2-224` to the freed RTX 5060 at 03:45 (killed the 1660 job mid-epoch-1; the pre-drain and
post-drain epoch-0 records are already saved). ETA ~05:00.

### 2026-09-22 04:20 — inference cost measured (not estimated)

Ran `scratchpad/bench_cost.py`: `torch.utils.flop_counter.FlopCounterMode` for MACs
and wall-clock for latency, CPU with `torch.set_num_threads(4)` (a realistic self-hoster budget).
Equivariant nets need **one** pass; check_orientation needs **four**.

| model | params | GMACs/img | CPU ms/img |
|---|---|---|---|
| p4_scratch@224 | 189,712 | 0.766 | **22.5** |
| p4_scratch@384 | 189,712 | 2.251 | **59.3** |
| p4_big@224 | 757,280 | 2.977 | **43.2** |
| p4_big@384 | 757,280 | 8.748 | 155.7 |
| resnext50_32x4d@224 x4 TTA (check_orientation) | 25,028,904 | 16.922 | **313.6** |

**This settles the resolution-vs-capacity tie, and not the way MACs suggest.** `v2-384` and `v2-big`
scored the same on the product metric, but:

- p4_big@224 uses **32% more MACs** than p4_scratch@384 (2.977 vs 2.251)
- ...yet runs **27% faster** on CPU (43.2 ms vs 59.3 ms)

MACs are the wrong currency here. Small channel counts at high resolution are memory-bound and
vectorise badly; wider channels at lower resolution hit far better GEMM efficiency. **For equal
accuracy, capacity at 224 is the cheaper deployment than resolution at 384** — and it would have been
easy to conclude the opposite from a FLOP count alone.

Against the shipping budget: p4_big@224 is **7.3x faster than check_orientation** (43.2 ms vs 313.6 ms)
and 132x smaller (0.76M vs 25.0M params), while its product metric (R=0.876 at P=0.911) is close to
check_orientation's measured P=0.94/R=0.94 at 0.8 but not yet matching it. p4_scratch@224 at 22.5 ms
is 14x faster; whether it is accurate enough is exactly what `v2-224` will say.

`v2-224` on the 5060 is running at **929 s/epoch** (vs 8040 s on the 1660 — 8.7x). ep 0: loss=0.6406,
oid 0.8632, library 0.9140. Six epochs plus final eval puts it at ~05:25.

### 2026-09-22 05:29 — v2-224 DONE. All four runs complete.

**v2-224 — p4_scratch, 189,712 params, res 224, v2 data (600k), 6 epochs, RTX 5060, 1h44m**

oid 0.9326 | library(3500) 0.9651 | library(full 13352) single 0.9637, tta 0.9637

| thr | P | R |
|-----|---|---|
| 0.5 | 0.647 | 0.942 |
| 0.8 | 0.863 | 0.889 |
| 0.9 | 0.923 | 0.828 |
| 0.95 | 0.957 | 0.775 |

(Note epoch 5 came in *below* epoch 4 on the subset, 0.9651 vs 0.9669 — within the noise band
established at 01:40, but a reminder that 6 epochs is where this model stops improving cleanly.)

---

## Summary: the overnight matched-data experiment

Four runs, all 6 epochs, all evaluated on the same 13,352-image library in stored orientation. The
three `v2-*` runs share identical 600k codec-matched training data, so they differ in exactly one
variable each. Cost is measured CPU latency at 4 threads, single pass (the nets are equivariant, so
no TTA is needed).

| run | params | res | train data | R@P≈0.92 | R@P≈0.95 | lib(full) | CPU ms | 
|---|---|---|---|---|---|---|---|
| `v2-224` (baseline) | 189,712 | 224 | 600k v2 | 0.828 | 0.775 | 0.9637 | **22.5** |
| `v2-384` (+resolution) | 189,712 | 384 | 600k v2 | 0.875 | 0.830 | 0.9655 | 59.3 |
| `v2-big` (+capacity) | 757,280 | 224 | 600k v2 | 0.876 | 0.836 | 0.9718 | 43.2 |
| `big-full` (+capacity +data) | 757,280 | 224 | 1.09M v1 | **0.905** | **0.870** | 0.9715 | 43.2 |
| check_orientation (2020) | 25.0M | 224 | — | — | — | — | 313.6 |

**Both levers are worth the same accuracy, but capacity is much cheaper.**

| lever | gain in R@P≈0.92 | cost |
|---|---|---|
| resolution 224 -> 384 | +4.7 pts | +164% latency (22.5 -> 59.3 ms) |
| capacity 190k -> 757k | +4.8 pts | +92% latency (22.5 -> 43.2 ms) |

Identical payoff, roughly half the cost. Per 10 ms spent, capacity buys 2.3 points of recall and
resolution buys 1.3 — capacity is ~1.8x the better lever. This is the opposite of what the earlier
v1 resolution result suggested, and it is only visible because the runs finally share training data.

**Data is the only lever that is free at inference.** `big-full` and `v2-big` are the same
architecture at the same cost (43.2 ms); the extra 1.8x of training data is worth another **+2.9
points of recall** for zero runtime. That makes finishing the v2 derivation of the remaining ~490k
originals the highest-value next step by a distance — it is the one axis where the accuracy is not
paid for in user CPU time.

**Caveat on the data comparison, unchanged from 03:34:** `big-full` differs from `v2-big` in both
data volume and training codec, so "+2.9 points" is an upper bound on the data effect alone. The
codec-matched 600k set did *not* beat the unmatched 1.09M one, so "train on what you serve" did not
pay for a 45% smaller corpus.

**Against check_orientation.** Our best configuration is 7.3x cheaper (43.2 ms vs 313.6 ms) and 33x
smaller (0.76M vs 25.0M params), at R=0.905/P=0.914. The 2020 model's headline P=0.94/R=0.94 was
measured in an earlier pass with different ground truth and a different aggregation; **the two are
not currently measured under the same protocol and should not be put in one table until they are.**
Re-scoring check_orientation against the present ground truth on `testset/prev_raw` is the right way
to settle it, and is cheap.

### What is settled vs still open

Settled: resolution and capacity buy the same accuracy and capacity is ~2x cheaper; MACs mispredict
CPU latency badly (p4_big@224 has 32% more MACs than p4_scratch@384 yet runs 27% faster); equivariance
holds end to end and makes TTA free; the 600k codec-matched set does not beat the 1.09M unmatched one.

Open: the data lever's true size (needs the remaining 490k derived); whether capacity and resolution
stack (p4_big@384 is 155.7 ms, likely too expensive); a same-protocol comparison against
check_orientation; whether more than 6 epochs helps p4_big, whose loss was still falling at the end.

## 2026-09-22 06:05 — check_orientation re-scored under the SAME protocol

Ran `scratchpad/co_rescore.py` as k8s job `co-rescore` on the RTX 3060: 13,352 images x 4 passes in
123 s. Protocol identical to the v2 runs in every respect that could bias it:

- same images (`testset/prev_raw`, stored orientation), same 693 rotated
- same ground truth (`testset/ground_truth.json`)
- same geometry: squash-to-square -> 224 BICUBIC, **then** `rot90` (matching `Library`, not
  rotate-then-resize; the two commute for squash-to-square)
- same ImageNet normalisation, same TTA aggregation `agg[c] += p_j[(c+j)%4]`, same thresholds
- verified check_orientation emits probabilities (rowsum == 1.0), so no double softmax

Real model figures, re-measured: **22,988,100 params, 16.914 GMACs/img, 323.7 ms/img** CPU at 4
threads. (The 25.0M / 313.6 ms in the 04:20 table was torchvision's 1000-class ResNeXt50 used as a
proxy; close, but these are the actual numbers.)

**The 2020 headline reproduces.** thr 0.8 with 4-rot TTA gives **P=0.9406 R=0.9365**, essentially the
P=0.94/R=0.94 recorded back on 250px webp thumbnails with the older ground truth. It is robust to the
input change, which also means the codec confound that bit our models did not bite this one.

### Full curves, matched protocol

| thr | v2-224 | v2-384 | v2-big | big-full | check_orientation TTA |
|---|---|---|---|---|---|
| 0.5 | .647/.942 | .663/.960 | .698/.958 | .700/.971 | **.800/.976** |
| 0.6 | .719/.929 | .736/.955 | .751/.948 | .762/.961 | **.857/.968** |
| 0.7 | .784/.913 | .804/.938 | .812/.932 | .811/.952 | **.892/.952** |
| 0.8 | .863/.889 | .864/.922 | .864/.918 | .863/.935 | **.941/.936** |
| 0.9 | .923/.828 | .928/.875 | .911/.876 | .914/.905 | **.962/.798** |
| 0.95 | .957/.775 | .952/.830 | .952/.836 | .944/.870 | .973/.465 |
| 0.99 | .987/.661 | .977/.724 | .981/.758 | **.982/.795** | .969/.091 |

Recall at matched precision — the honest comparison:

| R at P>= | v2-224 | v2-384 | v2-big | big-full | check_orientation |
|---|---|---|---|---|---|
| 0.85 | 0.889 | 0.922 | 0.918 | 0.935 | **0.968** |
| 0.90 | 0.828 | 0.875 | 0.876 | 0.905 | **0.936** |
| 0.93 | 0.775 | 0.830 | 0.836 | 0.870 | **0.936** |
| 0.95 | 0.775 | 0.830 | 0.836 | 0.795 | **0.798** |
| 0.96 | 0.661 | 0.724 | 0.758 | 0.795 | **0.798** |

Library 4-way accuracy: check_orientation single 0.9781 / **tta 0.9825**, versus our best 0.9718
(`v2-big`) and 0.9715 (`big-full`).

### This overturns the optimistic framing

**check_orientation is still clearly the better model.** It wins at every precision target we care
about, and at the auto-apply operating point (P~0.93-0.95) the gap is large: **R=0.936 at P=0.941 vs
big-full's R=0.870 at P=0.944 — 6.6 points of recall**, or 10 points against `v2-big`. The earlier
"AP tied, we win above P=0.95" reading does not survive a same-protocol measurement; the only place
our models are ahead is a narrow band right around P=0.95, and even there it is 0.836 vs 0.798 for
`v2-big` and a tie for `big-full`.

One real advantage remains, and it is a calibration difference rather than an accuracy one:
**check_orientation's confidence collapses at high thresholds** (R 0.936 -> 0.798 -> 0.465 -> 0.091
across thr 0.8/0.9/0.95/0.99) while ours degrade gracefully (`big-full` 0.935 -> 0.905 -> 0.870 ->
0.795). If you want a very-high-precision tier that still flags a useful number of images, ours is
better there — at thr 0.99 `big-full` holds P=0.982 R=0.795 where check_orientation manages R=0.091.

Also worth noting our nets need **one** pass to check_orientation's **four**: at single pass the 2020
model drops to P=0.8983 R=0.9307 at thr 0.8 and 0.9781 4-way, so a meaningful part of its lead is
bought with 4x the compute it already needs.

### What this means for shipping

The comparison is not "should Immich ship check_orientation" — **it is CC-BY-NC and cannot be
redistributed commercially**, which is why this whole training exercise exists. The question is how
much accuracy a shippable model gives up, and the answer is now measured rather than guessed:

| | params | CPU ms/img | R at P>=0.93 | licence |
|---|---|---|---|---|
| check_orientation | 22.99M | 323.7 | 0.936 | CC-BY-NC — **unshippable** |
| big-full (p4_big, 1.09M imgs) | 0.76M | 43.2 | 0.870 | ours, Apache-compatible data |
| v2-big (p4_big, 600k imgs) | 0.76M | 43.2 | 0.836 | ours |
| v2-224 (p4_scratch) | 0.19M | 22.5 | 0.775 | ours |

7.5x cheaper for 6.6 points of recall at P=0.93. Whether that trade is acceptable is a product call,
but the gap is real and I had been under-reporting it.

## 2026-09-22 11:40 — prior correction, and why a prior-matched retrain is pointless here

### Correcting the comparison method first

The 8-threshold grid in the earlier entries **under-samples the PR curve badly**. Dumped per-image
posteriors for all four models plus check_orientation (`scratchpad/dump_probs.py`, `probs/*.npz`,
argmax accuracies reproduce the training-time `lib(full)` exactly) and computed full curves:

| model | R@P>=.85 | R@P>=.90 | R@P>=.93 | R@P>=.95 | R@P>=.97 | max-R | AP |
|---|---|---|---|---|---|---|---|
| v2-224 | .893 | .859 | .811 | .784 | .747 | .958 | .9248 |
| v2-384 | .926 | .892 | .874 | .835 | .755 | .964 | .9377 |
| v2-big | .921 | .879 | .863 | .846 | .784 | .968 | .9444 |
| **big-full** | .944 | .926 | .906 | .880 | **.831** | .973 | **.9583** |
| check_orientation TTA | **.970** | **.948** | **.945** | **.906** | .478 | **.977** | .9457 |
| check_orientation 1-pass | .961 | .928 | .853 | .110 | .091 | .971 | .9107 |

On the full curve `big-full` gets R=0.906 at P>=0.93, not the 0.870 the grid reported, and **AP
0.9583 beats check_orientation's 0.9457**. The curves cross near P~0.96. The 06:05 conclusion
("check_orientation is still clearly the better model") was an artifact of the coarse grid.

### Where the errors actually are

At argmax, `big-full` misses only **8** genuinely rotated images. 361 of its 372 false positives are
upright images called rotated, and **101 of those are called 180deg — a class with 18 instances in the
entire 13,352-image library**. Prevalence: upright 12,659 / 90deg 431 / 180deg 18 / 270deg 244.
Training samples rotation uniformly, so 180deg is over-represented ~190x.

### Prior correction: helps, but only where it matters

Post-hoc reweighting `p'(c) ∝ p(c)·π(c)`:

| prior | R@P>=.93 | R@P>=.97 | R@P>=.98 | R@P>=.99 |
|---|---|---|---|---|
| uniform | .906 | .831 | .817 | .766 |
| [1,1,0.1,1] | **.916** | .854 | .835 | .802 |
| [5,1,0.1,1] | .913 | **.863** | .837 | .804 |
| [20,1,0.1,1] | .906 | .859 | **.840** | **.815** |

Down-weighting 180deg does nearly all the work, matching the FP analysis. It generalises across all
four models (v2-224 +5.6 pts at P>=0.93, v2-384 +4.7 at P>=0.97, v2-big +2.9/+4.0). Note **AP goes
down** while high-precision recall goes up — it is a trade along the curve, not a free win, and I
initially called it a dead end by only looking at P=0.85-0.95.

### A prior-matched RETRAIN cannot work — verified

`P4Net.head = nn.Linear(widths[-1], 1, bias=False)`: one shared weight vector applied to each of the
four group slices. **There is no per-class parameter anywhere in the network**, so the four logits are
forced to be cyclic shifts and the architecture cannot represent a class prior.

Stronger, and measured on the trained `p4_big` weights: cross-entropy for `(x, label 0)` versus
`(rot90(x,j), label j)` is **identical to six decimals** (1.2965983 / 1.2965962 / 1.2965995 /
1.2965981; max logit-shift error 9.5e-06). So `k = random.randrange(4)` in the training loop is an
**exact no-op** — same loss, same gradient. Changing the rotation sampling distribution has provably
zero effect on what this model learns.

Consequence: post-hoc correction is not a cheap stand-in for a proper retrain, it is the *only*
mechanism available. The sole alternative is adding a learnable per-class bias, which is the same
object as the post-hoc log-prior and would forfeit exact equivariance (and therefore free TTA).
Also: rotation augmentation is redundant for these nets and could be dropped.

### Re-framed for the actual product design

Not autonomous rotation. A utility screen with two bands: **A** = pre-rotated suggestion, one-press
approve; **B** = surfaced but not pre-rotated. Band A set to precision >= 0.95:

| model | A shown | A one-press correct | A wrong | B shown | B rotated | missed | total shown |
|---|---|---|---|---|---|---|---|
| big-full (uniform) | 642 | 610 | 32 | 404 | 64 | **19** | 1046 |
| big-full (+[5,1,0.1,1]) | 649 | **617** | 32 | 146 | 40 | 36 | 795 |
| v2-big (+prior) | 622 | 591 | 31 | 161 | 61 | 41 | 783 |
| check_orientation | 661 | **628** | 33 | 245 | 49 | **16** | 906 |

Against check_orientation the gap is **18 of 693 images** landing in manual instead of one-press, plus
3 more missed. 2.6%, for a model 30x smaller and 7.5x cheaper that can actually be shipped.

**Design recommendation: use a different score per band.** Prior-corrected ranking for band A
(617 one-press vs 610), plain uniform argmax for band-B membership (19 missed vs 36). The prior helps
precision at the top and hurts recall at the bottom, so applying it only where precision matters gets
both. Band A's 32 wrong pre-rotations are the cost of P=0.95; raising band A to P>=0.98 would roughly
halve that at some cost in band-A size.

### 2026-09-22 11:52 — closing the data lever properly (job `oid-full`)

Extending the v2 set from 600k to **1,094,589** — exactly v1's count. `pool.jsonl` (1,495,772
candidates) is consumed in order and `shard.py` skips what is already present, so this continues into
the same superset v1 used. That makes two comparisons clean at once:

- `v2-big` (600k v2) vs the new run (1.09M v2) — **data volume**, codec held fixed
- `big-full` (1.09M v1 webp) vs the new run (1.09M v2) — **training codec**, volume held fixed

which removes the confound caveated at 03:34 and 05:29.

Note the originals for the extra 494,589 have to be **re-downloaded**, not just re-derived: the first
pass discarded originals, which is why only 600k are on disk. Disk check before launching: 231 G free,
need ~149 G originals + ~27 G derived = ~176 G. Fits with ~55 G spare.

Job `oid-full`: s5cmd fetch (13 shards of 40k) into `oid/orig`, then `derive_all.sh` with SLICES=10
into `oid/img_v2`, then `manifest.py`. Fresh `SHARDDIR=oid/shards_full` so the old `.done`
markers do not collide. ETA ~10 min fetch + ~23 min derive.

Then: p4_big, res 224, bs 64, 6 epochs on the 1.09M v2 set, on the RTX 5060 (~4.5 h).

Fetch is running ~700 img/s per 40k shard.

### 2026-09-22 12:17 — full v2 set built, `v2-bigfull` launched

`oid-full` completed in 35 min (fetch ~700 img/s per 40k shard, then 10-way derive). Verified:

- `oid/orig` — **1,094,589** originals
- `oid/img_v2` — **1,094,589** derived, zero-byte scan clean (0 bad, vs 1 last time)
- `oid/manifest_v2.jsonl` — **1,094,589** rows

All three match v1's 1,094,589 exactly. Disk after: 437G used of 1008G, 571G free.

`v2-bigfull` launched on the RTX 5060: p4_big, res 224, bs 64, 6 epochs, RUN_ID `v2-bigfull`.
This is the run that separates the two effects that have been tangled since 03:34:

- vs `v2-big` (600k v2, same codec) -> **data volume**
- vs `big-full` (1.09M v1 webp, same volume) -> **training codec**

## 2026-09-22 12:5x — what is left besides scaling (free analysis from dumped posteriors)

### Confidence score: `1 - p(upright)` beats `max p`
The flag decision is "is this rotated", so the natural score is the detection probability, not the
max posterior. AP big-full 0.9583 -> 0.9623, v2-big 0.9444 -> **0.9535**; R@P>=0.97 v2-big
0.784 -> 0.817. Better at the high-precision end and on AP, marginally worse mid-curve. Free.

### Self-ensembling: dead end
Our four models share about half their errors (Jaccard 0.44-0.49 pairwise). No combination beats
`big-full` alone: `v2-big+big-full` AP 0.9595 (vs 0.9583) at double the cost, and the 4-way ensemble
is *worse* (0.9556). Same family, overlapping data, correlated mistakes.

### Cross-family ensembling: where the headroom actually is
Error overlap with big-full: v2-big 0.48, **mnv4_consistency 0.37**, check_orientation 0.32. Lower
overlap across families, and it converts into real accuracy:

| model / ensemble | GMACs | R@P>=.90 | R@P>=.93 | R@P>=.97 | AP |
|---|---|---|---|---|---|
| big-full | 2.98 | .926 | .906 | .831 | .9583 |
| mnv4_consistency (Apache-2.0) | 0.74 | .900 | .882 | .848 | .9453 |
| check_orientation | 16.91 | .948 | **.945** | .478 | .9457 |
| **big-full + mnv4_consistency** | **3.72** | .944 | .932 | **.866** | **.9681** |

Two-band product view (band A at P>=0.95):

| model | GMACs | one-press | wrong | manual | missed | shown |
|---|---|---|---|---|---|---|
| big-full alone | 2.98 | 610 | 32 | 404 | 19 | 1046 |
| big-full + mnv4 | 3.72 | **628** | 33 | 320 | **14** | 981 |
| check_orientation | 16.91 | 628 | 33 | 245 | 16 | 906 |

**The shippable ensemble reaches exact parity with check_orientation on one-press (628) and beats it
on missed (14 vs 16), at 4.5x less compute.** But against `big-full` alone it is only +18 one-press
and -5 missed for +25% compute and a second model to ship, version and maintain — **not worth
shipping two models on those grounds**. The value is as evidence that the remaining headroom is
cross-family diversity, not scale. `v2-bigfull` may absorb much of it into the single model.

### EXIF gating: my hypothesis was wrong, and backwards
I claimed "a valid Orientation tag ought to be close to a guarantee of correctness". Measured on a
700-asset sample enriched 300 rotated / 400 upright, corrected back to true prevalence:

| orientation tag | est. rotated % |
|---|---|
| `1` (declared upright) | 5.36% |
| `6` | 4.74% |
| `8` | 6.80% |
| **none** | **0.78%** |
| **`3`** (declared 180) | **100%** (10/10 sampled) |

Library baseline 5.19%. A valid tag is worth **nothing** — `orientation=1` is exactly baseline. And
**missing tags are 6.6x safer**, the opposite of my reasoning: gating on "has a tag" would have
discarded the safest group and kept the risky one. In hindsight obvious — the failure mode *is* a
camera writing `orientation=1` onto rotated pixels, so misrotated images are dominated by tag=1.

The one real signal is `orientation=3`: every sampled asset declaring a 180 flip was misrotated, and
180 is exactly the class the model is worst at. But it is ~23 assets in 13,352, so the ceiling is
tiny. Filename class is also weak (camera/phone 5.9% rotated vs other 3.0%; FP rates 3.1% vs 2.2%).

The user's methodological point stands and is the right one: a signal like this should have its
weight fitted from data, not asserted by me beforehand. Note a pixel model cannot learn EXIF at all
unless it is fed in, so "let the model decide" would mean adding it as an input — and on this evidence
there is almost nothing to learn.

### Still untested
- **Aspect ratio.** Squash-to-square destroys it, yet a 90/270 rotation is exactly what swaps it, and
  Immich stores width/height. `log(w/h)` negates under 90 deg rotation so it can be fed without
  breaking C4 equivariance. Free signal currently discarded.
- **Longer training.** big-full's loss was still falling at epoch 5.

### 2026-09-22 22:28 — derive384 finished damaged, as expected; repair running

The ENOSPC damage is confirmed by the slice totals: `SLICE n DONE ok=86404 / 86436 / 86479 ...` —
about **864k successes against 1,094,589 expected**, so roughly 230k writes were silently swallowed by
`except Exception: pass` while the volume was full.

Also note the job's closing `DERIVE_ALL_DONE / 1094589` is **misleading**: `derive_all.sh` ends with a
hardcoded `ls oid/img_v2 | wc -l`, which counts the *old 256 px* directory, not the 384 one it
just wrote. It would have reported success on a set that is a quarter missing.

`repair384` launched: up to 3 passes of (marker-scan `ffd8`/`ffd9` + size>1KB, delete bad, re-derive
the gaps with 10 parallel slices), stopping as soon as the count reaches 1,094,589.

Progress on the two training runs, both healthy:

```
v2-bigfull  ep 4 loss=0.1346 | oid 0.9522 | library 0.9743 | 3418s   (98% / 107W)
mnv4-v2     ep 3 loss=0.1892 | oid 0.9456/0.9532 | library 0.9679/0.9720 | 1503s   (97% / 159W)
```

`v2-bigfull` at epoch 4 is **0.9743 vs big-full's 0.9734** at the same epoch — the first point where
the codec-matched 1.09M set is ahead of the webp 1.09M set rather than behind. One epoch to go.

## 2026-09-23 23:25 — both confounds RESOLVED

`v2-bigfull` (p4_big, 1.09M **codec-matched**) and `mnv4-v2` (mnv4_consistency, same data) both
finished. Full PR curves from dumped posteriors — never the 8-threshold grid.

| model | R@P>=.90 | R@P>=.93 | R@P>=.95 | R@P>=.97 | max-R | AP |
|---|---|---|---|---|---|---|
| v2-224 (600k v2, 190k) | .859 | .811 | .784 | .747 | .958 | .9248 |
| v2-384 (600k v2, **upsampled**) | .892 | .874 | .835 | .755 | .964 | .9377 |
| v2-big (600k v2, 757k) | .879 | .863 | .846 | .784 | .968 | .9444 |
| big-full (1.09M **v1 webp**) | .926 | .906 | .880 | .831 | .973 | .9583 |
| **v2-bigfull (1.09M v2)** | .925 | .898 | .889 | **.860** | .974 | **.9593** |
| mnv4 (v1 webp) | .900 | .882 | .867 | .848 | .962 | .9453 |
| **mnv4-v2 (1.09M v2)** | .919 | .905 | .893 | **.889** | .961 | .9498 |
| check_orientation | **.948** | **.945** | **.906** | .478 | **.977** | .9457 |

### 1. DATA VOLUME — real, and free at inference
Codec held fixed, both v2: 600k -> 1.09M takes AP **.9444 -> .9593** and R@P>=0.93 **.863 -> .898**.
+3.5 recall points for 1.8x data, at zero runtime cost. This is the lever.

### 2. TRAINING CODEC — a wash on average, small gain at high precision
Volume held fixed, both 1.09M: big-full (webp) AP **.9583** vs v2-bigfull (matched) AP **.9593**.
Tied. Not uniform though: webp is slightly ahead at P>=0.93 (.906 vs .898), matched is ahead at
P>=0.95 (.889 vs .880) and clearly at P>=0.97 (**.860 vs .831**).

So the 03:34/05:29 caveat resolves as: **data was doing the work, the codec match is worth ~nothing on
average and a few points where precision matters.** My 05:29 reading ("train on what you serve did not
pay") is confirmed at matched volume rather than merely unrefuted.

**But it clearly helped the other architecture**: mnv4 AP .9453 -> .9498, R@P>=0.97 **.848 -> .889**.
Plausible mechanism: MobileNetV4 is ImageNet-pretrained on JPEG, so webp training data was doubly
mismatched for it, while our p4 net trains from scratch and simply learns whatever it is shown.

### 3. Ensembles

| ensemble | R@P>=.93 | R@P>=.97 | AP |
|---|---|---|---|
| v2-bigfull + mnv4-v2 | .929 | **.895** | **.9665** |
| big-full + mnv4 (v1 pair) | .932 | .866 | .9681 |
| v2-bigfull + mnv4-v2 + big-full | **.944** | .899 | **.9719** |
| check_orientation | .945 | .478 | .9457 |

### Two-band product view (band A pre-rotated at P>=0.95)

| model | GMACs | one-press | wrong | manual | missed | shown |
|---|---|---|---|---|---|---|
| big-full alone | 2.98 | 610 | 32 | 404 | 19 | 1046 |
| v2-bigfull alone | 2.98 | 616 | 32 | 369 | 18 | 1017 |
| **mnv4-v2 alone** | **0.74** | **619** | 32 | 348 | 27 | 999 |
| **v2-bigfull + mnv4-v2** | 3.72 | **635** | 33 | 284 | **17** | 952 |
| check_orientation | 16.91 | 628 | 33 | 245 | 16 | 906 |

**The shippable pair now beats check_orientation outright** — 635 one-press vs 628, 17 missed vs 16,
at 4.5x less compute and 30x fewer parameters, and roughly double its recall at P>=0.97 (.895 vs .478).

**`mnv4-v2` alone gets 619 one-press — but it is NOT cheaper.** Measured CPU latency (4 threads):

| model | params | GMACs/img | **CPU ms/img** | one-press | missed |
|---|---|---|---|---|---|
| v2-bigfull (p4_big) | 757k | 2.977 | **43.2** | 616 | 18 |
| mnv4-v2 | 2.50M | 0.739 | **62.0** | 619 | 27 |
| both ensembled | 3.26M | 3.716 | **105.2** | 635 | 17 |
| check_orientation | 22.99M | 16.914 | **323.7** | 628 | 16 |

mnv4 has **4x fewer MACs yet runs 44% slower**, because it is not equivariant and therefore needs
**four** forward passes where p4_big needs one — four lots of kernel-launch overhead at batch 1, on
depthwise-separable convs that vectorise poorly. **This is the third time in this project that MACs
mispredicted CPU latency, always in the same direction** (earlier: p4_big@224 has 32% more MACs than
p4_scratch@384 yet runs 27% faster). Treat GMACs as unusable for ranking these models; measure.

So **p4_big is the better single model** (43.2 ms, 616 one-press, 18 missed) and the ensemble costs
105.2 ms for 635/17 — still **3.1x faster than check_orientation** while beating it on both counts.

### 2026-09-23 00:45 — img_384 verified, and the experiment matrix finally has a shared baseline

`gapfill384` reached **1,094,589 / 1,094,589, missing 0**, and a full marker scan of all of them
returned **0 bad** — the ENOSPC damage is fully repaired. Spot-checked dimensions from inside the
training pod: `(574,384) (663,384) (580,384) (643,384) (512,384)` — genuinely 384 on the short side,
not upsampled from 256.

**Gap in the matrix I had not noticed:** `v2-384real` (p4_scratch @384, 1.09M) had no valid peer.
`v2-224` is p4_scratch @224 but only 600k, and `v2-bigfull` is 1.09M but p4_big — so neither isolates
resolution. Added `v2-224full` (p4_scratch @224 on the full 1.09M) to serve as the shared baseline.
Now every axis is one variable away from the same reference:

| run | arch | res | epochs | isolates |
|---|---|---|---|---|
| `v2-224full` | p4_scratch 190k | 224 | 6 | **baseline** (RX 9060 XT) |
| `v2-384real` | p4_scratch 190k | 384 | 6 | **resolution**, real detail (RTX 3060) |
| `v2-bigfull` | p4_big 757k | 224 | 6 | **capacity** — done, AP .9593 |
| `v2-big12` | p4_big 757k | 224 | 12 | **epoch count** (RTX 5060) |

All on the same 1,094,589 codec-matched images. The earlier resolution/capacity verdict compared
600k arms against each other and then against 1.09M runs, which is what made it shaky.

**First real training run on AMD.** `v2-224full` is on an RX 9060 XT with torch 2.9.1+rocm6.4,
numpy 2.5.3 and pillow 12.3.0. `train.py` needs no changes at all. 11,349 steps/epoch.

### 2026-09-23 01:27 — v2-384real is GPU-starved, and a batch-size confound to declare

```
v2-384real   98% util,  51.7 W   (RTX 3060, 170 W card)
v2-big12     99% util, 108.7 W   (RTX 5060)
```

51.7 W at "98% utilization" is the launch-bound signature again: p4_scratch at 384 has 8-64 channels
over a large spatial grid at batch 32, so the card spends its time on kernel launches and memory
traffic rather than arithmetic. Utilization only means a kernel was resident. Same effect that made
p4_scratch@384 slower per MAC than p4_big@224 on CPU.

**Confound to declare up front:** `v2-224full` runs at bs 96, `v2-384real` at bs 32 — 384 will not fit
96 in memory. So the resolution arm is not a pure resolution isolation; it also changes batch size and
therefore the effective LR schedule. It is at least *consistent* with the earlier pairing (the
original v2-224/v2-384 used the same 96/32 split), so the measured delta remains comparable to the
one it is replacing, but it should not be reported as "resolution alone".

Neither small run had printed epoch 0 at 45-56 min, so both are slower than estimated; getting exact
epoch times before deciding whether a larger batch is worth a restart.

### 2026-09-23 04:09 — AMD cards made genuinely usable; 2 more bad files caught

`derive384rbd` finished 1,094,589 on RBD. The marker scan then found **2 bad files** and deleted them
— worth noting because `train.py` builds its worklist from the manifest and opens each file, so two
missing images would have crashed the dataloader hours in, exactly as a single bad file killed
`v2-224` earlier. Re-derived both (`051e644650fdc040`, `052f1daee7e483f3`) and synced them; `img_384`
is now 1,094,589 on both volumes, matching the manifest.

`v2-384real` now runs on the RX 9060 XT, keeping the RTX 3060 free for the seed-repeat variance
check when `v2-224full` finishes (~06:00). 1,089,589 images at res 384, 34,049 steps/epoch.

Three cards productive across two nodes: RTX 5060 (`v2-big12`), RTX 3060 (`v2-224full`), RX 9060 XT
(`v2-384real`). Worth noting both faults tonight that could have wasted hours — the 2 corrupt 384
files and this missing ground truth — were caught by cheap checks that ran in under two minutes each.

## 2026-09-23 06:31 — the clean lever matrix (baseline finally exists)

`v2-224full` done: oid .9428, library(full) .9670. With it, every cell is one variable from a shared
reference on the same 1,094,589 codec-matched images.

| run | arch | res | data | R@P>=.90 | R@P>=.93 | R@P>=.95 | R@P>=.97 | AP |
|---|---|---|---|---|---|---|---|---|
| v2-224 | 190k | 224 | 600k | .859 | .811 | .784 | .747 | .9248 |
| **v2-224full** | 190k | 224 | **1.09M** | .872 | .846 | .828 | .771 | **.9329** |
| v2-384 (upsampled) | 190k | 384 | 600k | .892 | .874 | .835 | .755 | .9377 |
| v2-big | 757k | 224 | 600k | .879 | .863 | .846 | .784 | .9444 |
| **v2-bigfull** | 757k | 224 | **1.09M** | .925 | .898 | .889 | .860 | **.9593** |

```
DATA VOLUME at p4_scratch 224 (600k -> 1.09M):  AP .9248 -> .9329   +0.0081
DATA VOLUME at p4_big     224 (600k -> 1.09M):  AP .9444 -> .9593   +0.0149
CAPACITY    at 1.09M      (190k -> 757k):       AP .9329 -> .9593   +0.0265
```

**Capacity is the bigger lever, ~2-3x the data effect.** And data pays off *more* at higher capacity
(+.0149 vs +.0081), so the two compound rather than substitute — the small model cannot exploit the
extra data as well as the large one can.

Caveat that matters: **the noise floor is still unmeasured.** +0.0265 will survive almost any
plausible variance, but +0.0081 may not. `v2-bigfull-s2` (identical config, different seed) is now
running on the RTX 3060 to settle it; ETA ~13:45.

`v2-384real` has still not posted epoch 0 after ~2.5 h on the RX 9060 XT, implying >2.5 h/epoch and a
~19:00 finish. Leaving it: the 5060 does not free until ~11:00 and would finish later still (p4_scratch
at 384 is launch-bound, so a faster card helps less than the step count hurts).

### 2026-09-23 07:12 — first real-384 epoch: resolution looks real after all

```
v2-384real  ep 0 loss=0.4790 | oid 0.8944 | library 0.9377 | 8501s  (2.36 h/epoch, RX 9060 XT)
v2-big12    ep 7 loss=0.1221 | oid 0.9506 | library 0.9711
```

`v2-384real` epoch 0 is **0.9377 vs v2-224full's epoch-0 0.9251** — 1.3 points ahead with identical
architecture, data and epoch index, differing only in resolution (and batch size, the declared
confound). First evidence that the resolution arm was handicapped by upsampled 256 px inputs rather
than resolution being a weak lever.

At 2.36 h/epoch the 6 epochs finish ~18:20. Not cutting it short: a truncated run would not be
comparable to the 6-epoch baseline, which is the only reason to run it.

`v2-big12` at ep7 (.9711) sits below `v2-bigfull`'s ep5 (.9754), which is expected rather than
worrying — OneCycle spread over 12 epochs means it is mid-schedule and not yet annealed. Only its
ep11 is comparable.

---

# MORNING SUMMARY — 2026-09-23 09:05

## Settled overnight

**1. The two confounds from yesterday are resolved.** Running p4_big on 1.09M *codec-matched* images
(`v2-bigfull`) separated data volume from training codec for the first time:

- **Data volume is a real lever.** 600k -> 1.09M: AP .9444 -> .9593 at p4_big, .9248 -> .9329 at
  p4_scratch. Free at inference.
- **Codec matching is a wash for the equivariant net** (AP .9583 webp vs .9593 matched, tied) but
  worth +2.9 recall points at P>=.97, and it clearly helps MobileNetV4 (+4.1 pts at P>=.97) —
  plausibly because mnv4 is ImageNet-pretrained on JPEG, so webp was doubly mismatched for it.

**2. The full lever matrix now has a shared baseline** (all 1,094,589 codec-matched images):

| run | arch | res | data | AP |
|---|---|---|---|---|
| v2-224 | 190k | 224 | 600k | .9248 |
| v2-224full | 190k | 224 | 1.09M | .9329 |
| v2-384 (upsampled) | 190k | 384 | 600k | .9377 |
| v2-big | 757k | 224 | 600k | .9444 |
| v2-bigfull | 757k | 224 | 1.09M | **.9593** |

DATA +.0081 (small model) / +.0149 (large); **CAPACITY +.0265**. Capacity is the bigger lever, and
data compounds with it — the small model cannot exploit extra data as well as the large one.

**3. A shippable pair now beats the unshippable reference.** `v2-bigfull` + `mnv4-v2` ensembled:
AP .9665, **635 one-press / 17 missed / 952 shown at 105.2 ms CPU**, versus check_orientation's
628 / 16 / 906 at **323.7 ms** and AP .9457. 3.1x cheaper, better on one-press, level on missed.
Single best model is `v2-bigfull` alone: 616 / 18 at 43.2 ms.

**4. Findings that will outlive this project.** A million small files need local/block storage, not
a network filesystem — the difference measured ~73x on cold reads. The RX 9060 XT benchmarks at ~84%
of an RTX 3060 under ROCm. And MACs mispredicted CPU latency for the third time, always the same way.

## Noise floor — still pending, and it matters

`v2-bigfull-s2` (identical config, different seed) is the run that decides whether the smaller gaps
above are findings or noise. Epoch-wise so far against `v2-bigfull`:

```
ep0   .9383 vs .9451   (0.68 pts apart)
ep1   .9626 vs .9629   (0.03 pts apart)
```

Encouraging — the spread collapses as training anneals — but the final AP is the number that counts.
**+.0265 (capacity) survives almost any plausible variance; +.0081 (data at p4_scratch) may not.**
Treat that one as provisional until this lands.

## Open

- Whether resolution and capacity **stack** (p4_big at real 384) — untested, and expensive.
- **Aspect ratio** as an input. Still the most promising untried idea: squash-to-square destroys it,
  yet a 90/270 rotation is exactly what swaps it, and `log(w/h)` negates under 90 deg so it can be fed
  without breaking C4 equivariance.
- A same-protocol **re-score of check_orientation** exists, but no ensemble of it with the newer runs
  has been explored beyond the pairs already measured.

## 2026-09-23 11:22 — EPOCH COUNT is a major lever, and a single model now beats check_orientation

`v2-big12` (p4_big, res 224, bs 64, **12 epochs**, 1.09M codec-matched) finished: oid .9560,
library(full) .9744.

| model | R@P>=.90 | R@P>=.93 | R@P>=.95 | R@P>=.97 | max-R | AP |
|---|---|---|---|---|---|---|
| v2-bigfull (6 ep) | .925 | .898 | .889 | .860 | .974 | .9593 |
| **v2-big12 (12 ep)** | .931 | **.922** | **.911** | **.890** | **.983** | **.9699** |
| mnv4-v2 | .919 | .905 | .893 | .889 | .961 | .9498 |
| check_orientation | **.948** | **.945** | .906 | .478 | .977 | .9457 |

**Doubling epochs is worth +0.0106 AP.** For scale: data volume at p4_big was +0.0149, capacity was
+0.0265. So epochs sit between the two — and unlike either, **it costs nothing at inference**, only
training time. Also note max-R rises to .983, the highest of any single model including
check_orientation's .977.

### Two-band product view (band A pre-rotated at P>=0.95)

| model | one-press | wrong | missed | shown | CPU ms |
|---|---|---|---|---|---|
| v2-bigfull (6 ep) | 616 | 32 | 18 | 1017 | 43.2 |
| **v2-big12 (12 ep)** | **631** | 33 | **12** | 1015 | **43.2** |
| v2-big12 + mnv4-v2 | 636 | 33 | 14 | 948 | 105.2 |
| check_orientation | 628 | 33 | 16 | 906 | 323.7 |

**A single shippable model now beats check_orientation on both counts** — 631 one-press vs 628, and
12 missed vs 16 — at **7.5x less compute and 30x fewer parameters**. Until now that required the
two-model ensemble. This is the cleanest shipping story the project has produced.

Ensembles still add on AP: `v2-big12+mnv4-v2` .9712 (636/14), three-way with v2-bigfull .9740
(the best number measured), but a single model at 43.2 ms is far more attractive to ship than two or
three.

**Caveat:** the noise floor is still unmeasured (`v2-bigfull-s2`, ETA ~14:00). +0.0106 is well clear
of the +/-0.5 pt per-epoch wobble seen so far, but AP variance specifically has not been quantified.

## 2026-09-23 14:36 — NOISE FLOOR MEASURED: every lever survives

`v2-bigfull-s2` (identical config to `v2-bigfull`, different seed) finished: oid .9510,
library(full) .9730 vs .9736.

```
v2-bigfull     AP 0.9593
v2-bigfull-s2  AP 0.9608
|delta|           0.0015
```

| lever | gain | multiple of noise | verdict |
|---|---|---|---|
| capacity (190k -> 757k) | +0.0265 | **18.2x** | solid |
| data at p4_big (600k -> 1.09M) | +0.0149 | **10.3x** | solid |
| epochs (6 -> 12) | +0.0106 | **7.3x** | solid |
| data at p4_scratch (600k -> 1.09M) | +0.0081 | **5.5x** | solid |

**All four levers are real**, including the +0.0081 flagged as provisional in the morning summary.

**Limitation to state plainly:** this is a variance estimate from a *single pair* of runs. One
observed delta of 0.0015 does not pin down the true standard deviation — the real spread could be
meaningfully wider. But the narrowest margin here is 5.5x and the widest 18x, so the ordering and the
conclusions hold under any reasonable inflation of the estimate. A third seed would tighten it; it is
not worth a GPU-day given the margins.

Full table at this point:

| model | R@P>=.90 | R@P>=.93 | R@P>=.95 | R@P>=.97 | AP |
|---|---|---|---|---|---|
| v2-224 (600k) | .859 | .811 | .784 | .747 | .9248 |
| v2-224full | .872 | .846 | .828 | .771 | .9329 |
| v2-big (600k) | .879 | .863 | .846 | .784 | .9444 |
| v2-bigfull | .925 | .898 | .889 | .860 | .9593 |
| v2-bigfull-s2 | .935 | .912 | .900 | .883 | .9608 |
| **v2-big12** | .931 | **.922** | **.911** | **.890** | **.9699** |
| mnv4-v2 | .919 | .905 | .893 | .889 | .9498 |
| check_orientation | **.948** | **.945** | .906 | .478 | .9457 |

## 2026-09-23 17:14 — epochs help the SMALL model more, but capacity still wins

`v2-224full12` (p4_scratch, 12 epochs) done: oid .9468, library(full) .9707, subset .9737.

| model | CPU ms | R@P>=.93 | R@P>=.97 | AP | one-press | missed | shown |
|---|---|---|---|---|---|---|---|
| v2-224full (6ep) | 22.5 | .846 | .771 | .9329 | 574 | 29 | 1090 |
| **v2-224full12 (12ep)** | 22.5 | .867 | .824 | **.9492** | 592 | 21 | 1053 |
| v2-bigfull (6ep) | 43.2 | .898 | .860 | .9593 | 616 | 18 | 1017 |
| **v2-big12 (12ep)** | 43.2 | .922 | .890 | **.9699** | 631 | 12 | 1015 |
| mnv4-v2 | 62.0 | .905 | .889 | .9498 | 619 | 27 | 999 |
| check_orientation | 323.7 | .945 | .478 | .9457 | 628 | 16 | 906 |

```
EPOCHS on p4_scratch (6 -> 12): AP .9329 -> .9492  = +0.0164  (10.9x noise)
EPOCHS on p4_big     (6 -> 12): AP .9593 -> .9699  = +0.0106  ( 7.1x noise)
```

**The epoch lever is LARGER for the smaller model** — stretching the LR schedule helps more when
capacity is tight. But it does not close the capacity gap: .9492 vs .9699 is still 0.0207 apart, 14x
the noise floor.

**Correction to the 16:28 entry.** I wrote that `v2-224full12`'s ep10 subset score (.9731) being above
`v2-big12`'s final subset (.9720) meant the 4x smaller model might match it at half the cost. It does
not — the full 13,352-image PR curve puts them 0.0207 apart. The 3,500-image subset metric is the
coarsest number in the log and I read it as a preview of the real one, which it is not. Same mistake
shape as the 8-threshold grid earlier: judge on the full curve or not at all.

Useful secondary result: **`v2-224full12` matches `mnv4-v2` on AP (.9492 vs .9498) at a third of the
cost** (22.5 ms vs 62.0 ms), making it the better option if a cheap model is ever wanted.

Shipping recommendation unchanged: **`v2-big12`, 43.2 ms, 631 one-press / 12 missed, AP .9699** —
still the only single model beating check_orientation on both counts.

## 2026-09-23 19:14 — RESOLUTION SETTLED: real 384 detail is a wash at convergence

`v2-384real` (p4_scratch, res 384, **real 384 px source data**, 1.09M images, 6 epochs, RX 9060 XT,
14 h) finished: oid .9438, library(full) **.9706**.

Against `v2-224full` (same architecture, same data, same epochs, res 224): **.9707**. A difference of
0.0001 — the two are indistinguishable.

Per-epoch trajectory tells the real story:

| epoch | v2-224full | v2-384real | delta |
|---|---|---|---|
| 0 | .9251 | .9377 | +1.26 |
| 1 | .9523 | .9574 | +0.51 |
| 2 | .9626 | .9643 | +0.17 |
| 3 | .9646 | .9649 | +0.03 |
| 4 | .9686 | .9703 | +0.17 |
| 5 | .9691 | .9706 | +0.15 |

**Real 384 detail buys faster early convergence, not a higher ceiling.** The 224 model has erased a
1.26-point head start by epoch 3, and the remaining difference sits inside the +/-0.4 noise band.

So the original "resolution is the weaker lever" verdict was **correct after all** — even though the
experiment behind it was broken (it fed 256 px images upsampled to 384). Rebuilding the dataset at
384 was still the right call: the alternative was shipping a conclusion that rested on upsampled
inputs, and it could equally have overturned the verdict. Declared confound stands: batch size also
differs (32 vs 96, memory-forced at 384).

`hist.json` saved locally; `model.pt` persists, so the posterior dump can be
redone after the reboot. Two dump attempts failed on scheduling issues, neither costing anything
permanent: the AMD job would not schedule despite the device being free and unclaimed, and the
NVIDIA rerun hit a stale GPU-allocation error.

**`mnv4-v2-12` will be killed mid-run by the reboot** (~ep10 of 12). Deliberate: it is the least
valuable run — an ensemble partner whose ep8 already equalled the 6-epoch final — and not worth
delaying the user's migration for.

## 2026-09-23 20:12 — mnv4 epoch lever (ep10 checkpoint), and the ensemble ceiling

`dumpmnv412` succeeded off the **ep10 checkpoint** that survived only because `train.py` was
patched to `torch.save` inside the epoch loop. Saved a ~5h retrain. The run never reached ep12,
so the LR never fully annealed — every number below is a **lower bound** on true 12-epoch mnv4.

Rebuilt the scorer locally as `~/rotfix-experiment/ap.py` (full PR curve + AP from the npz,
never the 8-threshold grid). It reproduces the recorded table **exactly** for v2-big12,
mnv4-v2, v2-bigfull and v2-224full12 on all of AP / R@P>=.93 / R@P>=.97 / one-press / missed,
so the new rows are directly comparable. (`missed` = rotated images the model gets wrong at
pure argmax in *any* way, including right-flag-wrong-angle — not just the ones it fails to surface.)

| model | CPU ms | R@P>=.93 | R@P>=.97 | AP | one-press | missed |
|---|---|---|---|---|---|---|
| mnv4-v2 6ep | 62.0 | .905 | .889 | .9498 | 619 | 27 |
| mnv4-v2-12 **ep10** | 62.0 | .922 | .903 | **.9526** | 637 | 27 |

**The epoch lever does not clearly generalise to MobileNetV4.** +0.0028 AP is only 1.9x the
0.0015 noise floor, against +.0106 (p4 large) and +.0164 (p4 small). Even granting it is a
lower bound, it is an order of magnitude weaker than on the equivariant nets.

But AP hides the part we ship: at the high-precision end the same checkpoint moves
**one-press 619 -> 637 (+18)** and R@P>=.97 .889 -> .903. AP is dominated by the low-precision
tail we never operate in. Worth remembering when judging future arms — cf. the two earlier
times a summary statistic misled me.

It does not change the ranking. At ep10 mnv4 is still AP .9526 vs v2-big12's .9699 for
**44% more CPU**. v2-big12 remains the model to ship.

### Ensembles re-scored with the better mnv4

| combo | AP | R@P>=.93 | R@P>=.97 | one-press | missed |
|---|---|---|---|---|---|
| v2-big12 alone | .9699 | .922 | .890 | 631 | 12 |
| v2-big12 + mnv4-v2 (6ep) | .9712 | .942 | .900 | 636 | 14 |
| v2-big12 + mnv4-v2-12 ep10 | **.9747** | .945 | .906 | 642 | 12 |
| + v2-bigfull (three-way) | .9749 | .957 | .905 | 643 | 12 |

**The three-way ensemble is dead.** Two models now reach .9747, beating the old three-way
(.9740), and the third model buys +0.0002 AP and +1 one-press — inside noise. Ensembling is
a two-model game or nothing.

And even the two-model version stays a bad trade for this feature: 43.2 + 62.0 = **105ms**
(2.4x) to gain +11 one-press images out of 693. Consistent with the earlier call the user
pushed back on and was right about — the ensemble gain has never justified its cost here.

## 2026-09-23 20:26 — v2-384real scored: resolution is NOT a wash, and the subset metric lied again

GPU scheduling stayed broken after the reboot, so I stopped fighting it and ran the dump **CPU-only**:
13,352 images, 16 threads, **13 minutes**. A GPU was never needed for a 190k-param model at this
scale. First attempt died on the container's default 64MB `/dev/shm` — fixed with an
a memory-backed `/dev/shm`.

| model | res | bs | CPU ms | R@P>=.93 | R@P>=.97 | AP | one-press | missed |
|---|---|---|---|---|---|---|---|---|
| v2-224full 6ep | 224 | 96 | 22.5 | .846 | .771 | .9329 | 574 | 29 |
| **v2-384real 6ep** | 384 | 32 | 59.3 | .870 | .811 | **.9521** | 593 | 16 |
| v2-big12 12ep | 224 | 96 | 43.2 | .922 | .890 | .9699 | 631 | 12 |

**+0.0192 AP — 12.8x the noise floor.** I predicted this would land inside noise. It did not.

### I was reading the wrong number

The prediction rested on epoch-level library accuracy: 384's .9706 vs 224's .9707. That is the
**3,500-image subset metric**, which I have a standing rule never to judge on, and it misled a
third time. Full-set 4-way accuracy is .9706 vs .9671 — and the split shows why the subset was
useless here:

| model | 4-way acc (all) | acc on **rotated** only |
|---|---|---|
| v2-224full | .9671 | .9582 |
| v2-384real | .9706 | **.9769** |

Overall accuracy moves +0.35 pts; accuracy **on the rotated class moves +1.87 pts**. Rotated
images are 5.2% of the library, so any aggregate accuracy figure is 95% dominated by upright
images we do not care about. The product metric lives entirely in that 5%.

### But the cause is confounded, and I cannot separate it

384 ran at **bs32** against 224's **bs96** — 3x the gradient steps for the same 6 epochs. The
step lever on this exact small model is worth **+.0164 for 2x steps** (v2-224full -> v2-224full12).
So 3x steps plausibly accounts for most or all of +.0192. The honest claim is only:

> **384@bs32 beats 224@bs96 by +.0192 AP.** Resolution and optimisation steps are not separable
> from this pair. A clean test needs 384@bs96 or 224@bs32, which was never run.

### Product conclusion is unchanged

At 59.3 ms for AP .9521, the 384 small model is **strictly dominated** by v2-big12: 43.2 ms for
.9699. 37% more CPU for 1.8 points less AP. Spending the budget on capacity beats spending it on
pixels, whatever the underlying mechanism.

This also kills the open "do resolution and capacity stack?" question on cost alone —
p4_big@384 measures **155.7 ms**, 3.6x v2-big12, far outside a minor feature's budget. Not worth
running whatever the answer.

**v2-big12 remains the model to ship.**

# 2026-09-23 20:30 — FINAL SUMMARY: is rotation detection worth building into Immich?

The experiment matrix is complete. Every arm below was trained on the identical 1,094,589-image
derived OpenImages set and scored on the same 13,352-image held-out library slice (693 genuinely
rotated, 5.2%) in **stored** orientation. All numbers are full PR curves + AP from the posterior
dumps in `probs/`, never the subset metric or the 8-threshold grid.

**Noise floor: 0.0015 AP** (identical-config seed repeat).

## Final results

| model | licence | CPU ms | R@P>=.93 | R@P>=.97 | AP | one-press | missed |
|---|---|---|---|---|---|---|---|
| v2-224full 6ep | ours | 22.5 | .846 | .771 | .9329 | 574 | 29 |
| v2-224full12 12ep | ours | 22.5 | .867 | .824 | .9492 | 592 | 21 |
| v2-384real 6ep (bs32) | ours | 59.3 | .870 | .811 | .9521 | 593 | 16 |
| v2-bigfull 6ep | ours | 43.2 | .898 | .860 | .9593 | 616 | 18 |
| v2-bigfull-s2 (seed) | ours | 43.2 | .912 | .883 | .9608 | — | — |
| **v2-big12 12ep** | **ours** | **43.2** | **.922** | **.890** | **.9699** | **631** | **12** |
| mnv4-v2 6ep | ours | 62.0 | .905 | .889 | .9498 | 619 | 27 |
| mnv4-v2-12 ep10 | ours | 62.0 | .922 | .903 | .9526 | 637 | 27 |
| check_orientation | **CC-BY-NC** | 323.7 | .945 | .478 | .9457 | 628 | 16 |
| v2-big12 + mnv4-v2-12 | ours | 105.2 | .945 | .906 | .9747 | 642 | 12 |

## What moves the needle (all vs the 0.0015 noise floor)

| lever | gain | verdict |
|---|---|---|
| capacity (small -> big) | **+.0265** | 18x noise. The single best lever. |
| training steps (6 -> 12ep) | +.0106 large, **+.0164 small** | Real, and cheaper than capacity. Free at inference. |
| data volume (quarter -> full) | +.0149 large, +.0081 small | Real; more helps the bigger model more. |
| resolution 224 -> 384 | +.0192, **confounded** with 3x steps (bs32 vs bs96) | Costs 2.6x CPU. Dominated either way. |
| codec (webp vs jpeg) | ~0 for p4, +4.1pts @P>=.97 for mnv4 | Matters only for non-equivariant nets. |
| second model (ensemble) | +.0048 | 2.4x CPU for +11 images of 693. Not worth it. |
| third model (ensemble) | +.0002 | Dead. |
| rotation augmentation | exactly 0 | Provably a no-op for C4-equivariant nets. |
| EXIF orientation tag as a gate | negative | `orientation=1` is 5.36% rotated vs 5.19% baseline; **missing tags are 6.6x safer**. Refuted. |

## Recommendation: yes, ship it — one model, no ensemble

**v2-big12**: 757k params, 43.2 ms single-threaded CPU, no GPU required, ~3MB. Apache-compatible
(trained by us on OpenImages), unlike check_orientation which is CC-BY-NC and unshippable.

It **beats check_orientation on the metric that matters** (AP .9699 vs .9457) while being
**7.5x faster**, and check_orientation collapses at the high-precision end (R@P>=.97 of .478
against our .890) — it simply cannot support a pre-rotated one-press band.

Mapped onto the two-band utility screen:

- **Band A, pre-rotated, approve with one press** (P>=0.95): catches **631 of 693** rotated
  images (91%), with ~33 false suggestions mixed in. One press each.
- **Band B, surfaced but not pre-rotated** (argmax, no threshold): catches essentially all of
  the remainder. **12 of 693 (1.7%)** are wrong at argmax in any way and would need the user to
  find them unaided.

At 43 ms/image a 100k-image library is **~72 minutes of one CPU core**, once, and it parallelises
trivially. That is a reasonable price for a feature that is otherwise pure manual labour.

## Caveats worth stating in any proposal

- Ground truth is **one family's 13.4k-image library**, human-verified. Generalisation to other
  libraries is untested; the 5.2% rotated base rate especially will vary.
- **180 degrees is the weak class everywhere**, in every model and every configuration tested.
- The training set is derived OpenImages, not consumer photos. A consumer-photo training set would
  likely do better still.
- Training cost was ~5h on a mid-range GPU per 12-epoch arm. Retraining is not a casual operation,
  but it is a one-off — the artefact ships as a static model.

## 2026-09-23 20:52 — inference cost re-measured as THROUGHPUT, across real homelab-class CPUs

Two corrections to how cost was reported. First, the whole table was **batch-1 latency at 4
threads**, but the number that matters for a library scan is **throughput** — batched, all cores.
Second, every figure came from one machine, and I never checked whether that machine is
representative.

It is not obviously representative: the benchmark host is an **Intel Xeon E5-2699A v4**
(Broadwell-EP, Q1 2016, 2.2 GHz under load, AVX2, no AVX-512), and we used 4 of its 88 threads.

Re-measured at **BS=32, all cores within each node's budget**, p4 nets, 224px, 20s steady-state:

| CPU | cores | p4_scratch | p4_big | 100k lib (big) | 100k lib (scratch) |
|---|---|---|---|---|---|
| **Celeron N5105** @2.0 (Jasper Lake, 4c) | 3 | 14.8 img/s | **5.0 img/s** | **5.5 h** | 1.9 h |
| Ryzen 5 PRO 3400GE (Zen+ APU, 4c/8t) | 6 | 26.5 | 10.7 | 2.6 h | 1.0 h |
| Xeon E5-2699A v4 (Broadwell, 2016) | 16 | 53.9 | 27.4 | 61 min | 31 min |
| Xeon Gold 6244 @3.6 (Cascade Lake) | 16 | 67.7 | 33.0 | 51 min | 25 min |

**The benchmark host is mid-pack, and the spread is 6.6x.** An N5105 mini-PC/NAS — one of the most
common boxes self-hosted Immich actually runs on — is **5.5x slower than the benchmark host** on p4_big.

Two things I would have got wrong by reasoning instead of measuring:

1. **More cores barely help.** The Xeon E5-2699A v4 went from 59.7 ms/img (batch-1, 4 threads) to 36.5 ms/img
   (BS=32, 16 threads) — 4x the threads and batching bought **1.6x**. These models are too small to
   parallelise well. Do not assume a big host solves this.
2. **The scratch:big cost ratio is not constant — it widens as the CPU gets weaker.**
   Xeon Gold 2.05x, Xeon E5 1.97x, Ryzen 5 3400GE 2.48x, **Celeron N5105 2.96x**. p4_big's wider channels fall out of
   the N5105's small cache. The cheap model gets *relatively* cheaper exactly where cost matters most.

The recorded 43.2 ms / 22.5 ms figures stay valid as **relative** comparisons (all arms measured
identically), but as absolute deployment cost they are optimistic for weak hardware and pessimistic
for batching. Restate cost as throughput on a named CPU going forward.

### Does this reopen p4_scratch?

The accuracy retraction stands — v2-224full12 is AP **.9492** vs v2-big12's **.9699**, one-press
592 vs 631, and that came from the full curve, not the subset metric that originally oversold it.

But the product trade is now hardware-dependent:

| | one-press | missed | 100k on N5105 | 100k on Xeon Gold |
|---|---|---|---|---|
| v2-224full12 (scratch) | 592/693 | 21 | 1.9 h | 25 min |
| v2-big12 | 631/693 | 12 | 5.5 h | 51 min |
| **cost of the extra 39** | +39 | -9 | **+3.6 h** | +26 min |

On a modern CPU that is 26 minutes for 39 more one-press images and 9 fewer misses — trivially
worth it. On an N5105 it is +3.6 hours.

**Still ship v2-big12 as the single model.** The scan is a one-off background pass, and 5.5 h on an
N5105 is a rounding error against what Immich already makes that box do — on the one platform where
both were measured (Alder Lake iGPU, OpenVINO), Immich's existing per-image pipeline is ~66 ms/img
(CLIP 31.5 + buffalo_l det 22.7 + rec ~12), and a 100k initial index on an N5105 runs for days.
Adding one more pass that finishes in 5.5 h is not the bottleneck. *(Caveat: that ~66 ms is iGPU
OpenVINO, not the CPU figures above — the two are not directly comparable, and an apples-to-apples
CPU measurement of Immich's own models was never run.)*

p4_scratch is not dead, though. It is the obvious lever if a low-power tier is ever wanted — it
costs 39 of 693 one-press images (5.6% of rotated images) for a ~3x speedup on exactly the hardware
that needs it. Worth keeping the checkpoint. Not worth shipping two models for a minor feature
without evidence anyone is asking.

## 2026-09-23 21:20 — anchored on how immich-ml ACTUALLY runs inference. Two real findings.

Read `machine-learning/immich_ml/`. What Immich does on CPU:

- **`sessions/ort.py:185`** — CPU defaults are `inter_op_num_threads = 1`, **`intra_op_num_threads = 2`**,
  comment: *"avoid thread contention between models"*. So each model gets **2 threads**, not all cores.
  Everything I benchmarked at 4 threads / 16 threads / BS=32 was the wrong configuration.
- **`config.py:60,64`** — `workers = 1`, `request_threads = os.cpu_count()`. Concurrency comes from a
  request thread pool, not from batching.
- **`config.py:74`** — `max_batch_size = None`, and grepping `max_batch_size` shows batching exists
  only for `facial_recognition` and `ocr`. **CLIP and any per-image classifier run at batch 1.**

So the deployment shape is **ONNX Runtime, batch 1, 2 intra-op threads**, N images in flight.

### Finding 1: the model does not export to ONNX as written

`torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::rot90' to ONNX opset
version 17 is not supported`.

`models.py` calls `materialise()` **inside `forward()`**, so `rot90`/`roll` run on every inference.
The module docstring claims the opposite — *"Group convs are ordinary convs whose weights are
materialised from a smaller shared weight ... so the exported graph is a plain CNN with no custom
ops"* — but that only holds if materialisation happens at export time, which it never did.

Fixed with `scratchpad/bake.py`: walks the module tree and replaces every module exposing
`materialise()` with a plain `BakedConv` holding the precomputed weights. **Verified exact** —
`max|diff| vs eager = 0.00e+00` for both p4_scratch and p4_big. Exports cleanly afterwards.
This is a required, and now solved, porting step for any Immich PR.

### Finding 2: ORT + baking is much faster than my PyTorch numbers

Every previously reported cost was PyTorch eager **including per-forward weight materialisation**.
p4_big on the Xeon Gold: PyTorch 2-thread batch-1 **66.7 ms** -> ORT baked 2-thread **28.2 ms**. **2.4x**.
All earlier absolute cost figures in this file are therefore pessimistic by roughly that factor.

### Measured under Immich's own session options (ONNX Runtime 1.30.0, batch 1, inter_op=1)

ms per image. `buffalo_l recog` assumes ~2 faces/image; the others are one pass.

| model | role | Celeron N5105 | Xeon Gold 6244 |
|---|---|---|---|
| | | 1 thr / **2 thr** / 4 thr | 1 thr / **2 thr** / 4 thr |
| **p4_scratch (ours)** | rotation | 106.3 / **60.3** / 57.2 | 15.3 / **9.3** / 7.7 |
| **p4_big (ours)** | rotation | 383.9 / **208.1** / 178.0 | 48.9 / **28.2** / 22.3 |
| CLIP ViT-B-32 visual | smart search | 678.7 / **350.4** / 294.0 | 74.4 / **50.9** / 31.6 |
| buffalo_l detection | faces | 1695.1 / **870.5** / 676.2 | 198.0 / **102.9** / 63.4 |
| buffalo_l recognition x2 | faces | 7181.1 / **4024.2** / 3014.0 | 505.2 / **274.3** / 146.4 |
| **existing pipeline total** | | **5245.1** | **428.1** |

### The headline

**p4_big costs ~55-60% of a single CLIP pass, and ~4-7% of Immich's existing per-image CPU work.**

| | Celeron N5105 | Xeon Gold 6244 |
|---|---|---|
| p4_big vs CLIP alone | 59% | 55% |
| p4_big vs full existing pipeline | **4.0%** | **6.6%** |

The ratio is stable across a 7x hardware spread, which makes it the number to quote. Note the
2-thread default barely benefits from a 3rd/4th thread (N5105 p4_big 208 -> 178 ms), so
`intra_op=2` is a sensible default for this model too.

Scratch vs big under ORT is a **3.0-3.5x** gap (wider than PyTorch's 2x), so the low-power-tier
argument for p4_scratch is stronger here, not weaker — but p4_big is still cheaper than CLIP on
every box tested, so there is no cost case for shipping two models.

## 2026-09-23 22:05 — CLIP reuse tested directly and properly. Closed.

Earlier this file said *"Immich's existing CLIP embeddings would have the same problem"* as the
DINOv2 pooled probe. That was an **inference from a DINOv2 result, never a measurement.** The user
asked whether we had simply compared the CLIP embedding against text like "rotated 90 degrees
clockwise". We had not. Now tested, on the real model immich-ml loads
(`immich-app/ViT-B-32__openai`, its own preprocess_cfg, its own tokenizer with 77-token padding),
all 13,352 library images embedded at all 4 rotations — 53,408 visual passes, 1,312 s.
Embeddings cached as `clip_embed.npz`.

### 1. How rotation-invariant is the CLIP image embedding, really?

| pair | cosine |
|---|---|
| stored vs stored rotated 90 CW | 0.8819 +/- 0.0489 |
| stored vs stored rotated 180 CW | 0.8415 +/- 0.0718 |
| stored vs stored rotated 270 CW | 0.8800 +/- 0.0504 |
| stored vs a **random different image** | 0.5911 +/- 0.0784 |

Rotating an image moves its embedding only ~30% of the way towards "an entirely different photo".
Not perfectly invariant — there *is* residual signal — but most of it is thrown away.

### 2. Zero-shot against directional text prompts

| prompt set | 4-way acc | upright-kept |
|---|---|---|
| "a photo" / "rotated 90 degrees counterclockwise" / ... | **0.6668** | 0.6874 |
| "an upright photo" / "a sideways photo turned to the left" / ... | 0.5695 | 0.5895 |
| "upright" / "rotated left" / "upside down" / "rotated right" | 0.4320 | 0.4399 |

Best case flags **31% of upright images**. Above the 25% chance floor, and completely unusable.

### 3. Rotate-and-score ("which of the 4 rotations looks most upright?")

| prompt | 4-way acc |
|---|---|
| "an upright photograph" | 0.0700 |
| "a correctly oriented photo" | 0.0482 |
| "a normal photo the right way up" | 0.0207 |

**Far below the 0.25 chance floor** — systematically anti-correlated. CLIP's text-image similarity
to "upright" is driven by content, not orientation, and whatever it does track points the wrong
way. Not worth chasing; noted because it is the approach that sounds most likely to work.

### 4. Trained linear probe on the CLIP pooled vector — the linear ceiling

70/30 split, all 4 rotations as training samples, `LogisticRegression(C=1.0)`:

**4-way acc 0.9089, upright-kept 0.9311** (n_val=4006). Better than the 2026-09-19 DINOv2 pooled
probe (0.835 / 0.917) — CLIP retains more orientation signal than DINOv2 does — but on the product
metric it is not close:

| thr | flagged | P | R |
|---|---|---|---|
| argmax | 446 | **0.220** | **0.488** |
| 0.5 | 108 | 0.324 | 0.174 |
| 0.9 | 3 | 0.667 | 0.010 |

Against `v2-big12` at argmax: **P=0.671, R=0.983**. The probe finds under half the rotated images
and is wrong four times out of five when it fires.

### Verdict

Reusing the smart-search vectors is dead, now on direct evidence rather than analogy. The best
*any* linear readout of the pooled CLIP vector can do is ~0.91 4-way / 0.93 upright-kept, which is
~920 false positives on a 13.4k library, and its exact-angle product metric is an order of
magnitude off what the dedicated model delivers. Zero-shot text is far worse than the trained
probe, which is expected — the probe picks the optimal direction in embedding space, zero-shot
picks one fixed direction chosen by a text encoder that has no reliable notion of "upright".

The earlier conclusion was right. The reasoning behind it was only ever half-measured, and now is.
