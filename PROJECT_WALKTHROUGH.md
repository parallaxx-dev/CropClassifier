# Crop Classifier — Project Walkthrough

**Multi-country European + India expansion of the Sentinel-2 crop classification system**
Branch: `feature/api-sentinel2-upload` · Written 2026-08-10

This document is a narrative summary of the project for presentation purposes.
For the full dated, line-by-line engineering log (the source of truth for
exact numbers, commands, and file changes), see `progress.md`. For the
architecture/roadmap this work is measured against, see `CLAUDE.md`.

---

## 1. What this system does

A deployable ML system for **crop type mapping from Sentinel-2 satellite
imagery**. A user draws a field boundary (or, eventually, uploads one) and a
date range; the system fetches that field's real Sentinel-2 growing-season
time series, classifies the crop, and returns the predicted class with a
confidence score.

The model is **not** an image classifier — it's a **per-parcel time-series
Transformer**. Each prediction is built from up to 45 real Sentinel-2
observations (13 spectral bands each) across a growing season for one field,
not a single snapshot image. This is what lets the same architecture
generalize across countries and now continents: the shape of a crop's growth
curve over a season, not its look in one photo, is the signal.

---

## 2. Where this session started

The deployed model was trained on a **single country, single year**: France,
2018, 9 classes, via the BreizhCrops dataset. A prior code audit had already
flagged this as the hard ceiling on real-world usefulness — and a
multi-country EuroCrops retraining pipeline had been designed (18 European
countries) but not yet run to completion.

---

## 3. The multi-country retraining pipeline

A new package, `backend/training/eurocrops_pipeline/`, was built to:

- **Download** each EuroCrops country's official parcel shapefile
- **Sample** a capped, stratified set of parcels per class per country
- **Fetch** each parcel's real Sentinel-2 growing-season time series live,
  via the Copernicus Data Space Ecosystem (CDSE) Statistical API — the same
  fetch code path used by the deployed API, so training and inference never
  drift apart
- **Checkpoint** per-country, so a multi-hour fetch survives interruption
  and resumes exactly where it left off
- **Train** and **evaluate** with real per-class metrics (confusion matrix,
  precision/recall/F1) — this didn't exist anywhere in the repo before

The taxonomy was rebuilt from 9 France-specific classes to **11 classes**
harmonized across all of EuroCrops via its official HCAT3 taxonomy: `meadow,
wheat, barley, triticale, rapeseed, maize, sunflower, vineyards, fruit, nuts,
potatoes`.

---

## 4. The fetch journey — and two real incidents

Fetching is not a clean, uneventful process, and this project hit two real
operational problems worth documenting because they're the kind of thing a
"it works on my machine" demo hides:

**Austria** and **Belgium-Flanders** completed fully (1650 and 1464 parcels).
**Germany-Brandenburg** is still partially fetched. **Czech Republic** was
attempted, and hit a genuine data problem: its official EuroCrops zip file
ships with no coordinate reference system file at all. The pipeline's schema
probe correctly caught this — but the very first time it happened, it
**crashed the entire multi-hour fetch run**, because the safety net the
original design called for ("one bad country shouldn't halt everything") had
never actually been implemented. This was fixed live: the fetch loop now
catches a bad country, logs it, cleans up its scratch files, and moves on.

Separately, cumulative API usage across the session (two full countries plus
1000+ India parcels plus dozens of live test calls) **exhausted the CDSE
account's processing-unit quota**, which broke the live prediction API
outright — a real, live 403 error, not a theoretical one. Fixed by rotating
to new API credentials and restarting the affected services.

---

## 5. Results, tracked over three training iterations

| Checkpoint | Training data | Overall accuracy | Key generalization test |
|---|---|---|---|
| Interim (Austria only) | 2,334 parcels, 11 classes | 81.6% | Held-out Belgium: **5.6%** — proof of the single-country ceiling |
| Austria + Belgium complete | 3,114 parcels, 11 classes | 78.3% | Held-out Brittany (never trained on): **42.1%**, up from 5.6% |
| + India (see below) | ~4,300 parcels, 18 classes | 71.4% | India region: 46.3% (training split), **53.4%** on genuinely unseen fields |

Overall accuracy trending down across iterations is **expected, not a
regression** — each step adds a harder, more diverse test set (more
countries, more classes), not a stronger model on the same easy problem.
The generalization numbers are the real headline: one additional region of
training data took cross-region accuracy from single digits to over 40%,
direct proof the multi-country approach works as intended.

---

## 6. The India expansion

### The starting question, and an honest failure

Asked directly whether the model could predict crops in India, the honest
answer required a real test — with zero fabrication. No India-labeled data
existed anywhere in the project. Rather than invent coordinates or labels,
a real public dataset was found and used: the **AgriFieldNet Competition
Dataset** (Radiant Earth Foundation & IDinsight, 2022, CC-BY-4.0) —
GPS-surveyed real farm field boundaries with ground-truth crop labels across
Bihar, Uttar Pradesh, Rajasthan, and Odisha.

**Result with zero India training data: 21.4% overall, and 0/9 (0%) on
wheat specifically** — every wheat field was misclassified. The most likely
cause: India's wheat growing season (sown ~November, harvested ~March-April)
doesn't line up with the calendar position the model expects European winter
wheat's growth curve to occupy.

### Turning that into a fix

Rather than treat this as a dead end, India was integrated as a full new
training source, prioritized ahead of the remaining European countries:

- A new module downloads AgriFieldNet's public field data (no
  authentication required), extracts individual field boundaries and their
  true crop labels directly from the source rasters, and feeds them into the
  **exact same live CDSE fetch pipeline** used for every European country —
  same code path, same rigor.
- The taxonomy grew from **11 to 18 classes**, adding `mustard, sugarcane,
  lentil, rice, gram, garlic, fallow`. Three further AgriFieldNet classes
  (green pea, coriander, bersem) were deliberately **not** added — a full
  survey of the entire public dataset found only 8-12 examples of each,
  ever. That's not a "fetch more later" gap; it's the whole dataset's supply,
  a real and permanent limitation, documented rather than hidden.
- 1,009 real Indian field time series were fetched live and folded into
  training.

### The payoff, verified rigorously

Because a number from a random 80/20 split can look better than reality
warrants, the improvement was re-checked against fields **explicitly
confirmed to have never touched training at all**:

| | Before (no India data) | After (retrain, training split) | After (genuinely held-out fields) |
|---|---|---|---|
| **Wheat** | **0/9 = 0%** | 86% recall | **12/19 = 63.2%** |
| Maize | not tested | 94% recall | **13/14 = 92.9%** |
| Fallow (new class) | n/a | 60% recall | **6/10 = 60.0%** |

The core failure — wheat — is genuinely fixed, confirmed on fields the model
never saw. One honest exception: **mustard**, despite having as much
training data as wheat (150 examples), scored **0/15 (0%)** on the same
held-out test. This is flagged as a real, unresolved finding, not glossed
over — the working theory is that mustard's growth curve may be genuinely
too close to wheat, gram, and sugarcane's for the model to separate yet.

---

## 7. Region-conditioning: a tested idea that didn't pan out

Since growth curves clearly vary by country, an explicit "which region is
this" signal was built into the model — a learned embedding, added to the
transformer alongside the time series, derivable automatically from a
field's location (no extra user input needed). Built cleanly as a toggle,
so it could be A/B tested against the exact same data.

**Result: essentially a wash.** Overall accuracy was statistically
identical (71.5% vs. 71.4%), and India specifically got **slightly worse**
(43.4% vs. 46.3%) — the opposite of the intended effect. The likely
explanation: a Sentinel-2 signal from an Indian field may already look
different enough from a European one (soil, atmosphere, field size) that an
explicit region label doesn't add new information the model didn't already
have. **The simpler, no-region model was deployed instead** — it performs
at least as well, with less complexity. The region-conditioning code stays
in the pipeline, ready to revisit once more, better-balanced regional data
exists.

This is included deliberately: a tested idea that didn't help is a real
result, not a wasted effort — and worth presenting alongside the wins.

---

## 8. Full validation of what's deployed right now

**Whole model** (training's own held-out split): **71.4%** overall accuracy
(ensembled/production-matching evaluation), macro-F1 0.58.

**Crop-wise**: strong on established European classes — rapeseed (0.96 F1),
maize (0.90), barley (0.87). Weakest: `lentil` and `garlic` (0.00 F1 each) —
a **permanent** data ceiling (76 and 43 total examples in the whole public
dataset), not something more training fixes.

**Region-wise**: Austria 79%, Belgium 78%, India 46%. (Germany-Brandenburg
showed 97% in this split, but see below — that number is misleading.)

**Whole pipeline** (live API, real network calls — the number that actually
matters for a real user): Austria 75%, Belgium 73%, **India (genuinely
held-out) 53.4%**. Germany-Brandenburg's live-pipeline number was **33.3%**
— sharply lower than its training-split figure of 97%, because that figure
was computed on only 29 parcels from a still-partial fetch. This gap is
itself a useful finding: a training-eval number in isolation can mislead;
testing the live, deployed pipeline against real requests is what actually
validates the system end to end.

---

## 9. Known gaps and limitations (honest inventory)

- **No shadow masking** (cloud masking exists; shadow does not — no
  reliable signal exists for true L1C imagery yet).
- **No minimum-observation-count quality gate** — a field with 3 usable
  satellite scenes is treated identically to one with 45.
- **No graceful handling of CDSE failures** in the deployed API — a
  transient network or quota issue currently surfaces as a raw 500 error.
  This was hit live during this session's work.
- **No frontend** — the only way to use this system today is a direct API
  call (Postman/curl). Still the single highest-priority gap for making
  this useful to a non-technical user, independent of model accuracy.
- **Positional encoding uses ordinal step position, not real day-of-year** —
  a likely deeper contributor to the India calendar-mismatch problem, not
  yet addressed.
- **`lentil`, `garlic`, and (excluded from the taxonomy entirely) green
  pea/coriander/bersem** are permanently data-limited by the source
  dataset.
- **`mustard`** fails despite adequate training data — open, unexplained.
- **Region-conditioning** exists in code but is not proven beneficial —
  parked, not deleted.

---

## 10. What's deployed and testable right now

- Live API at `http://localhost:8000/api/v1/predict`, running the 18-class
  checkpoint trained on Austria + Belgium-Flanders + partial
  Germany-Brandenburg + India.
- Ready-to-use test requests (real field coordinates with true labels) in
  `backend/training/eurocrops_output/postman_test_parcels_18class.json` and
  `india_test_parcels.json`.

---

## 11. Next steps

1. Continue the EuroCrops fetch — 16 of 18 countries remain.
2. Investigate the `mustard` failure specifically.
3. Try real day-of-year positional encoding — flagged as more promising
   than region-conditioning for the India calendar-mismatch problem.
4. Revisit region-conditioning once more/better-balanced regional data
   exists.
5. Build the frontend — an AOI-drawing map UI wired to the existing API.
6. Add shadow masking, an observation-count quality gate, and graceful CDSE
   error handling to the deployed API.
