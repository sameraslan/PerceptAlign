# PerceptAlign v2: experiment plan

## Why a rerun
The 2023 draft's PAL could not affect training: the predicted spectrogram was decoded from
`argmax(logits)` inside a `@torch.no_grad` function (`decode_to_img`), so no gradient reached the
transformer; the only parameters PAL touched (the two feature extractors) were not in the
optimizer (`configure_optimizers` collects `self.transformer` parameters only); and the offset
loop wrote offset k into index k mod (2K+1), so the "zero offset" slot held offset +K. The
draft's hyperparameter plots also show FlashDot and VAS improving by identical ratios at every
setting. All numbers from the draft are withdrawn.

## Method (v2)
- Sync critic: audio encoder on mel (pooled to the 21.2 fps video rate) and video encoder on the
  2048-d frame features; Conv1D-GN-GELU blocks, L2-normalised 128-d embeddings.
- PAL: per segment, mean cosine similarity at offsets -K..K over the central frames only (no
  padding artefact), softmax with temperature tau, Gaussian binding-window weights
  w_o = exp(-o^2 / 2 sigma^2) (w_0 = 1), loss = -log mean_s sum_o w_o p_so.
- Stage 1: train the critic on real pairs (VQGAN reconstructions and ground truth), sigma = 0.
- Stage 2: freeze critic and VQGAN decoder; L = CE + beta * PAL on straight-through decoded logits
  (random 16-column window, 2-column margin, fp16 autocast).

## Hypotheses
- H1 (metric validity): DTW-Euclidean on mel frames barely responds to known delays, while onset
  F1 and envelope lag do. (metric_sensitivity.py on FlashDot and VAS.)
- H2 (FlashDot): CE+PAL improves tone-activity F1 / onset F1 over CE-only at matched steps, and
  most when training data is small. 3 seeds, data sizes {200, 400, 1600}.
- H3 (VAS): LoRA fine-tuning of the public SpecVQGAN VAS transformer with CE+PAL improves timing
  (onset F1/AP, envelope correlation, |lag|, held-out critic sync accuracy) over CE-only
  fine-tuning with the same budget, at no cost in FID / MKL.
- H4 (window): Gaussian window (sigma > 0) vs one-hot zero offset (sigma = 0, Niu & Mak style).
- H5 (alternatives): PAL training vs critic re-ranking of N samples from the baseline.

## Protocol
- Splits: VAS = SpecVQGAN vas_train minus a 5% stratified dev split / SpecVQGAN vas_valid as test.
  FlashDot = 1600 / 200 / 200. All selection on dev; test is run once per final configuration.
- Training critic != evaluation critic (different seed and architecture).
- Laptop budget: one job at a time, MPS memory capped at about 3.2 GB.
