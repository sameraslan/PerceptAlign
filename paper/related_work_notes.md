---
title: PAL related work notes
date: 2026-11-02
status: working notes for the arXiv rewrite (not paper prose)
---

All entries in `refs.bib` were checked on 2026-11-02 against a primary record (arXiv API, Crossref DOI record, PubMed, NeurIPS proceedings page, or the project's own repository). Anything not checked that way is marked **UNVERIFIED**. Columns used below: **Timing during generation?** (does the method act on sync while training or sampling the generator), **Sync metric?**, **Frozen critic?**.

## 1. Errors in the old draft (Perceptual-Alignment-Paper.pdf)

### 1a. Reference list

| Old entry | Problem | Fix (key in refs.bib) |
|---|---|---|
| Afouras et al. 2020 | No venue | ECCV 2020, LNCS pp. 208-224 (`afouras2020self`) |
| Arandjelovic & Zisserman 2018 | No venue | ECCV 2018, LNCS pp. 451-466 (`arandjelovic2018objects`) |
| Chen et al. 2020, IEEE TIP 29:8292-8302 | Correct | `chen2020regnet` |
| Chen et al. 2022, MTDVocaLiST | Year is the arXiv year; no venue. Published at ICASSP 2024, pp. 7755-7759 | `chen2024mtdvocalist` |
| Chung & Zisserman 2016, ACCV Workshops | Correct but incomplete: LNCS 10117, pp. 251-263, Springer (printed 2017) | `chung2016outoftime` |
| Chung, Chung & Kang 2019, ICASSP | Correct; add pp. 3965-3969 | `chung2019perfect` |
| Comunita et al. 2023, SyncFusion | No venue; 2023 is the arXiv year. Published ICASSP 2024, pp. 936-940 | `comunita2024syncfusion` |
| Du et al. 2023, Foley analogies | No venue: CVPR 2023, pp. 2426-2436 | `du2023conditional` |
| Hsieh, Colas & Kanwisher 2012 | Citation correct (J Neurophysiol 107(12):3428-3432) | `hsieh2012spatial` |
| Iashin & Rahtu 2021 | No venue: BMVC 2021 (oral) | `iashin2021taming` |
| Iashin et al. 2022 | No venue: BMVC 2022 (spotlight) | `iashin2022sparse` |
| Itseez 2015, OpenCV | **UNVERIFIED** software cite; the usual canonical OpenCV cite is Bradski 2000 (Dr. Dobb's). Keep only if OpenCV is mentioned in the text | not in bib |
| Kim, Hong & Ro 2022, VCA-GAN | Wrong year and no venue: NeurIPS 2021 (vol. 34, pp. 2758-2770) | `kim2021vcagan` |
| Owens & Efros 2018 | No venue: ECCV 2018, LNCS pp. 639-658 | `owens2018audio` |
| Owens et al. **2015**, Visually indicated sounds | Year should be 2016 (CVPR 2016, pp. 2405-2413); 2015 is the arXiv date. In-text cites "Owens et al., 2015" must change too | `owens2016visually` |
| Robert, Webbie et al. 2018, Pydub | **UNVERIFIED**: Pydub is github.com/jiaaro/pydub by James Robert; "Marc Webbie" could not be confirmed as an author. Cite as software with URL or drop | not in bib |
| Rombach et al. 2022 | No venue: CVPR 2022, pp. 10674-10685 | `rombach2022ldm` |
| Takeshima 2021, PLoS ONE 16(12):e0261129 | Correct | `takeshima2021visual` |
| Wang*, Li* et al. 2020, denseflow | Remove the equal-contribution asterisks from author names | `wang2020denseflow` |
| Winkler, Denham & Nelken 2009, TICS 13(12):532-540 | Correct | `winkler2009modeling` |
| Zhou et al. **2017**, Visual to sound | Year should be 2018 (CVPR 2018, pp. 3550-3558); 2017 is the arXiv date. In-text "Zhou et al., 2017" must change | `zhou2018visual` |
| "Niu and Mak" (in text, Sec. 3.4.1) | Missing from the reference list entirely | `niu2023synchronization`: Zhe Niu and Brian Mak, "On the Audio-visual Synchronization for Lip-to-Speech Synthesis", ICCV 2023, pp. 7809-7818 |

**Niu & Mak identification (confident).** Their offset predictor has exactly the block the old draft copied: two local feature extractors (video, audio), each "two Conv1D-BN-GELU blocks, a fully-connected layer and an L2 normalization", first Conv1D kernel 3 and second kernel 1, with video features linearly upsampled to the mel frame rate before the first Conv1D. Confirmed from the paper text (Sec. 3.3.1).

### 1b. Factual errors in the old text (not just formatting)

1. **VAS attribution.** VAS (Visually Aligned Sounds, 8 classes: dog, fireworks, drum, baby, gun, sneeze, cough, hammer) was introduced by **Chen et al. 2020** (RegNet, IEEE TIP), per the RegNet repository README. Zhou et al. 2018 introduced **VEGAS**; four VAS classes come from VEGAS and four from AudioSet. Any sentence crediting VAS to Zhou et al. 2018 should cite Chen et al. 2020 (optionally "derived in part from VEGAS (Zhou et al. 2018)").
2. **"Early efforts utilized recurrent neural networks (Owens et al.; Zhou et al.) ... employing an example-based synthesis procedure ... This work led to the creation of the Greatest Hits and VEGAS datasets."** Greatest Hits and the example-based synthesis come from Owens et al. 2016; VEGAS and the SampleRNN-style direct waveform generation come from Zhou et al. 2018. The sentence merges the two; split the attribution.
3. **"notable efforts in audio-visual synchronization exist, but they align audio and video post-generation" / "first to focus on improving alignment during the generation process".** False as of 2023 and more so now. See Section 3 below: Niu & Mak 2023 (ICCV) trained a generator with a sync loss; Kim et al. 2021 (NeurIPS) used a contrastive sync objective in a lip-to-speech generator; Wav2Lip 2020 used a frozen sync expert as a generator loss; Diff-Foley (NeurIPS 2023) applies an alignment classifier as guidance during sampling; CondFoley (CVPR 2023) re-ranks samples with a sync model.
4. **Takeshima 2021 as support for "the brain recalibrates ... much like a loss function".** Takeshima measures the point of subjective simultaneity, the temporal binding window and rapid recalibration across central versus peripheral visual fields (TOJ, SJ, stream/bounce). It does support "TBW and rapid recalibration exist and depend on the task". For recalibration as a general phenomenon, Fujisaki et al. 2004 and the Vroomen & Keetels 2010 review are the standard cites. The "like a loss function" analogy is the authors' own framing and should be presented as an analogy.
5. **Hsieh et al. 2012 paraphrase.** The paper shows early auditory cortex *contains information about* a visual stimulus when the visual stimulus implies sound, and shows no visual influence when it does not. "Capable of processing visual stimuli" overstates it. It is about cross-modal content; it says nothing about timing, so it is weak support for a temporal-alignment loss. Optional to keep.
6. **"Real-time" generation.** SpecVQGAN's own claim is generating a clip "in less time than it takes to play it on a single GPU". "Real-time" is acceptable only with that qualifier; streaming or causal real-time generation is not what SpecVQGAN does.
7. **Symmetric Gaussian TBW.** The human window is asymmetric: people tolerate audio lagging video more than audio leading (Dixon & Spitz 1980; van Wassenhove et al. 2007; reviewed in Vroomen & Keetels 2010). A zero-centred symmetric Gaussian is a simplification; either say so or make the window asymmetric. Reviewers in this area will notice.

## 2. Per-paper notes

### 2a. Base model and tools

- **iashin2021taming** (SpecVQGAN, BMVC 2021). VQGAN spectrogram codebook plus GPT transformer conditioned on visual features; Melception-based FID and MKL metrics. Timing during generation: no explicit sync objective. Sync metric: none (relevance and fidelity only). This is PAL's base model.
- **esser2021taming** (VQGAN, CVPR 2021). The codebook-plus-transformer recipe SpecVQGAN adapts. Cite for the VQGAN codebook.
- **rombach2022ldm** (Latent diffusion, CVPR 2022). The old draft cites it only as an application ("adding audio to generated videos"). Optional.
- **wang2020denseflow** (software). Optical-flow extraction for the BN-Inception flow stream.

### 2b. Datasets and early V2A

- **owens2016visually** (CVPR 2016). Greatest Hits dataset; predicts sound features from video and synthesizes by example retrieval. Timing: implicit only.
- **zhou2018visual** (CVPR 2018). VEGAS dataset; direct waveform generation from video frames and flow. Timing: implicit only.
- **chen2020regnet** (IEEE TIP 2020). RegNet; introduces VAS (8 classes) and the "visually irrelevant sound" problem; uses BN-Inception RGB+flow features, the same features SpecVQGAN uses on VAS. Timing: aligned features, no sync loss.

### 2c. Audio-visual sync and correspondence models (critics and metrics)

- **chung2016outoftime** (SyncNet, ACCV-W 2016). Two-stream contrastive embedding for lip sync; offset by sliding window. The ancestor of every sync critic and of LSE-C / LSE-D metrics.
- **owens2018audio** (ECCV 2018). Self-supervised AV features trained to detect artificially shifted audio. Direct precedent for "pretrain a critic on real versus time-shifted pairs".
- **arandjelovic2018objects** (ECCV 2018). AV correspondence (semantic rather than temporal). Cite for contrastive AV embeddings generally.
- **chung2019perfect** (ICASSP 2019). Multi-way (softmax over many offsets) cross-modal matching for sync, replacing pairwise contrastive loss. Very close to PAL's critic objective: softmax over offsets with the true offset as the positive.
- **afouras2020self** (ECCV 2020). Self-supervised AV object localization with a sync-based objective.
- **kadandale2022vocalist** (Interspeech 2022). Transformer AV sync model for lips and voices. MTDVocaLiST distils it.
- **iashin2022sparse** (SparseSync, BMVC 2022). Offset classification for open-domain videos where sync cues are sparse; flags codec artefacts that let models cheat.
- **chen2024mtdvocalist** (ICASSP 2024). Distils VocaLiST into a smaller sync model. Sync model only; no generation.
- **iashin2024synchformer** (Synchformer, ICASSP 2024). Segment-level contrastive pretraining of AV features, then a separate offset classifier. Now the standard open-domain sync critic: MMAudio's DeSync metric and many 2025-26 rewards use it. **Strongly consider reporting DeSync** (or at least Synchformer offset accuracy) on VAS in addition to the DTW metric, since every recent V2A paper does.

### 2d. Sync losses inside generators (the closest precedents)

- **prajwal2020wav2lip** (Wav2Lip, ACM MM 2020). Pretrains a SyncNet-style "expert" on real videos, then trains the face generator with an "expert sync-loss" E_sync = mean of -log P_sync; the paper states the expert's weights "remain frozen during the training of the generator" and argues fine-tuning it on generated frames hurts. Timing during generation: yes. Frozen critic: yes. Modality direction: audio to video (lips). **This is the main precedent for PAL's new frozen-critic design.**
- **kim2021vcagan** (VCA-GAN, NeurIPS 2021). Lip-to-speech GAN with a contrastive-learning synchronization term during training. Timing during generation: yes. Frozen critic: no (learned jointly, per abstract; details not re-checked).
- **niu2023synchronization** (ICCV 2023). Lip-to-speech. Offset predictor: Conv1D-BN-GELU local extractors, cross-correlation over offsets within a radius K, softmax with temperature to a distribution over offsets. The Self-Synchronization Module trains the generator with L_SSM = -log P(k = 0 | video, generated mel). Timing during generation: yes. Frozen critic: no; the SSM predictor has its own parameters and appears to be trained jointly (no pretraining stage described). **The old PAL (cross-correlation, softmax over offsets with temperature, reward the zero offset) is structurally Niu & Mak's SSM loss plus a Gaussian weighting over offsets near zero and segmentation.** This must be stated openly in the new paper.
- **li2024latentsync** (LatentSync, arXiv 2024). Adds SyncNet supervision to an audio-conditioned latent diffusion lip-sync model and studies how SyncNet convergence affects it. Timing during generation: yes. Frozen critic: pretrained SyncNet supervision (exact freezing not re-checked). Shows the Wav2Lip recipe carried into diffusion models.

### 2e. Video-to-audio generation, 2022 onward

- **sheffer2023im2wav** (Im2Wav, ICASSP 2023). CLIP-conditioned two-stage transformer over VQ-VAE audio tokens; ImageHear benchmark. Semantics only; no timing mechanism.
- **du2023conditional** (CondFoley, CVPR 2023). Conditional Foley via a user-supplied AV example; autoregressive over spectrogram VQGAN codes (close to SpecVQGAN). **Uses an AV sync model at test time to re-rank many samples** and pick the best-aligned one. Onset metrics on Greatest Hits: onset-count accuracy and AP of onsets within 0.1 s. Timing during generation: only via sampling plus re-ranking. Good extra baseline idea: "SpecVQGAN plus re-ranking with the same critic".
- **luo2023difffoley** (Diff-Foley, NeurIPS 2023). CAVP: contrastive AV pretraining with a semantic term (negatives from other videos) and a temporal term (negatives are time-shifted clips from the same video); CAVP features condition an LDM. A separate alignment classifier (positives: aligned pairs; negatives: shifted or mismatched) both defines the Align Acc metric and guides sampling ("double guidance"). Timing during generation: yes, at inference via classifier guidance; CAVP is not used as a loss on generated audio. Frozen critic: CAVP fixed; classifier fixed.
- **wang2024v2amapper** (AAAI 2024). Maps CLIP to CLAP space and generates with AudioLDM. Semantic only; no timing.
- **comunita2024syncfusion** (ICASSP 2024). Extracts onset tracks from video and conditions a diffusion model on them. Timing via explicit onset conditioning.
- **xing2024seeing** (Seeing and Hearing, CVPR 2024). Training-free: an ImageBind-based latent aligner steers pretrained generators during denoising, similar to classifier guidance. Gradients flow through a frozen AV model, but at inference and for semantics more than timing.
- **mei2024foleygen** (MLSP 2024). Codec-token language model for V2A; studies visual attention masks to reduce misalignment. Timing via architecture.
- **wang2024tiva** (TiVA, ACM MM 2024). Predicts a low-resolution "audio layout" from video, then conditions a diffusion model on it. Timing via an intermediate coarse layout. (A later paper describes its method differently; the ACM record title and authors are verified, method summary from secondary sources.)
- **wang2024frieren** (Frieren, NeurIPS 2024). Rectified-flow V2A with channel-level cross-modal fusion for strong temporal alignment; reports Diff-Foley-style alignment accuracy. Timing via architecture.
- **zhang2026foleycrafter** (FoleyCrafter, IJCV 2026; arXiv 2024). Adapts a text-to-audio model with a semantic adapter and a temporal controller driven by an onset detector and timestamps. Timing via onset conditioning. Note the IJCV author list adds Bin Liu relative to arXiv.
- **jeong2025rewas** (ReWaS, AAAI 2025). Estimates an energy curve from video and uses it to control a text-to-audio model. Timing via energy conditioning.
- **viertola2025vaura** (V-AURA, ICASSP 2025). Autoregressive V2A with high-frame-rate visual features and cross-modal fusion; VisualSound benchmark (VGGSound filtered for AV-aligned clips). Same lab as SpecVQGAN and Synchformer; evaluates sync with a Synchformer-based score. The most natural "modern autoregressive" comparison for PAL.
- **cheng2025mmaudio** (MMAudio, CVPR 2025). Flow-matching V2A jointly trained on text-audio data; synchronization module feeds 24 fps Synchformer features into the network. **DeSync metric**: Synchformer-predicted offset in seconds, averaged over the first and last 4.8 s windows of an 8 s clip. Synchformer is used as conditioning features only; no Synchformer training loss.
- **mo2025foleyflow** (Foley-Flow, CVPR 2025). Masked AV alignment pretraining plus dynamic conditional flows. Argues plain contrastive alignment helps semantics but is weak for rhythmic timing (relevant caution for a contrastive critic).
- **ton2025taro** (TARO, ICCV 2025). Flow transformer with timestep-adaptive representation alignment and onset-aware conditioning. Timing via onset conditioning.
- **chan2026v2adpo** (V2A-DPO, ICASSP 2026). DPO for flow-based V2A (Frieren, MMAudio) with an "AudioScore" that includes temporal alignment. Timing during training: yes, through preference pairs scored by a frozen model (a reward rather than a differentiable loss).
- **cheng2026syncdpo** (SyncDPO, arXiv 2026, under review). DPO post-training for joint video-audio generation with rule-based temporally distorted negatives (shift, replace, mask); argues MSE gives weak penalties for small misalignments. Close in spirit to PAL's motivation; uses preference pairs rather than a critic gradient.

### 2f. UNVERIFIED or not added to the bib

- **UNVERIFIED** VisioSonic / "Hear What You See: Video-to-Audio Generation with Diffusion Transformer and Semantic-Temporal Alignment-Ranked DPO" (CVPR 2026). Uses Synchformer and ImageBind to rank candidates for DPO. Title and venue appear on CVF open access, but aggregators give conflicting author lists and the PDF could not be fetched. Check before citing.
- **UNVERIFIED** Training-Free Multimodal Guidance for Video to Audio Generation (Grassucci et al., arXiv 2509.24550, 2025). Inference-time guidance from modality-embedding volume; not checked whether it has a temporal term.
- **UNVERIFIED** Rhythmic Foley (arXiv 2409.08628), Video-to-Audio Generation with Hidden Alignment (VTA-LDM, arXiv 2407.07464, metadata verified on arXiv but left out to keep the list short), Syncphony (audio-to-video, ICLR 2026). Possible extras if space allows.
- Not checked at all: technical reports for HunyuanVideo-Foley, Kling-Foley, ThinkSound. A last search before submission should look for any V2A paper that backpropagates through Synchformer or CAVP into the generator.

### 2g. Human audio-visual timing (for the TBW motivation)

- **dixon1980detection** (Perception 1980). Classic finding that audio-lead is detected at smaller asynchronies than audio-lag. Supports making the window asymmetric.
- **fujisaki2004recalibration** (Nat Neurosci 2004). Exposure to a constant AV lag shifts perceived simultaneity toward that lag. Standard cite for recalibration.
- **vanwassenhove2007temporal** (Neuropsychologia 2007). Temporal window of integration for AV speech (McGurk); asymmetric, wider for audio lag.
- **winkler2009modeling** (TICS 2009). Predictive regularity representations in auditory scene analysis. Peripheral to timing; optional.
- **vroomen2010perception** (Atten Percept Psychophys 2010). Tutorial review of intersensory synchrony: windows, PSS, recalibration. Best single review cite.
- **hsieh2012spatial** (J Neurophysiol 2012). Cross-modal content in early auditory cortex. Weak for timing; optional.
- **stevenson2012individual** (JEP:HPP 2012). Individual TBW width predicts susceptibility to AV illusions. Supports "TBW is a measurable, variable quantity", which could justify treating window width as a tunable hyperparameter.
- **wallace2014construct** (Neuropsychologia 2014). Review of the multisensory TBW construct. Primary cite for the term "temporal binding window".
- **takeshima2021visual** (PLoS ONE 2021). PSS, TBW and rapid recalibration differ by visual field and task. Fine as a secondary cite.

## 3. Novelty assessment

**What the prior work already covers.**

1. *Sync objective during generation (training time).* Wav2Lip (2020, frozen pretrained expert, -log P_sync loss on generated output), VCA-GAN (2021, contrastive sync in lip-to-speech), Niu & Mak (2023, softmax over cross-correlation offsets, -log P(k = 0) on generated mel), LatentSync (2024, SyncNet supervision in diffusion). So "first to improve alignment during generation" is false.
2. *Sync model used during sampling.* Diff-Foley (classifier guidance with an alignment classifier), CondFoley (sync-model re-ranking on a SpecVQGAN-like code model), Seeing and Hearing (frozen ImageBind aligner gradients).
3. *Contrastive AV pretraining for temporal alignment.* Owens & Efros 2018 (shift detection), Perfect Match 2019 (softmax over offsets), Diff-Foley CAVP 2023 (temporal negatives within the same video), Synchformer 2024.
4. *Sync-aware V2A post-training.* V2A-DPO (ICASSP 2026) and SyncDPO (2026) push sync with preference pairs scored or constructed for timing.
5. *Explicit timing conditioning.* SyncFusion, FoleyCrafter, ReWaS, TiVA, TARO, MMAudio.
6. *The loss form itself.* Softmax with temperature over a cross-correlation of local AV embeddings, with the zero offset as target, is Niu & Mak's SSM. PAL's additions are the Gaussian weighting over near-zero offsets (a soft target instead of a one-hot at k = 0) and per-segment computation.

**What still looks defensible (based on a bounded search; not exhaustive).**

- I found no published open-domain video-to-audio (Foley) method that trains the generator with a gradient from a frozen, contrastively pretrained AV sync critic. Recent open-domain work uses sync models as conditioning (MMAudio), guidance (Diff-Foley), re-ranking (CondFoley) or DPO rewards (V2A-DPO, SyncDPO). The Wav2Lip-style frozen-expert loss has been used for lips (video generation) and lip-to-speech, and not, as far as found, for open-domain sound effects with a discrete-token generator.
- Making that loss differentiable for a discrete VQ token generator via soft (expected-codebook) decoding is a concrete technical piece. Prior frozen-critic losses act on continuous outputs (pixels, mels). Check whether straight-through or Gumbel-softmax tricks were used for a sync loss elsewhere before claiming it.
- A perceptually motivated soft target over offsets (binding-window weighting) is a small, specific change relative to Niu & Mak's one-hot zero-offset target. It is defensible as "a variant" rather than a new loss family. An ablation of Gaussian versus one-hot (and versus asymmetric) would make it count.
- FlashDot as a controlled synthetic probe is a reasonable minor contribution.

**Claims to drop.** "First to focus on improving alignment during the generation process"; "first to apply a perception-based temporal alignment loss"; "real-time" without qualification; "advancing current state of the art" (SpecVQGAN on VAS is far behind MMAudio-class models, so do not frame against SOTA).

**Suggested wording (pick one and adapt; avoid "first").**

- Option A (precedent-forward): "Following the frozen sync-expert loss of Wav2Lip and the offset-distribution loss of Niu and Mak, we apply a frozen, contrastively pretrained synchronization critic as a training loss for open-domain video-to-audio generation, and make it differentiable for a discrete-token generator through expected-codebook decoding."
- Option B (scoped novelty): "To our knowledge, this is the first application of a frozen synchronization critic as a differentiable training loss for open-domain, codebook-based video-to-audio generation. Prior open-domain methods use synchronization models for conditioning, sampling guidance, re-ranking or preference optimization."
  Use Option B only after a final search right before submission, and keep "to our knowledge".
- For the window: "We replace the one-hot zero-offset target of Niu and Mak with a Gaussian weighting over small offsets, a simplified stand-in for the human audio-visual temporal binding window (Wallace and Stevenson, 2014)." If the window stays symmetric, add one clause noting that the human window is asymmetric (Vroomen and Keetels, 2010).

**Evaluation implications.** Reviewers will expect: (1) a standard sync metric such as Synchformer DeSync or offset accuracy, alongside the DTW metric; (2) the critic used for training to differ from the critic used for evaluation, or an explicit caveat, since optimizing and scoring with the same model invites reward hacking; (3) baselines that use the same critic in other ways: SpecVQGAN with critic re-ranking (as in CondFoley) and SpecVQGAN with critic guidance during sampling.
