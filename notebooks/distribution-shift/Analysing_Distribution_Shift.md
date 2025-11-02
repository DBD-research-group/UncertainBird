# Analyzing Distribution Shift of Datasets

Understanding and quantifying distribution shift between datasets is essential for building robust deep learning systems. When the data distribution at training time differs from the one at deployment (or between different benchmarks), performance and calibration can degrade in subtle ways. This document outlines practical methods to detect, measure, and reason about distribution shift — with special attention to audio tasks.

## What do we mean by “distribution shift”?

Let Dataset A and Dataset B be drawn from joint distributions $p_A(x, y)$ and $p_B(x, y)$ over inputs $x$ and labels $y$. We use the usual marginals and conditionals:

- $p_A(x)$, $p_B(x)$: input (covariate) distributions in A and B.
- $p_A(y)$, $p_B(y)$: label prior distributions in A and B.
- $p_A(y\mid x)$, $p_B(y\mid x)$ and $p_A(x\mid y)$, $p_B(x\mid y)$: conditionals.

Here $p(\cdot)$ denotes a probability law (a probability mass function for discrete variables or a probability density for continuous variables). When convenient, we refer to A as the source and B as the target and write $S\equiv A$, $T\equiv B$; later equations that mention $S/T$ can be read as $A/B$.

Shift can arise via:

- Covariate shift: $p_A(x) \ne p_B(x)$ while $p(y\mid x)$ is (approximately) unchanged [1,2].
- Label shift (prior shift): $p_A(y) \ne p_B(y)$ while $p(x\mid y)$ is unchanged [3].
- Concept shift: $p(y\mid x)$ itself changes (e.g., new labeling rules, sensors, environments) [1].

In practice, multiple types co-occur. The goal is to characterize the differences and assess their impact on model performance and uncertainty.



## Methods to quantify shift

Below, $x$ denotes features/embeddings and $y$ labels (multi-label in many audio tasks).

### 1) Kernel two-sample tests (MMD)

The Maximum Mean Discrepancy (MMD) compares distributions via RKHS mean embeddings [4]:

$\operatorname{MMD}^2(\mathcal{D}_S, \mathcal{D}_T) = \mathbb{E}_{x,x'\sim S}[k(x,x')] + \mathbb{E}_{z,z'\sim T}[k(z,z')] - 2\, \mathbb{E}_{x\sim S, z\sim T}[k(x,z)]$

- Choose characteristic kernels (e.g., RBF with median heuristic). Larger MMD indicates greater shift.
- Use permutation/bootstrap to assess significance.

#### Sidebar: What is an RKHS?

An RKHS (Reproducing Kernel Hilbert Space) is a Hilbert space of functions associated with a positive‑definite kernel \(k\). It has two key properties:

- Reproducing property: for any \(f\) in the space and any \(x\), evaluation is an inner product
   \[
   f(x) = \langle f,\; k(\cdot, x) \rangle_{\mathcal{H}_k}.
   \]
- Feature map view: there exists (possibly infinite‑dimensional) \(\varphi(x)\) with
   \(k(x,x') = \langle \varphi(x), \varphi(x') \rangle\), enabling the kernel trick.

Kernel mean embeddings map a probability distribution \(p\) to its RKHS mean
\(\mu_p := \mathbb{E}_{x\sim p}[\,k(\cdot, x)\,] \in \mathcal{H}_k\). The MMD is the RKHS distance between embeddings:
\[
\operatorname{MMD}(p,q) = \|\mu_p - \mu_q\|_{\mathcal{H}_k}.
\]
With a characteristic kernel (e.g., Gaussian/RBF), this distance is a true metric on distributions, which is why MMD is a principled two‑sample test [4].

### 2) Optimal Transport / Wasserstein distance

The 1-Wasserstein (Earth Mover’s) distance measures the minimal mass-transport cost between empirical distributions [5,6]. On embeddings, it captures geometry more directly than KL-type divergences.

Practical tips:

- Use entropic-regularized OT (Sinkhorn) for efficiency on large samples.
- Report both the raw distance and a normalized variant (per-dimension) for comparability.

### 3) Fréchet distance (FID-style) and KID

Assuming Gaussian feature distributions, the Fréchet distance between means and covariances [7]:

$\operatorname{FID}(S,T) = \lVert\mu_S-\mu_T\rVert_2^2 + \operatorname{Tr}\left(\Sigma_S + \Sigma_T - 2\,(\Sigma_S\Sigma_T)^{1/2}\right)$

- Use FID on audio embeddings (AudioMAE/BirdMAE) as an “Audio FID”. KID is an unbiased alternative [7].

### 4) Energy distance

The energy distance is a metric with strong statistical foundations [8] and performs well as a nonparametric shift indicator over embeddings.

### 5) Domain discriminators (learnability of the shift)

Train a binary classifier to distinguish S vs T (on embeddings or spectrogram statistics). High AUC means the domains are separable; the AUC can be seen as a practical proxy for domain divergence [9].

Controls:

- Use the same train/validation protocol for the discriminator across comparisons.
- Monitor calibration of the discriminator to avoid conflation with class-imbalance.

### 6) Label/occurrence prior shift (label shift)

For multi-label audio, estimate per-class prevalences p(y_c=1). Compare S vs T using distances (L1/L2/EMD) on prevalence vectors, or use black-box shift estimation (BBSE) to correct predictions under label shift [3].

### 7) Calibration shift

Model calibration often drifts across domains [10]. Measure with Expected Calibration Error (ECE) on the target domain and compare to source:

$\operatorname{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left|\operatorname{acc}(b) - \operatorname{conf}(b)\right|$

Use reliability diagrams per-model and per-subset (the repo provides plotting utilities). Consider temperature or Platt scaling learned on a held-out calibration set.

### 8) Performance and class-wise analysis

Report absolute and relative deltas for key metrics (mAP, F1, AUROC) overall and per class. Pair with per-class sample counts to highlight whether classes with strong shift also dominate performance changes.

## Visualization toolkit

- UMAP/t-SNE on embeddings, colored by dataset and by top-1 prediction.
- Per-class frequency bar plots and cumulative coverage curves (e.g., “top classes covering 50% of samples”).
- Reliability diagrams to visualize calibration drift.
- Temporal histograms (time-of-day/season) and geospatial maps when metadata is available.

## Audio-specific considerations

- Sampling rate and resampling artifacts (16 kHz vs 32 kHz), windowing and hop size.
- Microphone/device frequency response; channel layout (mono vs stereo) and beamforming.
- Background noise types, SNR distributions, and acoustic environments (indoor/outdoor, habitat, reverberation).
- Segment duration and event density (5 s vs variable length segments alters context and label sparsity).
- Spectrogram parametrization (mel bins, f_max/f_min, log scaling); consistent normalization.
- Label noise and multilabel prevalence skew; long-tail species distributions; seasonal and geographic shifts (for bioacoustics/birds).
- Annotation policies (what counts as a positive) and ontology changes (merges/splits).

Practical tip: Always compare distributions at multiple levels — raw wave stats (RMS/SNR), spectrogram stats (mean/var per bin), and semantic embeddings — to isolate where the shift originates.


## A practical workflow (end-to-end)

1) Define the scope and subsets:

   - What splits or collections are being compared (e.g., Train vs Test_5s; Dataset A vs Dataset B)?
   - Ensure comparable preprocessing and segmenting.
1) Extract representations that capture semantics:

   - Use task-agnostic embeddings (e.g., AudioMAE/BirdMAE/CLAP) on both datasets.
   - Repository hook: `uncertainbird/scripts/dump_audiomae_embeddings.py` generates clip- and frame-level embeddings for BirdSet subsets.
1) Quantify distributional differences with complementary metrics:

   - Feature-level distances (MMD, Wasserstein/OT, Fréchet distance/FID-like, Energy distance).
   - Learnability of a domain discriminator (how easily a classifier separates S vs T).
   - Label/occurrence priors (class prevalence), and calibration drift (ECE/NLL).
1) Visualize and diagnose:

   - UMAP/t-SNE of embeddings colored by dataset/domain and by predicted class.
   - Per-class frequency plots and confusion deltas.
1) Report and act:

   - Summarize shift indicators; identify at-risk classes and conditions (e.g., SNR, devices).
   - Decide mitigations: reweighting, adaptation, augmentation, calibration, or targeted data collection.

## Putting it together in this repository

- Generate semantic embeddings with AudioMAE:
    - Script: `uncertainbird/scripts/dump_audiomae_embeddings.py` (saves clip and frame embeddings per subset/split).
- Quantify shift:
    - Compute MMD/FID/KID/Wasserstein on the saved clip embeddings between source and target subsets.
    - Compare per-class prevalences from `targets` artifacts (if available) and ECE via the provided plotting utilities.
- Visualize:
    - UMAP on clip embeddings with domain color; reliability diagrams per domain.

## Reporting and mitigation

When shift is significant:

- Reweighting: importance weighting for covariate shift (density ratio estimation) [2].
- Prior correction for label shift (BBSE) [3].
- Domain adaptation/regularization: CORAL/Deep CORAL [11], adversarial alignment, or fine-tuning on a subset of target data.
- Calibration: temperature or Platt scaling on target-like calibration data; evaluate post-calibration ECE and NLL.
- Data: targeted collection for underrepresented acoustic conditions or species; denoising/augmentation (SpecAugment, noise mixing) [12].

---

## References

[1] Quionero-Candela, Sugiyama, Schwaighofer, Lawrence. “Dataset Shift in Machine Learning.” MIT Press, 2009.

[2] Sugiyama et al. “Covariate Shift Adaptation by Importance Weighting.” JMLR, 2008.

[3] Lipton, Wang, Smola. “Detecting and Correcting for Label Shift with Black Box Predictors.” ICML, 2018.

[4] Gretton et al. “A Kernel Two-Sample Test.” JMLR, 2012.

[5] Villani. “Optimal Transport: Old and New.” Springer, 2009.

[6] Cuturi. “Sinkhorn Distances: Lightspeed Computation of Optimal Transport.” NIPS, 2013.

[7] Heusel et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NIPS, 2017. (FID); Binkowski et al., “Demystifying MMD GANs.” ICLR, 2018. (KID)

[8] Szekely and Rizzo. “Energy Statistics: A Class of Statistics Based on Distances.” J. Stat. Planning and Inference, 2013.

[9] Ben-David et al. “A Theory of Learning from Different Domains.” Machine Learning, 2010. (Domain divergence bounds)

[10] Guo et al. “On Calibration of Modern Neural Networks.” ICML, 2017.

[11] Sun and Saenko. “Deep CORAL: Correlation Alignment for Deep Domain Adaptation.” ECCV Workshops, 2016.

[12] Park et al. “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.” Interspeech, 2019.