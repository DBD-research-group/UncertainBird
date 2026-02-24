---
theme: ./theme
info: |
  Uncertainty Calibration of Multi-Label Bird Sound Classifiers — ICAART 2026
transition: slide-left
mdc: true
duration: 7min
drawings:
  persist: false

defaults:
  layout: default

themeConfig:
  paginationX: r
  paginationY: b
  paginationPagesDisabled: [1]

title: Uncertainty Calibration of Multi-Label Bird Sound Classifiers
venue: ICAART 2026
email: raphael.schwinger@cs.uni-kiel.de
presenter: Raphael Schwinger
bibFile: references.bib
---

<div class="flex flex-col h-full justify-center">

# Uncertainty Calibration of Multi-Label Bird Sound Classifiers

## ICAART 2026

**Raphael&nbsp;Schwinger**<sup>1</sup>, Ben&nbsp;McEwen<sup>2</sup>, Vincent&nbsp;S.&nbsp;Kather<sup>2,3</sup>, René&nbsp;Heinrich<sup>4,5</sup>, Lukas&nbsp;Rauch<sup>5</sup>, Sven&nbsp;Tomforde<sup>1</sup>

<div class="text-sm mt-4 space-y-0.5">
  <p class="m-0"><sup>1</sup>Kiel University · <sup>2</sup>Tilburg University · <sup>3</sup>Naturalis Biodiversity Center</p>
  <p class="m-0"><sup>4</sup>Fraunhofer IEE · <sup>5</sup>University of Kassel</p>
  <p class="m-0 mt-2"><a :href="`mailto:${$slidev.configs.email}`">{{ $slidev.configs.email }}</a></p>
</div>

</div>

---
layout: two-cols-header
---

# Motivation

::left::

- **Passive acoustic monitoring (PAM)** enables large-scale biodiversity assessment <Cite refKey="sugai_terrestrial_2019" />
- Reliable decisions need **well-calibrated uncertainty** — not only high accuracy <Cite refKey="guo2017calibration" />
- **Overconfidence** → biased occupancy, missed human review
- **Underconfidence** → scores less useful for prioritisation
- **Challenges**: multi-label (overlapping vocalisations), long-tailed species, distribution shift train vs. deployment <Cite refKey="stowell_computational_2022" />

::right::

<div class="flex flex-col items-center justify-center h-full">
  <figure>
    <img src="/specrogram_PER.svg" class="w-80 mx-auto" />
    <figcaption class="text-xs text-center mt-1 opacity-70">Spectrogram of a 5s snippet of the PER PAM dataset</figcaption>
  </figure>
</div>

---
layout: center
---

# Contributions

::content::

<ContributionBox>

<p><span class="c-label">C1.</span> <b>Benchmark calibration</b> of four state-of-the-art multi-label bird sound classifiers on BirdSet, reporting global, per-dataset and per-class calibration with threshold-free metrics, revealing that per-domain (e.g., dataset, class, etc.) calibration is highly variable.</p>

<p><span class="c-label">C2.</span> <b>Investigate post hoc strategies </b>(temperature and Platt scaling) showing that Platt scaling with 10 minutes of test data can significantly improve calibration.</p>

<p><span class="c-label">C3.</span> <b>Summarise implications </b> to give practitioners concrete suggestions for deployment-specific calibration evaluation and improvement. Code is available online.<sup>a</sup></p>

<template #footnote>
  <sup>a</sup> <a href="https://github.com/DBD-research-group/UncertainBird">https://github.com/DBD-research-group/UncertainBird</a>
</template>

</ContributionBox>

---
layout: figure-side
figureCaption: Example reliability diagram.
figureUrl: /figures/reliability-diagram.pdf
---

# Reliability Diagram

::content

- Predictions are binned by confidence; bar height = observed positive rate in that bin
- **Diagonal line** = perfect calibration (confidence matches reality)
- **Below diagonal** = overconfident (model predicts higher than reality)
- **Above diagonal** = underconfident (model predicts lower than reality)

---
layout: default
---

# Calibration Metrics: ECE and MCS

**ECE** (Expected Calibration Error) — magnitude of miscalibration (lower is better):

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|
$$

- Does not reveal *direction* (over- vs underconfident)

**MCS** (Miscalibration Score) <Cite refKey="aoTwoSidesMiscalibration2023" /> — signed version reveals direction:

$$
\text{MCS} = \sum_{m=1}^{M} \frac{|B_m|}{N} \big( \text{conf}(B_m) - \text{acc}(B_m) \big)
$$

- MCS = 0: perfectly calibrated · MCS &gt; 0: overconfident · MCS &lt; 0: underconfident  
- Caveat: over- and underconfident bins can cancel each other out

---
layout: default
---

# OCS and UCS: Decomposing Miscalibration

**OCS** (Overconfidence Score) — only positive deviations:

$$
\text{OCS} = \sum_{m=1}^{M} \frac{|B_m|}{N} \max\big( \text{conf}(B_m) - \text{acc}(B_m), 0 \big)
$$

**UCS** (Underconfidence Score) — only negative deviations:

$$
\text{UCS} = \sum_{m=1}^{M} \frac{|B_m|}{N} \big| \min\big( \text{conf}(B_m) - \text{acc}(B_m), 0 \big) \big|
$$

- Together they give a complete picture: a model can be overconfident in some bins and underconfident in others
- **ECE** = OCS + UCS · **MCS** = OCS − UCS

---
layout: two-cols-header
---

# Experimental Setup

::left::

**Models**: 4 SoTA multi-label bird sound classifiers

| **Model** | **Architecture** | **Params** |
|---|---|---|
| $AudioProtoPNet-20$ <Cite refKey="HEINRICH2025103081" />| ConvNeXt + prototypes | 287M |
| $BirdMAE$ <Cite refKey="rauchCanMaskedAutoencoders2025" />| ViT-B, SSL+SL | 92.9M |
| $ConvNeXt_{BS}$ <Cite refKey="rauchBirdSetLargeScaleDataset2025" /> | ConvNeXt, SL | 88M |
| $Perch\, v2$ <Cite refKey="merrienboerPerch20Bittern2025" /> | EfficientNet-B3, SSL+SL | 12M |

<style scoped>
th { border-bottom: 2px solid currentColor; }
</style>

::right::

**Dataset**:
- BirdSet benchmark <Cite refKey="rauchBirdSetLargeScaleDataset2025" />
- Train: XCL (~7,200 h, 9,734 classes)
- Val: POW (6.3 h)
- Test: PER, NES, UHH, HSN, NBP, SSW, SNE (7 PAM datasets, expert-annotated)
- 5s windows and 32 kHz


---
layout: figure-side
figureCaption: Reliability diagrams computed over all test segments.
figureUrl: /figures/reliability-combined.pdf
---

# Global Calibration Results

- $Perch\, v2$ and $ConvNeXt_{BS}$: best global calibration (MCS $\approx -1$), slightly **underconfident**
- $AudioProtoPNet$ and $BirdMAE$: **overconfident** (MCS 4.4, 7.0)
- **Best cmAP ≠ best calibration** — $AudioProtoPNet$ leads on discrimination but is poorly calibrated

---
layout: figure-side
figureCaption: $ConvNeXt_{BS}$ reliability per BirdSet dataset. Large variability across domains.
figureUrl: /figures/convnext_bs_reliability_diagram.pdf
---

# Per-Dataset Calibration

- **Strong variability** across datasets (e.g. $Perch\, v2$ ECE from 1.37 to 22.46)
- **Aggregated ECE** across all test data can **mask** domain-specific miscalibration
- $Perch\, v2$ & $ConvNeXt_{BS}$: underconfident on most sets; $AudioProtoPNet$ & $BirdMAE$ overconfident except on some (e.g. POW, UHH)

---
layout: figure-side
figureCaption: Class-wise ECE over number of samples per class.
figureUrl: /figures/ece-vs-sample-count.pdf
---

# Per-Class Calibration — ECE vs. Sample Count

- $Perch\, v2$ & $ConvNeXt_{BS}$: **low ECE** that increases **linearly** with sample count
- $AudioProtoPNet$ & $BirdMAE$: no clear structure
- For the **largest** classes, $AudioProtoPNet$ / $BirdMAE$ eventually reach lower ECE than $Perch\, v2$ / $ConvNeXt_{BS}$
- **Wide ECE spread** for underrepresented classes in $AudioProtoPNet$ / $BirdMAE$ explains their poor per-dataset scores — poorly calibrated classes can mask well-calibrated ones

---
layout: figure-side
figureCaption: Reliability diagrams for common (↑) and rare (↓) species subsets.
figureUrl: /figures/reliability-top-bottom.pdf
---

# Per-Class Calibration — Common vs. Rare Species

- Rare classes generally have less positives per probability bin.
- **All models show lower MCS for rare classes.**
- $ConvNeXt_{BS}$ has a particular strong improvement.

---
layout: default
---

# Calibration Methods

**Post hoc** — learn a mapping from logits to calibrated probabilities, no retraining:

- **Temperature Scaling (TS)** <Cite refKey="guo2017calibration" /> — single scalar $T$ rescales logits
- **Platt Scaling (PS)** <Cite refKey="platt1999probabilistic" /> — adds a bias $b$, corrects offset errors
- Isotonic regression <Cite refKey="niculescu2005predicting" />, vector scaling <Cite refKey="kull2019beyond" /> — more flexible variants

**Bayesian / ensemble** — principled but computationally expensive:

- MC Dropout <Cite refKey="gal2016dropout" />
- Deep Ensembles <Cite refKey="lakshminarayanan2017simple" />

**Ad hoc** — bake calibration into training (label smoothing <Cite refKey="muller2019does" />, focal loss <Cite refKey="lin2017focal" />, mixup <Cite refKey="zhangMixupEmpiricalRisk2018" />)


→ **We focus on TS and PS**: simple, parameter-efficient, no retraining required, and well-suited for limited calibration data <Cite refKey="guo2017calibration" /> <Cite refKey="platt1999probabilistic" />


---
layout: figure-side
figureCaption: Temperature scaling — effect of T on σ(z/T).
figureUrl: /figures/fig_a_temperature_scaling.pdf
---

# Temperature Scaling

$$\hat{p}^{\text{TS}}_c = \sigma(z_c / T), \quad T > 0$$

- $T = 1$: uncalibrated baseline
- $T > 1$: smoothes distribution → reduces overconfidence
- $T < 1$: sharpens distribution → reduces underconfidence
- Preserves monotonic ranking of scores when $T$ is global <Cite refKey="guo2017calibration" />
- **Limitation**: $\sigma(0) = 0.5$ → cannot correct systematic offset errors <Cite refKey="alexandari2020maximum" />

---
layout: figure-side
figureCaption: Platt scaling — temperature T and bias b. σ(z/T + b).
figureUrl: /figures/fig_c_platt_T_beta_joint.pdf
---

# Platt Scaling

$$\hat{p}^{\text{PS}}_c = \sigma(z_c / T + b), \quad T > 0,\ b \in \mathbb{R}$$

- **Bias $b$**: shifts the sigmoid left/right
  - $b > 0$: increases confidence (corrects underconfidence)
  - $b < 0$: decreases confidence (corrects overconfidence)
- Overcomes TS offset limitation <Cite refKey="alexandari2020maximum" />
- **Per-class** variant: fit $(T_c, b_c)$ per class <Cite refKey="frenkel2021network" />
- Parameters fitted by minimising NLL on held-out data <Cite refKey="lin2007note" /> <Cite refKey="guo2017calibration" />

---
layout: default
---

# Post Hoc Calibration: Experimental Setting

### Two calibration-data settings:

1. **Global** — fit single $(T)$ or $(T, b)$ on **POW** validation set
   - POW classes differ from test sets → only global params
2. **Per-class** — fit $(T_c)$ or $(T_c, b_c)$ on **first 10 min** of each test set
   - Enables class-level correction; test set slightly reduced for evaluation

Optimisation: Adam <Cite refKey="kingma2014adam" />, lr = 0.001, 1000 steps.

---
layout: two-cols-header
---

# Post Hoc Calibration: Results

::left::

| Model          | TS POW | PS POW | TS/cl. | PS/cl. |
|----------------|--------|--------|--------|--------|
| AudioProtoPNet | 56%    | 56%    | 75%    | **92%** |
| BirdMAE        | 81%    | 89%    | 94%    | **95%** |
| ConvNeXtBS     | 10%    | 20%    | **67%** | 12%   |
| Perch v2       | 33%    | 24%    | 23%    | **61%** |

Median % MCS improvement over base (all test sets).

::right::

- **Per-class PS** gives largest, most consistent gains (AudioProtoPNet ~92%, BirdMAE ~95%)
- Global TS/PS works well for BirdMAE but degrades on some datasets for AudioProtoPNet (e.g. UHH, HSN)
- ConvNeXtBS benefits most from **per-class TS**; Perch v2 from **per-class PS**
- No single method best — improvements are model- and domain-specific

---
layout: default
---

# Conclusions & Implications

1. **Calibration varies** across datasets and classes; aggregate metrics can hide systematic miscalibration.
2. **Perch v2 / ConvNeXtBS** consistently underconfident; **AudioProtoPNet / BirdMAE** mixed, with overconfidence at low probabilities.
3. **Simple post hoc** (Platt scaling, 10 min labelled data) can **strongly improve** deployment-specific calibration; effectiveness is model- and domain-dependent.

**For deployment**: evaluate per-dataset and per-class; report OCS/UCS; consider per-class or subgroup-aware calibration for rare species.

**Future**: Bayesian/distance-aware methods, calibration@k, more deployment scenarios.

---
layout: outro
---

::takeaways::

## Takeaways

- **Calibration varies across datasets/classes**; aggregated metrics hide domain-specific failures
- Perch v2 / ConvNeXt$_\text{BS}$ are underconfident; AudioProtoPNet / BirdMAE overconfident
- **Per-class Platt scaling (10 min of data) strongly improves calibration**

::right::

<div class="flex flex-col items-center gap-4">
  <QRCode text="https://github.com/DBD-research-group/UncertainBird" class="w-48" />
  <div class="text-sm text-center space-y-1">
    <p><b>Contact:</b> <a href="mailto:raphael.schwinger@cs.uni-kiel.de">raphael.schwinger@cs.uni-kiel.de</a></p>
    <p><b>Code:</b> <a href="https://github.com/DBD-research-group/UncertainBird" target="_blank" rel="noopener noreferrer">github.com/DBD-research-group/UncertainBird</a></p>
  </div>
</div>

::footer::

Funding: German Ministry for the Environment (DeepBirdDetect — 67KI31040C)
