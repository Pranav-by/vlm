# Technical Report: Membership Inference Attacks Against Vision-Language Models

**Paper:** Membership Inference Attacks Against Vision-Language Models  
**Implementation Date:** 2026-05-08  
**Environment:** CPU-only Ubuntu (synthetic data pipeline)

---

## 1. Problem Statement

Vision-Language Models (VLMs) like LLaVA and MiniGPT-4 are trained on curated image-text instruction datasets. The central question this paper asks:

> **Given a trained VLM and a specific image-text pair, can we determine whether that pair was part of the model's training data?**

If yes, this constitutes a **membership leakage** — a privacy vulnerability. The attacker treats the VLM as a black box (no access to weights or gradients) and can only query it with images and questions.

---

## 2. Why VLMs Are Vulnerable

### 2.1 The Memorization Hypothesis

During **Visual Instruction Tuning** (the phase this paper targets), VLMs are fine-tuned on small, curated datasets for only 1–3 epochs. This creates a specific vulnerability:

- **Few training iterations** → The model doesn't generalize fully; it partially memorizes specific training examples.
- **Curated data** → Unlike pre-training on billions of web-scraped samples, instruction tuning uses thousands of carefully crafted Q&A pairs, making each sample more "memorable."

### 2.2 Observable Signal: Similarity Gap

When a trained VLM is queried:

```
Member data (seen during training):
  Question: "What is in this image?"
  Model output: "A cat is sitting on a windowsill looking outside at the birds."
  Ground truth: "A cat is sitting on a windowsill looking outside at the birds."
  → Similarity: HIGH (model "remembers" the answer)

Non-member data (never seen during training):
  Question: "What is in this image?"
  Model output: "There appears to be an animal near a window."
  Ground truth: "A cat is sitting on a windowsill looking outside at the birds."
  → Similarity: LOW (model is guessing)
```

This **similarity gap** is the signal all 5 attacks exploit.

### 2.3 Temperature's Role

Temperature controls randomness in text generation:
- **Low temperature (0.1):** Model outputs its most confident tokens → Member data gets very high similarity (nearly reciting the training answer). Non-member data gets moderate similarity.
- **High temperature (1.5):** Model outputs more random tokens → Both member and non-member similarity drops, but **members drop more** because there's more "memorized" content to lose.

This temperature sensitivity is exploited by the Target-Only attack.

---

## 3. The Five Attack Algorithms — How They Work

### 3.1 Attack 1: Shadow Model Attack (AUC = 1.00 in our run)

```
Architecture: 3-layer Binary Neural Network
Input: 32 features (16 temperature means + 16 temperature variances)
Output: probability(member)
```

**How it works:**

1. **Split data into 4 groups:**
   - Shadow-member (400 samples) — known training data for shadow model
   - Shadow-non-member (100 samples) — known withheld data for shadow model
   - Target-member (400 samples) — the actual training data we want to detect
   - Target-non-member (100 samples) — the actual withheld data

2. **Train a shadow VLM** on shadow-member data (mimics the target VLM).

3. **Query the shadow VLM** at 16 different temperatures (0.01 to 1.8) and compute similarity between VLM output and ground truth.

4. **Create feature vectors:** For each group of `granularity` samples (15 in our case):
   - Compute the **mean** similarity at each temperature → 16 features
   - Compute the **variance** of similarity at each temperature → 16 features
   - Label: 1 (member) or 0 (non-member)
   - Total: 32-dimensional feature vector + label

5. **Train a binary classifier** (3-layer NN: Linear(32→64)→ReLU→Linear(64→64)→ReLU→Linear(64→1)→Sigmoid) on shadow data.

6. **Test on target data** — classify whether target samples are members or not.

**Why it's the strongest attack:**
- Uses 32 features per sample group (most information of any attack)
- The NN can learn complex non-linear decision boundaries
- Has access to a labeled training set (shadow model provides ground truth)

**Mathematical formulation:**
```
For a group of G samples at temperature t:
  μ_t = (1/G) Σ sim(VLM(x_i, t), y_i)    [mean similarity]
  σ²_t = (1/G) Σ (sim(x_i, t) - μ_t)²     [variance]

Feature vector: [μ_0.01, μ_0.05, ..., μ_1.8, σ²_0.01, ..., σ²_1.8]
Label: 1 if member, 0 if non-member

Classifier: f(features) → P(member) ∈ [0, 1]
Loss: BCE = -[y·log(f(x)) + (1-y)·log(1-f(x))]
```

---

### 3.2 Attack 2: Reference Member Inference (AUC = 0.9945)

**How it works:**

Uses a statistical **Z-test** to compare distributions. The key assumption: if a target group's similarity is as high as a known member reference group, it's likely member data.

1. **Split member data in half:**
   - Reference-member (half) — used as the reference baseline
   - Target-member (other half) — what we're testing

2. **For 1000 iterations:**
   - Sample `granularity` (50) data points from each group
   - Compute the Z-statistic between target and reference

3. **Z-test formulation:**
```
H₀: Target group has the same mean similarity as the reference member group
H₁: Target group has different (lower) mean similarity

Z_member = (μ_target_member - μ_reference_member) / √(σ²_target/n + σ²_ref/n)
p_member = 1 - Φ(Z_member)    [one-tailed test]

Z_non_member = (μ_reference_member - μ_target_non_member) / √(σ²_ref/n + σ²_non/n)
p_non_member = 1 - Φ(Z_non_member)
```

4. **Decision logic:**
   - Low p-value for member test → target looks like reference members → MEMBER
   - Low p-value for non-member test → target looks different from reference → NON-MEMBER
   - AUC is computed over all (p-value, label) pairs

**Why it works well (AUC=0.99):**
- With granularity=50, the Central Limit Theorem kicks in — sample means have much smaller variance than individual scores
- Even though individual member/non-member scores overlap (std=0.14), the mean of 50 samples has std ≈ 0.14/√50 ≈ 0.02
- The 0.10 gap in means is now ~5 standard errors apart → statistically significant

---

### 3.3 Attack 3: Reference Non-Member Inference (AUC = 0.9993)

**Same as Attack 2, but uses non-member reference instead:**

1. Split non-member data in half:
   - Reference-non-member → baseline
   - Target-non-member → what we test

2. Compare target groups against the non-member reference:
```
Z_member = (μ_target_member - μ_reference_non_member) / √(σ²_target/n + σ²_ref/n)
    → Large positive Z → target has much higher similarity than non-members → MEMBER

Z_non_member = (μ_target_non_member - μ_reference_non_member) / √(σ²_target/n + σ²_ref/n)
    → Z near zero → target looks like non-members → NON-MEMBER
```

**Why slightly better than Attack 2 (0.9993 vs 0.9945):**
- The reference non-member group provides a cleaner baseline — members are clearly above it
- In Attack 2, splitting members in half means the reference and target come from the same distribution, making the Z-test compare "member vs member" which gives weaker signal

---

### 3.4 Attack 4: Target-Only Inference (AUC = 0.5262)

**The hardest attack — requires NO reference data at all.**

**Hypothesis:** Member data shows higher **temperature sensitivity** than non-member data.

```
For member data:
  sim(member, temp=0.1) ≈ 0.32  [high: model recites memorized answer]
  sim(member, temp=1.5) ≈ 0.23  [lower: randomness disrupts memory]
  Δ = 0.09  (significant drop)

For non-member data:
  sim(non-member, temp=0.1) ≈ 0.22  [moderate: model guesses]
  sim(non-member, temp=1.5) ≈ 0.16  [slightly lower: more randomness]
  Δ = 0.06  (smaller drop)
```

**The Z-test compares low-temp vs high-temp similarity WITHIN the same group:**
```
Z = (μ_low - μ_high) / √(σ²_low/n + σ²_high/n)
p = P(Z > observed | H₀: no temperature effect)

Members → large Z → small p-value (significant temperature effect)
Non-members → smaller Z → larger p-value
```

**Why it performs poorly (AUC=0.53):**
- The temperature sensitivity difference between members and non-members is very subtle
- Both groups show some temperature effect (similarity always decreases with temperature)
- The attack relies on a second-order effect (difference of differences), which has much higher variance
- With only 2 temperature points, there's minimal signal

**In the paper's real experiments:**
- Target-Only achieves AUC ~0.55-0.75
- It works better with more extreme temperature differences and when the model strongly memorizes

---

### 3.5 Attack 5: Image-Only Inference (AUC = 0.9408)

**Requires NO ground truth text — only the image.**

**Hypothesis:** When asked to describe the same image multiple times, a VLM will produce:
- **More consistent responses** for member images (it "knows" what to say — memorized the training answer)
- **More varied responses** for non-member images (it's improvising each time)

**How it works:**

1. Query the model with the **same image** 5 times at temperature=0.1
2. Compute **pairwise similarity** between all response pairs:
```
For 5 responses, there are C(5,2) = 10 pairs:
  avg_sim = (1/10) Σ sim(response_i, response_j)
```

3. Member images → high avg_sim (consistent responses)
4. Non-member images → lower avg_sim (varied responses)

**Decision:** Simply threshold the average pairwise similarity.
```
AUC = ROC_AUC(labels=[1,0,1,0,...], scores=[avg_sim_member, avg_sim_non_member, ...])
```

**Why it works well (AUC=0.94):**
- Consistency is a strong signal of memorization
- Multiple repetitions amplify the effect (5 queries × 10 pairs)
- Doesn't need ground truth at all — purely behavioral

**Why it's not perfect:**
- Some non-member images may get consistent responses if they're "easy" (e.g., simple scenes)
- Some member images may get varied responses at higher temperatures

---

## 4. Similarity Metrics Explained

The pipeline uses two metrics to quantify "how close is the VLM's response to the truth":

### 4.1 Rouge-2 F-Score (Text Overlap)

Measures bigram (2-word) overlap between two texts:
```
Precision = |bigrams(VLM) ∩ bigrams(truth)| / |bigrams(VLM)|
Recall    = |bigrams(VLM) ∩ bigrams(truth)| / |bigrams(truth)|
F-score   = 2 × (Precision × Recall) / (Precision + Recall)

Example:
  VLM:   "A cat sitting on a windowsill"     → bigrams: {A cat, cat sitting, sitting on, on a, a windowsill}
  Truth: "A cat is sitting on a windowsill"   → bigrams: {A cat, cat is, is sitting, sitting on, on a, a windowsill}
  Shared bigrams: {A cat, sitting on, on a, a windowsill} = 4
  Precision = 4/5 = 0.80
  Recall = 4/6 = 0.67
  F = 2×0.80×0.67/(0.80+0.67) = 0.73
```

### 4.2 MPNet Embedding Cosine Similarity (Semantic Similarity)

Maps text to a 768-dimensional vector using the `all-mpnet-base-v2` sentence transformer, then computes cosine similarity:
```
sim = cos(θ) = (embed(VLM) · embed(truth)) / (||embed(VLM)|| × ||embed(truth)||)

Range: -1 to 1 (typically 0.3 to 0.95 for related texts)
```

This captures **semantic** similarity even when exact words differ:
```
VLM:   "A feline rests on the window ledge watching birds"
Truth: "A cat is sitting on a windowsill looking outside at the birds"
Rouge-2: low (few shared bigrams)
MPNet: high (semantically similar)
```

---

## 5. Our Experimental Setup

### 5.1 Data Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total samples | 1000 | Small enough for CPU, large enough for statistical tests |
| Member/Non-member split | 80/20 (800/200) | Matches paper's recommendation |
| Shadow split | 400/100/400/100 | Even split of member and non-member data |
| Temperatures (Shadow) | 16 values: 0.01→1.8 | Full range to capture temperature effects |
| Temperatures (Reference) | 0.1 | Single low temperature for stable output |
| Temperatures (Target-Only) | 0.1, 1.5 | Two extremes to maximize sensitivity gap |
| Repeat (Image-Only) | 5 | 5 queries per image → 10 pairwise comparisons |
| Granularity | 15 (shadow), 50 (others) | Group size for statistical tests |

### 5.2 Similarity Score Distributions

Our synthetic scores model the paper's findings:

```
                Member Distribution        Non-Member Distribution
                ──────────────────         ───────────────────────
Rouge-2 (t=0.1):
  Mean:         0.3213                     0.2173
  Std:          0.1418                     0.1405
  Range:        [0.01, 0.84]              [0.01, 0.57]
  
  │     ╭─╮
  │    ╱   ╲         ╭─╮
  │   ╱     ╲       ╱   ╲
  │  ╱       ╲     ╱     ╲
  │ ╱         ╲   ╱       ╲
  │╱           ╲─╱         ╲
  └──────────────────────────────→ similarity
  0    0.1   0.2   0.3   0.4   0.5
       non-member  ↕overlap↕  member
```

The **overlap region** (roughly 0.1–0.4) is where individual samples are ambiguous — this is why attacks aren't perfect and produce AUC < 1.0.

---

## 6. Results Analysis

### 6.1 Final Results

| Attack | AUC/Accuracy | Information Required | Difficulty |
|--------|-------------|---------------------|------------|
| Shadow Model | **1.0000** | Shadow model + 16 temperatures + labels | Easiest (most info) |
| Reference Non-Member | **0.9993** | Non-member reference set + 1 temperature | Easy |
| Reference Member | **0.9945** | Member reference set + 1 temperature | Easy |
| Image-Only | **0.9408** | Repeated queries only (no ground truth) | Moderate |
| Target-Only | **0.5262** | 2 temperatures only (no reference) | Hardest (least info) |

### 6.2 Key Observations

**1. More information → better attack:**
The attacks form a clear hierarchy based on how much auxiliary information the attacker has:
```
Shadow (32 features + labeled data) > Reference (statistical test + reference set) 
    > Image-Only (behavioral test) > Target-Only (minimal information)
```

**2. Group-level testing overcomes individual overlap:**
Even though individual member and non-member scores overlap heavily (std ≈ 0.14 vs gap ≈ 0.10), the Reference attacks achieve AUC ~0.99 because they test groups of 50 samples. By the Central Limit Theorem:
```
σ_group_mean = σ_individual / √n = 0.14 / √50 ≈ 0.020
Gap / σ_group = 0.10 / 0.020 = 5.0 standard deviations
→ P(overlap) ≈ 0.0000003 → extremely unlikely → high AUC
```

**3. Target-Only is near random (0.53):**
This attack has the least information — no reference data, no labels, only the temperature sensitivity of the target group itself. The signal is a second-order effect (member temperature sensitivity minus non-member temperature sensitivity), which is drowned out by noise at the individual level.

**4. Shadow Model achieves perfect classification:**
With 32 features (16 means + 16 variances across temperatures), the NN has enough dimensionality to perfectly separate even overlapping distributions. The curse of dimensionality works in the attacker's favor here — with more features, the separating hyperplane becomes easier to find.

---

## 7. Threat Model Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     ATTACKER'S KNOWLEDGE                     │
├─────────────┬───────────────────────────────────────────────┤
│ Has access  │ • Black-box query access to the trained VLM   │
│   to:       │ • Some candidate image-text pairs              │
│             │ • (Optionally) a reference dataset              │
├─────────────┼───────────────────────────────────────────────┤
│ Does NOT    │ • Model weights or architecture details        │
│ have access │ • Training procedure or hyperparameters         │
│   to:       │ • Gradients or loss values                      │
│             │ • Direct access to the training dataset         │
├─────────────┼───────────────────────────────────────────────┤
│ Goal        │ Determine if a specific (image, text) pair     │
│             │ was in the VLM's training set                   │
├─────────────┼───────────────────────────────────────────────┤
│ Success     │ AUC > 0.5 means the attack works               │
│ metric      │ (can distinguish members from non-members)     │
└─────────────┴───────────────────────────────────────────────┘
```

---

## 8. Code Architecture

```
vlm_mia/
│
├── Phase 1-3: VLM Training & Querying (REQUIRES GPU)
│   ├── LLaVA/
│   │   ├── conversation_llava.py        ← Query trained LLaVA model
│   │   ├── llava/train/train_mem.py     ← Training script
│   │   └── scripts/finetune.sh          ← Finetune shell script
│   └── MiniGPT-4/
│       ├── conversation_minigpt4.py     ← Query trained MiniGPT-4
│       ├── train.py                     ← Training script
│       └── train_configs/               ← Training YAML configs
│
├── Phase 4: Similarity Calculation (CPU OK)
│   ├── similarity_with_ground_truth.py       ← Original (needs OpenAI API)
│   ├── similarity_with_repeating_generation.py ← Original (needs OpenAI API)
│   └── scripts/02_compute_similarity.py      ← Modified (local Rouge + MPNet only)
│
├── Phase 5: Inference Attacks (CPU OK)
│   ├── shadow_model_inference.py        ← Attack 1: NN classifier
│   ├── reference_member_inference.py    ← Attack 2: Z-test (member ref)
│   ├── reference_non_member_inference.py ← Attack 3: Z-test (non-member ref)
│   ├── target_only_inference.py         ← Attack 4: Z-test (temp sensitivity)
│   └── image_only_inference.py          ← Attack 5: Pairwise similarity
│
├── Our Scripts (CPU synthetic pipeline)
│   ├── scripts/01_generate_synthetic_data.py
│   ├── scripts/02_generate_similarity_scores.py
│   └── scripts/run_fast_pipeline.sh
│
└── Generated Data
    └── data/
        ├── *.json                       ← Instruction data + splits
        ├── conversation_*.json          ← Simulated VLM conversations
        └── similarity/*.json            ← Computed similarity scores
```

---

## 9. Limitations of Our Synthetic Approach

| Limitation | Impact | What Real Data Would Show |
|-----------|--------|--------------------------|
| No actual VLM training | Similarity distributions are approximated, not real | Real distributions would have more complex, non-Gaussian shapes |
| No real images | Image features don't influence synthetic scores | Some images are inherently easier → creates multi-modal distribution |
| Fixed noise model | We use Gaussian noise with constant std | Real noise varies per sample (some answers are longer/shorter) |
| Shadow Model AUC=1.0 | Too high — real experiments get 0.70-0.90 | With more overlap in real data, the NN can't perfectly separate |
| No embedding computation | We skip the actual MPNet/Rouge calculation | Real Rouge/MPNet scores have more nuanced relationships |

---

## 10. Privacy Implications

This research demonstrates that:

1. **VLMs leak training data membership** even through black-box access only
2. **Visual Instruction Tuning is the most vulnerable phase** due to small dataset + few epochs
3. **Multiple attack strategies exist** with varying levels of attacker knowledge required
4. **Group-level attacks are very effective** — even with overlapping individual distributions, testing groups of 50+ samples makes the signal statistically significant
5. **Temperature is a privacy-revealing parameter** — querying at different temperatures exposes memorization patterns

**Defensive implications:**
- Differential privacy during instruction tuning could mitigate these attacks
- Limiting temperature range in production APIs reduces temperature-based attacks
- Rate-limiting repeated identical queries defends against Image-Only attacks

---

*Report stored at: ~/vlm_mia/TECHNICAL_REPORT.md*
