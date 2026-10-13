# From the paper to the code

An equation-by-equation walkthrough of Section V, pointing at the exact
function that implements each step. Symbols follow the paper.

---

## 0. The feedback loop (Eqs. 1–3)

```
D_t        = Φ_extract(f̂, f, T_t)          apl/logic/evidence.py :: EvidenceExtractor.extract
G_t, y_t   = Ψ_match(D_t, S)                EvidenceExtractor.compatibility / .gaps
T_{t+1}    = θ_FG(T_t, α^gap(G_t))          apl/refinement/prompt_refiner.py :: refine_features
```

The loop is driven by `apl/refinement/loop.py :: run_refinement_loop` and
orchestrated by `APLModel.infer`.

## 1. Vision-language encoding (Sec. V-A)

| Paper | Code |
| --- | --- |
| `f̂ = E_v(I) ∈ R^d` | `CLIPBackbone.encode_image(...)[0]` |
| `f = {fⁿ}_{n=1..N}, fⁿ ∈ R^d` | `CLIPBackbone.encode_image(...)[1]` |
| frozen `E_t`, `E_v` | `EncoderConfig.freeze = True` → `requires_grad_(False)` |
| ViT-B/16, 12 layers, width 512 text, `d = 512`, `N = 196`, context 77, vocab 49,408 | `EncoderConfig` defaults |

Visual prompt tokens are concatenated to the patch sequence *after* the
positional embedding and before `ln_pre`, i.e. at the encoder input; the local
features returned exclude them, so `f` always has exactly `N` rows.

## 2. Base prompt and text feature (Sec. V-B1)

```
p₀ = "Describe the traffic scene including vehicles, signals, road markings"
w  = E_t(p₀) ∈ R^{C×d}
```

`apl/modules/prompt_learner.py`. One row per semantic category: the template
`PromptConfig.class_template` renders `p₀` together with the category name, the
batch of `C` prompts is tokenised once and cached, and the four learnable text
tokens are inserted right after the SOT token on every forward pass.
`zero_shot_features()` returns `w^zs` from the unmodified encoder (needed by the
GI regulariser) and is cached.

## 3. Feature generator θ_FG (Sec. V-B2)

```
Q    = LN(w)·θ_in1,  K = LN(α)·θ_in2,  V = LN(α)·θ_in3
Attn = w + (softmax(QKᵀ/√d_k)·V)·θ_out
h    = Attn + ReLU(LN(Attn)·θ_fc1)·θ_fc2        h ∈ R^{C×d}
```

`apl/modules/feature_generator.py`. `d = d_k = 512`, `d_mid = 256`, one head.
Both residual additions are kept. `θ_out` and `θ_fc2` are zero-initialised, so
`θ_FG(w, α) = w` before training: the generator starts as the identity and
learns how far to move each row. The module broadcasts `w` and `α` over a batch
dimension, which is what lets the same instance serve the image-conditioned
branch (`α = f̂_b`), the learnable banks (`α^GI`, `α^FG`) and the refinement
step (`α^gap`).

## 4. The three branches

### 4.1 Instance-dependent (Sec. V-B3)

```
α^ID_b = f̂_b ,  h^ID_b = θ_FG(w, α^ID_b) ∈ R^{C×d}
p_b(m|I_b) = exp(cos(f̂_b, h^ID_{b,m})/τ) / Σ_j exp(cos(f̂_b, h^ID_{b,j})/τ)
L^ID_CE    = (1/B) Σ_b ( −Σ_m y_{b,m} log p_b(m|I_b) )
```

`InstanceDependentBranch` + `losses.cross_entropy`, `τ = 0.07`.

### 4.2 Fine-grained, evidence-gap conditioned (Eq. 4, Sec. V-B4)

```
α^FG ∈ R^{Z×d},  Z = 4,  α^FG ~ N(0, 0.02²)
h^FG_z = θ_FG(w, α^FG_z) ∈ R^{C×d}
sim(h^FG_{z,m}, f) = (1/zη) Σ_n 1_top-zη(n) · cos(h^FG_{z,m}, fⁿ)
L^FG_CE = (1/Z) Σ_z ( −Σ_m y_m log p^FG_z(m|I) )
```

`FineGrainedBranch`. The `top_k` ladder is `k_z = ⌈z/Z · η · N⌉` by default and
`⌈zη⌉` with `fg_topk_mode = "count"` (see README, deviation 2). The gradient of
the APL loss w.r.t. `α^FG` comes only from `L^FG_CE`, as stated in the paper:
the other two branch losses do not depend on `α^FG`.

### 4.3 Rule-conditioned, global-invariant (Sec. V-B5)

```
L^GI_CE = L_CE + λ₁‖w^GI − w^zs‖ + λ₂‖f̂ − f̂^zs‖       λ₁ = λ₂ = 0.1
```

`GlobalInvariantBranch` derives `w^GI = θ_FG(w, α^GI)` from a learnable
image-independent bank; `losses.global_invariant_loss` adds the two anchors to
the zero-shot features, which is what prevents the branch from drifting toward
privacy-sensitive specifics such as appearance or plate detail.

### 4.4 Joint objective and inference (Sec. V-B6, Sec. V-H)

```
L^APL = λ_GI L^GI_CE + λ_ID L^ID_CE + λ_FG L^FG_CE      λ_GI = λ_ID = λ_FG = 1
p(m|I) = ⅓ [ p^GI(m|I) + p^ID(m|I) + (1/Z) Σ_z p^FG_z(m|I) ]
```

`losses.apl_loss`, `losses.ensemble_probability`. With a branch disabled, both
reduce to the mean over the active branches, which is what the ablation
variants need.

## 5. Structured evidence extraction (Sec. V-D)

```
T = {w^GI, h^ID, h^FG_1, …, h^FG_Z}
z_i = cos(f̂, t_i),  t_i ∈ rows(T)
D_t = Φ(f̂, f, T_t)
```

`apl/logic/evidence.py`. Each branch contributes a `[B, C]` score block; the
blocks are averaged into `z ∈ [B, P]`, where the `P` columns are the scenario
bank's predicates because the rows of `w` *are* those predicates. Fine-grained
rows use the top-`k` local pooling by default (`logic.fg_local_evidence`).

## 6. Soft-logic interface and scenario matching (Sec. V-E)

```
ψ(z_jq) = 1 / (1 + exp(−κ(z_jq − τ_q)))        τ_q init 0.25, κ = 10
γ_j^(t) = Σ_q ω_jq ψ(z_jq),  ω_jq = 1/Q_j,  Σ_q ω_jq = 1 ⇒ γ_j ∈ [0,1]
```

`apl/logic/soft_logic.py`. `τ_q` is a learnable per-predicate parameter;
`SoftLogicGate.calibrate` fits it with predicate-level binary cross entropy on
held-out calibration data and it is frozen at inference, exactly as Table I
prescribes. `γ_j` summarises evidence *compatibility* — it is deliberately not
the joint probability that all required predicates hold.

## 7. Iterative prompt refinement (Sec. V-F)

```
G_{j,t} = required predicates absent or unresolved in D_t
G_t     = ∪_{j=1..M} {(j, e) : e ∈ G_{j,t}}
p_{t+1} = R(D_t, G_t, S)
T_{t+1} = θ_FG(T_t, α^gap(G_t))
stop when G_t = ∅, t = T_max = 5, or Δγ^(t) = max_j |γ_j^(t) − γ_j^(t−1)| < ε = 1e-3
```

`apl/refinement/`. Missing-evidence checks run against **every** scenario in
`S`, not only those whose initial score crosses a threshold, so an initial
description that identifies no violation can still trigger targeted extraction.
`t★` is the iteration at which the loop actually stops and is reported as
`InferenceOutput.iterations` together with `stop_reason`.

## 8. Rule-based decision (Sec. V-G)

```
y_j = V  if γ_j^(t★) ≥ Γ_strict = 0.80
      U  if Γ_loose = 0.40 ≤ γ_j^(t★) < Γ_strict
      N  if γ_j^(t★) < Γ_loose
V(I) = { ν_j : y_j = V }
```

`apl/logic/rule_engine.py`. One decision per scenario — the engine does not
stop at the first positive violation, so `V(I)` may contain several violation
types. `U` is a first-class outcome: a low score is not proof that an
unobserved predicate is false. Each `ScenarioDecision` also carries the
per-predicate satisfaction values, the list of still-missing predicates, the
Boolean verdict of the deterministic rule `r_j` and the legal reference, which
is what makes the output auditable.

## 9. Algorithm 1 (Sec. V-I)

`run_refinement_loop` implements lines 1–24 and `RuleEngine.evaluate` lines
25–28. The ordering of the stopping tests follows the pseudo-code: empty gap
set or `t = T_max` first, then the `Δγ < ε` convergence test from `t ≥ 1`.

## 10. Evaluation (Sec. VI)

| Metric | Code |
| --- | --- |
| Privacy Score, Eq. (6) | `eval/privacy.py :: privacy_score` |
| Semantic Utility, Eq. (7) | `eval/utility.py :: semantic_utility` |
| Avg. Risk, Eq. (5) | `eval/privacy.py :: membership_inference_attack` |
| SSIM / PSNR / MSE / SRRA | `eval/privacy.py` (pixels in `[0,255]`, averaged per image then over images) |
| BLEU / ROUGE-L / CIDEr-D | `eval/text_quality.py` |
| Flow RMSE, sign accuracy, pedestrian MAE | `eval/utility.py :: downstream_report` |
| Base/New/H | `train/fewshot_eval.py` |
| Table IX variants | `eval/ablation.py` |

Semantic Utility is measured on the *affirmative* part of the description
(`SceneDescription.positive_text`), so that listing a predicate under "not
observed" can never be counted as retaining its evidence. Privacy Score is
measured on the full released text.
