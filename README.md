# APL — Adaptive Prompt Learning for Privacy-Preserving via Vision-Language Models

Reference implementation of

> **Adaptive Prompt Learning for Privacy-Preserving via Vision-Language Models**
> Abdolazim Rezaei, Mahboobeh Haghparast, Ahmad Patooghy, Mehdi Sookhak
> *IEEE Transactions on Big Data*

APL replaces the raw camera frame in an Intelligent Transportation System (ITS)
with a **structured textual description** that keeps the evidence a traffic rule
needs and drops the information that identifies people. Instead of a fixed
library of expert prompts, a lightweight **Surrogate Feature Generator**
synthesises three complementary families of prompted text features from a
single base prompt, and a **gap-conditioned feedback loop** keeps refining the
prompts until every evidentiary predicate required by a DMV-aligned scenario
bank is either confirmed or explicitly reported as unresolved. A deterministic
**rule engine** then issues one decision per scenario.

```
      image ──► frozen CLIP ViT-B/16 ──► f̂ (global), f = {fⁿ} (local)
                                            │
             base prompt p₀ ──► w ──► θ_FG ─┼─► w^GI   global-invariant
                                            ├─► h^ID   instance-dependent
                                            └─► h^FG_z fine-grained (Z scales)
                                            │
                       Φ_extract ──► D_t ──► ψ(z_jq) ──► γ_j ──► rule engine
                                            │                     │
                              gaps G_t ◄────┘            V / U / N per scenario
                                 │
                 R(D_t,G_t,S) ──► prompts ──► α^gap ──► T_{t+1}  (Algorithm 1)
```

Everything in the paper's Section V is implemented: the frozen vision-language
encoding, the cross-attention feature generator, the three branches and their
losses, the structured evidence extraction, the differentiable soft-logic
interface, the scenario bank, the iterative prompt refinement of Algorithm 1,
the rule-based violation decision, and the full evaluation suite (few-shot and
base-to-new accuracy, Privacy Score, Semantic Utility, reconstruction and
replay attacks, membership inference, BLEU/ROUGE/CIDEr, downstream ITS tasks,
and the component ablation).

---

## Installation

```bash
git clone https://github.com/azim015/APL.git
cd APL
pip install -r requirements.txt     # torch, numpy, pyyaml, pillow
```

Optional, for the real CLIP weights:

```bash
pip install open_clip_torch            # or: pip install git+https://github.com/openai/CLIP.git
```

**Backbone resolution.** `encoder.provider: auto` tries the OpenAI `clip`
package, then `open_clip`, then falls back to a bundled CLIP-geometry
re-implementation (`apl/encoders/fallback_clip.py`) with randomly initialised
weights. The fallback keeps the ViT-B/16 structure (12 layers, width 768, 512-d
output, 196 patches, context length 77) so the whole pipeline — training,
Algorithm 1, every metric — runs offline. Accuracy numbers only become
meaningful with real pretrained weights; the fallback exists so the code is
runnable and testable anywhere.

## Quickstart

```bash
python scripts/demo.py --config configs/smoke.yaml
```

This trains the prompt learner on synthetic traffic scenes, runs the full
feedback loop on a handful of them, prints the generated descriptions, the
targeted refinement prompts, the per-scenario decisions with their legal
references, and the Privacy Score / Semantic Utility / BLEU / ROUGE / CIDEr
summary. It takes well under a minute on CPU and downloads nothing.

Using the model directly:

```python
from apl import APLConfig, APLModel

model = APLModel(APLConfig())          # rows of w = the scenario-bank predicates
out = model.infer(images[:1])          # Algorithm 1 + rule engine

print(out.description.text)            # privacy-preserving scene description
print(out.engine.report())             # one V / U / N decision per scenario
print(out.iterations, out.stop_reason) # t★ and why the loop stopped
```

## Repository layout

```
apl/
  config.py                 every hyper-parameter of Table I, YAML + CLI overrides
  model.py                  APLModel: the five modules wired end to end
  describe.py               description composition + the PII redaction filter
  cli.py                    shared argument parsing / dataset construction
  encoders/
    clip_backbone.py        frozen CLIP wrapper: global + local features, prompt insertion
    fallback_clip.py        CLIP-geometry ViT + text transformer for offline runs
    tokenizer.py            CLIP BPE when available, deterministic fallback otherwise
  modules/
    prompt_learner.py       base prompt p₀, w = E_t(p₀), learnable text/visual tokens
    feature_generator.py    θ_FG cross-attention block (Sec. V-B2)
    branches.py             GI / ID / FG branches and the top-k local matching
    losses.py               L^GI, L^ID, L^FG, L^APL and the 3-branch ensemble
  logic/
    scenario_bank.py        s_j = (ν_j, E_j, r_j); the M = 12 DMV scenarios
    soft_logic.py           ψ(z) calibrated gate, γ_j, gap masks, τ_q calibration
    evidence.py             Φ_extract: T = {w^GI, h^ID, h^FG_z} → D_t
    rule_engine.py          Γ_loose / Γ_strict decision, per-scenario reports
  refinement/
    prompt_refiner.py       R(D_t, G_t, S) and α^gap(G_t)
    loop.py                 Algorithm 1
  data/                     few-shot folders, annotated traffic manifests, synthetic data
  train/                    SGD trainer, few-shot and base-to-new protocols
  eval/                     privacy, utility, text quality, ablation
configs/                    apl_base (Table I), fewshot_16shot, its_inference, smoke
scripts/                    demo, train_fewshot, eval_base_to_new, run_inference,
                            eval_privacy, eval_downstream, run_ablation
tests/                      90 unit tests covering every component
data/scenarios/             the DMV-aligned scenario bank (JSON)
docs/METHOD.md              equation-by-equation map from the paper to the code
```

## What maps to what

| Paper | Code |
| --- | --- |
| `f̂ = E_v(I)`, `f = {fⁿ}` (Sec. V-A) | `CLIPBackbone.encode_image` |
| `w = E_t(p₀) ∈ R^{C×d}` (Sec. V-B1) | `PromptLearner.forward` |
| `h = θ_FG(w, α)` (Sec. V-B2) | `FeatureGenerator.forward` |
| `h^ID_b = θ_FG(w, f̂_b)`, `L^ID_CE` | `InstanceDependentBranch`, `losses.apl_loss` |
| `h^FG_z`, top-`zη` local matching (Eq. 4) | `FineGrainedBranch.similarity` |
| `w^GI`, `L^GI_CE` with λ₁, λ₂ | `GlobalInvariantBranch`, `losses.global_invariant_loss` |
| `p(m\|I) = ⅓[p^GI + p^ID + 1/Z Σ p^FG_z]` | `losses.ensemble_probability` |
| `T = {w^GI, h^ID, h^FG_1..Z}`, `z_i = cos(f̂,t_i)` | `logic.evidence.EvidenceExtractor` |
| `ψ(z_jq) = σ(κ(z_jq − τ_q))` | `logic.soft_logic.SoftLogicGate` |
| `γ_j = Σ_q ω_jq ψ(z_jq)`, `ω_jq = 1/Q_j` | `soft_logic.compatibility_scores` |
| `G_t = ∪_j {(j,e)}` | `EvidenceExtractor.gaps` / `gap_set` |
| `p_{t+1} = R(D_t,G_t,S)`, `T_{t+1} = θ_FG(T_t, α^gap)` | `refinement.PromptRefiner` |
| Algorithm 1 | `refinement.loop.run_refinement_loop` |
| `y_j ∈ {V, U, N}`, Γ_loose/Γ_strict | `logic.rule_engine.RuleEngine` |
| Privacy Score (Eq. 6) / Semantic Utility (Eq. 7) | `eval.privacy_score` / `eval.semantic_utility` |
| Avg. Risk (Eq. 5) | `eval.membership_inference_attack` |

`docs/METHOD.md` carries the full derivation-to-code walkthrough.

## Reproducing the experiments

All scripts share `--config`, `--set key=value …`, `--device`, `--seed` and
`--output`.

```bash
# Table II — 16-shot top-1 accuracy (and the Fig. 2 shot curve)
python scripts/train_fewshot.py --config configs/fewshot_16shot.yaml \
    --set data.root=/datasets/caltech101 data.dataset=imagefolder --seeds 1 2 3
python scripts/train_fewshot.py --config configs/fewshot_16shot.yaml --shot-curve

# Table III — base-to-new generalisation
python scripts/eval_base_to_new.py --config configs/fewshot_16shot.yaml --seeds 1 2 3

# Tables IV–VII — privacy, replay attacks, membership inference, text quality
python scripts/eval_privacy.py --config configs/its_inference.yaml \
    --reconstructions attacker_outputs.pt

# Table VIII — downstream ITS tasks on each input representation
python scripts/eval_downstream.py --config configs/its_inference.yaml

# Table IX — component ablation
python scripts/run_ablation.py --config configs/its_inference.yaml

# Inference on your own frames
python scripts/run_inference.py --images scene1.jpg scene2.jpg --verbose-trace
```

Image-folder datasets use the usual `root/train/<class_name>/<image>.jpg`
layout. Annotated traffic scenes use the JSON manifest documented at the top of
`apl/data/traffic.py`, which carries the per-sample violation labels, the
sensitive attributes (for Privacy Score) and the task-relevant attributes (for
Semantic Utility). With no dataset configured, every script falls back to the
synthetic generator so each table can be produced end to end as a dry run.

### Scenario bank

`data/scenarios/dmv_scenario_bank.json` holds the `M = 12` scenarios: red-light
running, stop-sign violation, wrong lane use, illegal turn, crosswalk
obstruction, speed-zone violation, bus-lane intrusion, railroad-crossing
violation, failure to yield to an emergency vehicle, parking in a prohibited
zone, unsafe following distance and work-zone violation. Each entry carries its
violation type `ν_j`, its evidentiary predicates `E_j`, its decision rule `r_j`
and a legal reference. Predicates are phrased as privacy-neutral observations
about road state, vehicle state and signage — a unit test asserts that none of
them mentions an identity attribute. Swap in your own jurisdiction by editing
the JSON; nothing in the code hard-codes the twelve scenarios.

## Tests

```bash
python -m unittest discover -s tests -t tests     # 90 tests, ~3 s on CPU
pytest tests                                       # same suite under pytest
```

They cover the feature generator's algebra (including that it starts as the
identity), each branch's shapes and semantics, the top-`k` ladder, the loss
weighting and regularisers, the scenario bank's invariants, the soft-logic
gate's calibration, the three-way decision thresholds, gap detection,
Algorithm 1's three stopping criteria, the PII filter, every metric, the
optimiser schedule, and the end-to-end forward/infer/checkpoint paths.

## Implementation notes and deviations

The paper leaves a few details open; each choice below is configurable, and the
default is named in brackets.

1. **Semantic categories `C`.** Sec. V-D scores every row `t_i ∈ rows(T)`
   against the visual embedding and then feeds those scores to the rule engine,
   which consumes *predicates*. We therefore make the rows of `w` the predicate
   inventory of the scenario bank, so `z_i` is directly `z_jq`. For the
   prompt-learning benchmarks the rows are the dataset's class names instead,
   and the discrete layer is disabled (`model.logic_enabled`).
2. **Top-`zη` selection (Eq. 4).** Read literally, `η = 1` selects `z ∈ {1..4}`
   patches out of `N = 196`, while the text describes `η` as the *proportion* of
   local features used. We implement both: `branches.fg_topk_mode =
   proportion` [default] uses `k_z = ⌈z/Z · η · N⌉`, a genuine coarse-to-fine
   ladder; `count` follows the formula literally.
3. **Evidence for fine-grained rows.** `logic.fg_local_evidence` [true] scores
   `h^FG_z` against the local features with the same top-`k` pooling the branch
   is built for; set it to `false` to score every row against `f̂` as written in
   Sec. V-D.
4. **`w^GI`.** The paper says it is "derived from the base feature `w`" without
   fixing the operator. We reuse `θ_FG` with a small learnable, image-independent
   conditioning bank, which keeps the branch scene-agnostic while leaving `w^GI`
   trainable under the λ₁/λ₂ regularisers.
5. **One shared `θ_FG`.** Sec. V-B2 describes a single generator conditioned by
   three signal classes, so all branches and the refinement step share one
   instance. Its output projections are zero-initialised, which makes
   `θ_FG(w, α) = w` at step 0 and keeps early training stable.
6. **`α^gap(G_t)`.** Encoded missing-predicate prompts, ranked by how many
   scenarios each gap blocks and by how far the predicate sits below the gap
   threshold, pooled into `L = 8` rows and added to a learnable gap bank.
7. **Trainable budget.** The default configuration yields ≈1.33 M trainable
   parameters (`model.parameter_report()`), against the ≈1.7 M quoted in
   Table I; the gap is in how the generator's projections are counted, and the
   budget stays in the "single-prompt" regime the paper targets.
8. **Description generation.** The paper does not specify the surface
   realisation, so `DescriptionComposer` emits a deterministic, auditable
   template over observed / partially visible / unobserved predicates, followed
   by the rule-engine verdict. `apl/describe.py` additionally redacts residual
   PII (plates, faces, names, contact details, VINs, addresses) and reports what
   it removed, so the Privacy Score is computed on real counts.
9. **Attack models.** Reconstruction and replay metrics (SSIM/PSNR/MSE/SRRA)
   and the membership-inference attacks are implemented as *measurement*
   functions; the attacker itself is pluggable (`--reconstructions`, or the
   embedding probe/gallery pair). We do not ship AdvFace or the RL-MoE
   baselines.
10. **Multi-label evaluation.** The paper flags the missing per-violation
    breakdown as a limitation; `eval.scenario_prf` supplies micro precision,
    recall, F1 and exact-set accuracy whenever scenario-level annotations exist.

Known limits carried over from the method: coverage is bounded by the scenario
bank, a low compatibility score is not proof that an unobserved predicate is
false, and the scene-level accuracy does not certify that every co-occurring
violation was recovered.

## Citation

```bibtex
// copy citation here any time published
```

This work was supported in part by the National Science Foundation under
Grants No. CNS-2543209 and CNS-2318725; by the U.S. Department of
Transportation, University Transportation Center (UTC), Transportation
Cybersecurity Center for Advanced Research and Education (CYBER-CARE), under
Grant No. 69A3552348332.

## License

MIT — see [LICENSE](LICENSE).
