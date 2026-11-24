---
base_model: Qwen/Qwen3.5-2B
language:
- en
- zh
tags:
- classification
- agent-safety
- policy-routing
- lora
- privacy
- cybersecurity
library_name: peft
pipeline_tag: text-classification
license: other
license_name: research-preview
license_link: https://huggingface.co/hubin/jev-guard-v3.2-2b/blob/main/LICENSE-NOTICE.md
---

<video controls playsinline src="https://huggingface.co/hubin/jev-guard-v3.2-2b/resolve/main/assets/hermes-jev-demo.mp4" poster="https://huggingface.co/hubin/jev-guard-v3.2-2b/resolve/main/assets/demo-poster.png"></video>

[Watch the introduction with English/Chinese captions](https://hkust-knowcomp.github.io/jev-guardrail/#hermes)

# Jev Guard v3.2 · 2B

[Website](https://hkust-knowcomp.github.io/jev-guardrail/) · [English translation of the upstream Chinese card](UPSTREAM_MODEL_CARD.zh-CN.md)

Your policy tree. Jev reviews each agent step.

Maintain an editable policy tree for **content, cyber, privacy and compliance**. Route every user request and proposed tool action from the root to a leaf, then classify its **safety and category** before the harness proceeds. Users can change routing descriptions, leaf taxonomies and review rules. A blocked user request never reaches the assistant model; a blocked tool action never executes, and the interception appears in the conversation.

**Wenbin Hu · HKUST × [ModeIO](https://www.modeio.ai/)**

**93.20% full-path routing accuracy** · **91.78% overall safety accuracy** · **58.0 ms mean native forward latency**.

Routing uses 1,000 held-out cases; safety is 73,415/79,987 supervised records. Forward latency is a case-weighted mean over 147,643 records and excludes the complete routing-plus-review sequence.

## Model information

| Item | Configuration |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` |
| Adaptation | Rank-8 LoRA on language-backbone attention projections |
| Checkpoint | v3.2 continuation from v3.1; selected update 1,508 after one epoch |
| Context | 16,384 tokens; oversized serving requests rejected |
| Routing / single choice | Candidate-conditioned softmax |
| Multi-label categories | Independent candidate sigmoid probabilities |
| Safety | Linear two-class head: `unsafe` / `safe` |
| Ordinal score | Scalar severity plus three ordered learned thresholds |
| Serving | Text only; native multimodal training/evaluation requires original media |

The package contains the adapter, typed heads, routing tree and standalone inference code. Download the Qwen base weights separately. This is a classifier, not a chat model; it does not generate a rationale. A category/severity prediction is not an authorization decision.

## Training data

v3.2 trains on **377,930 native task cases + 8,000 router cases**, including retained GSPR3 and Omni examples plus cybersecurity and privacy sources. Code tasks retain complete source functions from PrimeVul/DiverseVul; other tasks cover phishing, prompt injection, vulnerability severity/root cause, scoped security assistance, privacy-policy practices, information flow and policy compliance. v3.1 introduced meaningful CWE names and hierarchical router supervision. v3.2 unifies English leaf taxonomies, merges content/privacy branches, removes the agent branch and uses category masks: unknown labels are ignored, weak rule labels have weight 0.25.

The full native validation corpus has **65,530** records; checkpoint selection uses **2,048** stratified native development records plus **1,000** routing records. Router train/validation/test = **8,000/1,000/1,000**. The current data archive is private: [hubin/jev-guard-training](https://huggingface.co/datasets/hubin/jev-guard-training). It preserves versioned data, provenance, hashes and original media references. Source-specific licensing restrictions remain applicable.

The continuation uses global batch 256, maximum microbatch 32, 65,536 padded tokens per GPU, seed 42, adapter LR 2e-5 and head LR 5e-5 on four GPUs. A validation plateau was not established for this one-epoch run. See `training-config.json` and `checkpoint/identity.json` for the recorded configuration and identity.

## Evaluation

| Domain | Task records | Safety-labeled records | Safety accuracy | Category micro-F1 |
|---|---:|---:|---:|---:|
| Content | 11,909 | 11,909 | 85.20% | 61.86% |
| Cyber | 61,972 | 57,478 | 93.77% | 46.96% |
| Privacy | 69,051 | 5,889 | 82.92% | 78.40% |
| Compliance | 4,711 | 4,711 | 95.31% | 58.17% |

Native evaluation covers **147,643 task-case records across 109 benchmark views**. Safety uses only safety-supervised records; category micro-F1 uses only category-supervised records. These are task records, not necessarily independent source examples. The numbers above are ordinary **safety accuracy**, not balanced accuracy. Original v3.1 test inputs, candidate strings, gold labels, modes and supervision are retained; no mapped-gold evaluation is substituted.

Full-path routing accuracy is **93.20% (932/1,000)**: greedy routing from root, without gold intermediate nodes or probability/margin filtering, with explicit caller audit context. Ordinal severity exact accuracy is **15.28%** on 1,466 cases, normalized MAE **0.1727**. Controlled AgentDojo action-state replay caught **49/52** malicious proposals and allowed **87/97** benign proposals; 9 benign non-allow decisions and 1 API error remained. No attacker action was executed; this is not an end-to-end attack-success evaluation.

Per-benchmark results and denominators: `evaluation-summary.json`. Single-choice, multi-label and ordinal probabilities match the deployed checkpoint in `inference-parity.json`.

## Inference

### Hosted HTTPS API

The deployed v3.2 reviewer is available without a local GPU or API key:

```bash
curl -fsS https://approaches-lemon-antique-garden.trycloudflare.com/health
curl -fsS https://approaches-lemon-antique-garden.trycloudflare.com/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"Classify this request: explain gravity. Educational explanations are allowed.","options":["No applicable risk category","Unauthorized credential access"],"mode":"single","safety_applicable":true}'
```

This is a temporary hosted preview endpoint. Requests use the existing v3.2 checkpoint (update 1,508); inputs are not written to gateway logs. It accepts text up to 16,384 tokens without truncation. A full routing path makes several `/predict` calls. Capacity limits return HTTP 429; try again later. The endpoint may change when the tunnel is recreated.

### Self-hosted inference

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf download hubin/jev-guard-v3.2-2b --local-dir jev-model
cd jev-model
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python serve.py --checkpoint checkpoint \
  --base-model Qwen/Qwen3.5-2B --port 29660
```

```python
from jev_inference import JevClassifier
model = JevClassifier("checkpoint", base_model="Qwen/Qwen3.5-2B")
result = model.predict(
    "Assess this request: explain what gravity is.",
    options=["No applicable risk category", "Unauthorized credential access"],
    mode="single", safety_applicable=True,
)
print(result["safety"], result["safety_probs"], result["probabilities"])
# Multi-label: mode="multi". Severity: mode="score", options in this order:
# ["low", "medium", "high", "critical"].
```

```bash
curl -fsS http://127.0.0.1:29660/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"Assess this request: explain gravity.","options":["No applicable risk category","Unauthorized credential access"],"mode":"single","safety_applicable":true}'
```

The self-hosted server binds loopback. For your own remote deployment set `JEV_API_KEY`, use an authenticated HTTPS proxy and pass the API key through the client environment. Hugging Face hosts the downloadable weights; the hosted API above runs separately.

## Use in Hermes

Run these commands to connect Hermes to our hosted reviewer:

```bash
curl -fsSL https://huggingface.co/hubin/jev-guard-v3.2-2b/resolve/main/downloads/install-hermes-jev.txt -o /tmp/install-hermes-jev.sh
bash /tmp/install-hermes-jev.sh --endpoint https://approaches-lemon-antique-garden.trycloudflare.com
jev-hermes
```

The installer installs official Hermes if missing, uses its managed Python, installs the Jev supplement, enables the plugin and checks the reviewer. Configure your assistant provider with `hermes setup`; existing provider settings and user policy are preserved. To use your own reviewer, replace `--endpoint` with its HTTPS URL or `http://127.0.0.1:29660` for self-hosted inference; both client and server use `JEV_API_KEY` when enabled.

Use **`jev-hermes`** or **`hermes jev-chat`** for both request and action review. Plain official Hermes has only the enabled pre-tool hook. In chat, `/safety-policy` opens the editable tree and review-policy panel. Jev interceptions appear in red. `ask` currently stops execution and shows a notice; this plugin does not supply an approval-bypass workflow. The integration was verified against Hermes commit `eb7e8620324b32424c06218f6a28094df2e921f8`.

```bash
jev doctor
jev-hermes --once -q "Explain gravity in two sentences."
jev-hermes --resume SESSION_ID
```
