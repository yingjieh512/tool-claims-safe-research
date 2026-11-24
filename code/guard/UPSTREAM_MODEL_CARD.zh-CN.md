---
language:
- en
- zh
---

> English translation of the upstream Chinese model card by Wenbin Hu / HKUST / ModeIO. The original filename is retained for source traceability. This is historical upstream documentation; the temporary hosted endpoint below is retained as a historical reference, not a claim of current service availability. For this package's current local setup, see [the local Guard guide](README.md).

<video controls playsinline src="https://huggingface.co/hubin/jev-guard-v3.2-2b/resolve/main/assets/hermes-jev-demo.mp4" poster="https://huggingface.co/hubin/jev-guard-v3.2-2b/resolve/main/assets/demo-poster.png"></video>

[Watch the introduction with English/Chinese captions](https://hkust-knowcomp.github.io/jev-guardrail/#hermes)

# Jev Guard v3.2 · 2B

[Full upstream English card](UPSTREAM_MODEL_CARD.md) · [Upstream Chinese-language website](https://hkust-knowcomp.github.io/jev-guardrail/zh/)

**Wenbin Hu · HKUST × ModeIO**

Maintain an editable policy-classification tree. Route user requests and proposed actions through successive levels, predict safety and category at a leaf, and let the harness combine those predictions with authorization rules to decide whether to proceed. Blocked requests do not reach the language model; blocked tools do not execute. Interceptions appear in the Hermes conversation.

## Model and training data

Qwen3.5-2B with rank-8 LoRA, a binary safety head, a candidate-category similarity head, and a severity head consisting of a scalar with ordered thresholds. The text service supports 16,384 tokens without silent truncation. Version 3.2 continues training from v3.1 for one epoch and selects update 1,508; this run did not establish a validation plateau.

Training includes 377,930 review records and 8,000 router records. The data covers GSPR3 content safety, Omni policy compliance, vulnerability assessment with complete source code retained, phishing, prompt injection, authorization, and privacy. The full validation corpus contains 65,530 records. Model selection uses 2,048 fixed native examples and 1,000 routing examples. Routing train/validation/test sizes are 8,000/1,000/1,000. Unknown category labels are excluded from the loss, and weak rule-based category labels receive weight 0.25. Training data is stored in the private [Hugging Face dataset](https://huggingface.co/datasets/hubin/jev-guard-training).

## Eval

**93.20% full-path routing accuracy** · **91.78% overall safety accuracy** · **58.0 ms mean native forward latency**.

Routing uses 1,000 held-out cases; safety is 73,415/79,987 supervised records. Forward latency is a case-weighted mean over 147,643 records and excludes the complete routing-plus-review sequence.

| Domain | Task records | Safety-labeled records | Safety accuracy | Category micro-F1 |
|---|---:|---:|---:|---:|
| Content | 11,909 | 11,909 | 85.20% | 61.86% |
| Cyber | 61,972 | 57,478 | 93.77% | 46.96% |
| Privacy | 69,051 | 5,889 | 82.92% | 78.40% |
| Compliance | 4,711 | 4,711 | 95.31% | 58.17% |

The safety accuracy above is ordinary accuracy; records without safety supervision are excluded. Evaluation covers 147,643 task records and 109 views, preserving the original v3.1 candidate categories and labels; views may overlap. Routing makes greedy predictions from the root without access to gold intermediate nodes, while receiving the caller's audit context. Severity exact accuracy is 15.28%, with normalized MAE 0.1727. Controlled action replay blocked 49/52 malicious proposals and allowed 87/97 benign proposals; this is not an end-to-end attack-success measurement.

## Inference

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf download hubin/jev-guard-v3.2-2b --local-dir jev-model
cd jev-model
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python serve.py --base-model Qwen/Qwen3.5-2B --port 29660
```

The upstream card described a public preview API that required neither a local GPU nor an API key:

```bash
curl -fsS https://approaches-lemon-antique-garden.trycloudflare.com/health
```

The complete API example is in [the upstream English card](UPSTREAM_MODEL_CARD.md#hosted-https-api). The upstream description states that the public interface connected to the existing v3.2 update-1,508 model, accepted 16K text without truncation, and did not log inputs at the gateway. The address was a temporary HTTPS tunnel and could change when recreated; capacity limits returned HTTP 429. The preceding installation commands are for self-hosting, and the adapter's base weights must be downloaded separately.

## Hermes

The upstream installation procedure installed the supplement and connected it to the deployed public review service:

```bash
curl -fsSL https://huggingface.co/hubin/jev-guard-v3.2-2b/resolve/main/downloads/install-hermes-jev.txt -o /tmp/install-hermes-jev.sh
bash /tmp/install-hermes-jev.sh --endpoint https://approaches-lemon-antique-garden.trycloudflare.com
jev-hermes
```

Configure the language-model provider with `hermes setup`. Enter `/safety-policy` during chat to edit the actual policy tree. Use `jev-hermes` to review both user requests and tool actions; interceptions are highlighted in red. For self-hosting, replace `--endpoint` with your HTTPS address or `http://127.0.0.1:29660`. Configure `JEV_API_KEY` when authentication is required.
