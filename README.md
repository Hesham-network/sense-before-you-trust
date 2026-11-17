# Sense Before You Trust

Reproducibility code for the paper **"Sense Before You Trust: A Two-Channel Defense
(Physical Audit + Relay Guard) against Prompt-Infection in LLM Multi-Agent IoT Networks"**
by Hesham A. Sakr.

This repository contains the single self-contained experiment script used for all
real-LLM results in the paper.

## What it does

A closed, simulated IoT network of LLM relay agents is exposed to a prompt-infection
payload. The script measures, with 95% confidence intervals:

- **F1 — forwarding as a choice:** whether an agent relays a manipulation and/or
  misreports its physical measurement, even when given an explicit refusal option
  (eight-payload behavioral suite, measured in isolation).
- **Live cascades** over the agent graph under the `alert` and audit-evading
  `under-report` payloads, for the defenses: `none`, `guard`, `audit`, `audit+guard`,
  `sense_only` (controller-only sensing baseline), `infaguard`, `infaguard+audit`
  (a propagation-aware text baseline that screens both the note and relay channels).

It reports extinction probability, infected agent-rounds, SINR outages per round,
energy overhead, and the measurement (probe) budget per agent-round.

## Models

Qwen2.5-7B-Instruct, IBM Granite-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
all 4-bit (NF4) on a single T4/P100 GPU. Mistral is gated on Hugging Face and needs
an `HF_TOKEN`.

## How to run (Google Colab or Kaggle, one GPU)

```bash
pip install -U transformers accelerate bitsandbytes
# choose the model, then run:
SBYT_MODEL=qwen python sbyt_llm_v11.py        # then granite, then mistral
```

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `SBYT_MODEL` | `qwen` \| `granite` \| `mistral` | `qwen` |
| `HF_TOKEN` | Hugging Face token (required for Mistral) | — |
| `SBYT_OUT` | output directory | current dir |
| `SBYT_LIVE_REPS` | replicates per (payload, defense) | 10 |
| `SBYT_LIVE_PAYLOADS` | comma list, e.g. `underreport,adaptive` | `underreport,adaptive` |
| `SBYT_LIVE_DEFS` | comma list of defenses to run | all |
| `SBYT_SKIP_F1` | `1` to skip the forwarding-choice block | 0 |
| `SBYT_BATCH_REPS` | checkpoint every N replicates | 2 |
| `SBYT_MOCK` | `1` for a no-GPU pipeline test (fake LLM) | 0 |

The run is **checkpointed every `SBYT_BATCH_REPS` replicates** and is **resumable**:
re-running with the same `SBYT_MODEL` and output continues only the
`(payload, defense)` pairs that are not yet complete.

## Outputs

For each model the script writes:

- `sbyt_v11_<model>_results.txt` — human-readable log (tables with CIs).
- `sbyt_v11_<model>_results.json` — structured results (per-defense extinction,
  infected-rounds, outages, probe cost, and the measured cascade primitives).
- `sbyt_v11_<model>_raw.jsonl` / `_samples.txt` — raw per-call records and curated
  transcript samples.

Exact model names, revisions, quantization and decoding parameters are logged in the
header of every run for provenance.

## Note

This is a **defensive** study in a **closed, fully simulated** IoT/RF network. The only
"attack" is fudging one reported interference value; no real network, model service, or
third party is involved.
