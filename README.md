# Claw-Eval (AISG Internal Fork)

> **This is an internal AISG fork of [claw-eval/claw-eval](https://github.com/claw-eval/claw-eval) for evaluating locally-served models. It is not connected to the official Claw-Eval leaderboard and results are not submitted externally.**
>
> Default model under evaluation: [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), served locally via vLLM. Judge and user-agent: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b), also served locally.

Claw-Eval benchmarks autonomous agents on 300 human-verified real-world tasks across 9 categories. Agents are graded on completion, safety, and robustness (Pass^3 across 3 independent trials).

**Splits evaluated in this fork** (multimodal M* tasks skipped — require Docker):

| Split | Count | Description |
|-------|-------|-------------|
| `general` (T*) | 161 | Core agent tasks across communication, finance, ops, productivity, etc. |
| `multi_turn` (C*) | 38 | Conversational tasks with simulated user personas |

## Quick Start

```bash
pip install uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[vllm]"
```

Works with CUDA 13 (loaded automatically in the SLURM script).

### Running on SLURM

Use `submit_claweval.sh` to submit jobs. It reads model settings from `config_vllm.yaml` and automatically sets the correct GPU count:

```bash
# Default model (from config_vllm.yaml eval.default_model)
./submit_claweval.sh

# Specific model
./submit_claweval.sh Qwen/Qwen3-32B

# Benchmark overrides via env vars
PARALLEL=4 TRIALS=1 ./submit_claweval.sh Qwen/Qwen3.6-27B

# Chinese tasks only
LANGUAGE=zh ./submit_claweval.sh
```

To add a new model, add an entry to `config_vllm.yaml` under `models:` with its `tp`, `enable_thinking`, `tool_call_parser`, and `reasoning_parser`. No SLURM changes needed.

| Variable          | Default                        | Source                          | Description                                              |
| ----------------- | ------------------------------ | ------------------------------- | -------------------------------------------------------- |
| `MODEL`           | `Qwen/Qwen3.6-27B`             | `config_vllm.yaml` → submit arg | HuggingFace model ID under evaluation                    |
| `MODEL_TP`        | `1`                            | `config_vllm.yaml models[MODEL].tp` | Tensor parallel size — set per model in config      |
| `JUDGE_MODEL`     | `openai/gpt-oss-120b`          | `config_vllm.yaml eval.judge_model` | Judge + user-agent model                            |
| `JUDGE_TP`        | `1`                            | `config_vllm.yaml eval.judge_tp` | Tensor parallel size for judge                         |
| `PARALLEL`        | `8`                            | `config_vllm.yaml eval.parallel` | Concurrent claw-eval workers                           |
| `TRIALS`          | `3`                            | `config_vllm.yaml eval.trials` | Independent trials per task (Pass^k metric)              |
| `LANGUAGE`        | `en`                           | `config_vllm.yaml eval.language` | Filter by language (`en`/`zh`). Empty = all.           |
| `OUTPUT_DIR`      | `traces/<model basename>`      | SLURM script                    | Trace output directory                                   |

Traces are saved to `OUTPUT_DIR/`. Both splits write to the same directory so the final summary covers completed tasks. Re-runs use `--continue` to skip already-completed trials.

## Reference

Upstream leaderboard, dataset, contributor list, and citation info below — not used in this fork's evaluation setup.

<details>
<summary>Upstream documentation</summary>

<div align="center">

<img src="claw_eval.png" width="160" alt="Claw-Eval Logo">

[![Tasks](https://img.shields.io/badge/tasks-300-blue)](#tasks)
[![Models](https://img.shields.io/badge/models-14-green)](#leaderboard)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2604.06132v1)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-purple)](https://claw-eval.github.io)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/claw-eval/Claw-Eval)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

> 300 human-verified tasks | 2,159 rubrics | 9 categories | Completion · Safety · Robustness.

</div>

### Leaderboard

Browse the full leaderboard at **[claw-eval.github.io](https://claw-eval.github.io)**.

### Dataset

Available on Hugging Face: [claw-eval/Claw-Eval](https://huggingface.co/datasets/claw-eval/Claw-Eval)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique task identifier |
| `query` | string | Task instruction / description |
| `fixture` | list[string] | Fixture files required (available in `data/fixtures.tar.gz`) |
| `language` | string | `en` or `zh` |
| `category` | string | Task domain |

### Core Contributors

[Bowen Ye](https://github.com/pkuYmiracle) (PKU), [Rang Li](https://github.com/lirang04) (PKU), [Qibin Yang](https://github.com/yangqibin-caibi) (PKU), [Zhihui Xie](https://zhxie.site/) (HKU), [Yuanxin Liu](https://llyx97.github.io/) (PKU), [Linli Yao](https://yaolinli.github.io/) (PKU), [Hanglong Lyu](https://github.com/Albus2002) (PKU), [Lei Li](https://lilei-nlp.github.io/) (HKU, project lead)

### Citation

```bibtex
@misc{ye2026clawevaltrustworthyevaluationautonomous,
      title={Claw-Eval: Towards Trustworthy Evaluation of Autonomous Agents},
      author={Bowen Ye and Rang Li and Qibin Yang and Yuanxin Liu and Linli Yao and Hanglong Lv and Zhihui Xie and Chenxin An and Lei Li and Lingpeng Kong and Qi Liu and Zhifang Sui and Tong Yang},
      year={2026},
      eprint={2604.06132},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.06132},
}
```

</details>

## License

This project is released under the [MIT License](LICENSE).
