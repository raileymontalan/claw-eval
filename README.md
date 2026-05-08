# Claw-Eval (AISG Internal Fork)

> **This is an internal AISG fork of [claw-eval/claw-eval](https://github.com/claw-eval/claw-eval) for evaluating locally-served models. It is not connected to the official Claw-Eval leaderboard and results are not submitted externally.**
>
> Default model under evaluation: [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B), served locally via vLLM. Judge and user-agent: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b), also served locally.

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

Works with CUDA 12.4 (default driver on the cluster).

### Running on SLURM

For cluster jobs, use the provided script which handles vLLM startup, health-checking, config patching, and teardown:

```bash
# Default run (Qwen3-32B, general + multi_turn splits, 3 trials)
sbatch run_claweval.slurm

# Different model
MODEL=aisingapore/Other-Model sbatch run_claweval.slurm

# Fewer parallel workers and trials (faster, less robust)
PARALLEL=4 TRIALS=1 sbatch run_claweval.slurm

# General split only
FILTER_MULTI_TURN="" sbatch run_claweval.slurm
```

| SLURM variable    | Default                    | Description                                              |
| ----------------- | -------------------------- | -------------------------------------------------------- |
| `MODEL`           | `Qwen/Qwen3-32B`           | HuggingFace model ID under evaluation                    |
| `MODEL_TP`        | `1`                        | Tensor parallel size for agent model                     |
| `JUDGE_MODEL`     | `openai/gpt-oss-120b`      | Judge + user-agent model (served locally on GPU 1)       |
| `JUDGE_TP`        | `1`                        | Tensor parallel size for judge model                     |
| `PARALLEL`        | `8`                        | Concurrent claw-eval workers                             |
| `TRIALS`          | `3`                        | Independent trials per task (Pass^k metric)              |
| `FILTER_GENERAL`  | `tasks/t`                  | Substring filter for general (T*) split                  |
| `FILTER_MULTI_TURN` | `tasks/c`                | Substring filter for multi_turn (C*) split               |
| `OUTPUT_DIR`      | `traces/<model basename>`  | Trace output directory                                   |

Traces are saved to `OUTPUT_DIR/`. Both splits write to the same directory so the final summary covers all 199 non-multimodal tasks.

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
