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

### 1. Install (one-time, GPU node required)

```bash
sbatch setup_env.slurm
```

Check `logs/setup_<jobid>.out` for completion. Creates `.venv/` with claw-eval + vLLM. A GPU node is required — vLLM compiles CUDA kernels on first install.

### 2. Add models to evaluate

Edit `config_vllm.yaml`:

```yaml
eval:
  default_model: Qwen/Qwen3.6-27B   # ← change this to switch default

models:
  Qwen/Qwen3.6-27B:
    tp: 1
    enable_thinking: true
    reasoning_parser: qwen3
    tool_call_parser: qwen3_coder

  # Example: add a new model
  Your/Model-Name:
    tp: 1
    enable_thinking: true
    tool_call_parser: qwen3_coder   # check vLLM docs for your model
    reasoning_parser: qwen3
```

`tp` = tensor parallel size (number of GPUs). The submit script reads this to request the right GPU count automatically.

### 3. Submit

```bash
# Default model from config_vllm.yaml
./submit_claweval.sh

# Specific model
./submit_claweval.sh Qwen/Qwen3.6-27B

# Chinese tasks only
LANGUAGE=zh ./submit_claweval.sh

# Smoke test
PARALLEL=4 TRIALS=1 ./submit_claweval.sh Qwen/Qwen3.6-27B
```

Logs: `logs/claweval_<jobid>.out`

Both splits (T* general, C* multi-turn) run sequentially into the same `OUTPUT_DIR`. Re-runs automatically use `--continue` to skip completed trials.

| Variable          | Default                        | Source                              | Description                                              |
| ----------------- | ------------------------------ | ----------------------------------- | -------------------------------------------------------- |
| `MODEL`           | `Qwen/Qwen3.6-27B`             | `config_vllm.yaml` → submit arg     | HuggingFace model ID under evaluation                    |
| `MODEL_TP`        | `1`                            | `config_vllm.yaml models[MODEL].tp` | Tensor parallel size — set per model in config           |
| `JUDGE_MODEL`     | `openai/gpt-oss-120b`          | `config_vllm.yaml eval.judge_model` | Judge + user-agent model                                 |
| `JUDGE_TP`        | `1`                            | `config_vllm.yaml eval.judge_tp`    | Tensor parallel size for judge                           |
| `PARALLEL`        | `8`                            | `config_vllm.yaml eval.parallel`    | Concurrent claw-eval workers                             |
| `TRIALS`          | `3`                            | `config_vllm.yaml eval.trials`      | Independent trials per task (Pass^3 metric)              |
| `LANGUAGE`        | `en`                           | `config_vllm.yaml eval.language`    | Filter by language (`en`/`zh`). Empty = all.             |
| `OUTPUT_DIR`      | `traces/<model basename>`      | SLURM script                        | Trace output directory                                   |

## Results

After the job finishes, the SLURM script runs `score_summary.py` on the trace directory. Output is printed to the job log and saved to `OUTPUT_DIR/score_summary.json`.

To re-score manually:

```bash
srun --gres=gpu:0 --mem=8G .venv/bin/python score_summary.py traces/
```

### AISG evaluation results (T* general + C* multi-turn tasks)

Columns match the official [claw-eval leaderboard](https://claw-eval.github.io/) definitions. Models ordered alphabetically.

| Model | Tasks | Avg Score | Completion | Robustness | Safety | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|:-----:|:---------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|
| aisingapore/gemma4_e2b_cand1 | 107 | 0.344 | 0.254 | 0.864 | 0.966 | 12/107 (11%) | 0.084 | 8/107 (7%) | 7/107 (7%) |
| aisingapore/gemma4_e2b_cand2 | 107 | 0.357 | 0.266 | 0.905 | 0.966 | 12/107 (11%) | 0.087 | 9/107 (8%) | 7/107 (7%) |
| aisingapore/Qwen-SEA-LION-v4.5-27B | 107 | 0.634 | 0.597 | 0.900 | 0.955 | 60/107 (56%) | **0.511** | **56/107 (52%)** | **48/107 (45%)** |
| google/gemma-4-31B-it | 107 | 0.522 | 0.457 | 0.864 | **0.981** | 32/107 (30%) | 0.252 | 27/107 (25%) | 22/107 (21%) |
| google/gemma-4-E2B-it | 107 | 0.387 | 0.303 | **0.912** | 0.976 | 18/107 (17%) | 0.143 | 17/107 (16%) | 11/107 (10%) |
| google/gemma-4-E4B-it | 107 | 0.329 | 0.232 | 0.898 | 0.969 | 9/107 (8%) | 0.072 | 7/107 (7%) | 7/107 (7%) |
| Qwen/Qwen3.5-27B | 107 | 0.624 | 0.592 | 0.858 | 0.952 | 60/107 (56%) | 0.480 | 52/107 (49%) | 42/107 (39%) |
| Qwen/Qwen3.6-27B | 107 | **0.645** | **0.609** | 0.910 | 0.958 | **62/107 (58%)** | **0.511** | 55/107 (51%) | 47/107 (44%) |

### Thinking-off (nothink) variants

| Model | Tasks | Avg Score | Completion | Robustness | Safety | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|:-----:|:---------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|
| aisingapore/gemma4_e2b_cand1_nothink | 107 | 0.370 | 0.281 | 0.915 | 0.959 | 14/107 (13%) | 0.103 | 11/107 (10%) | 8/107 (7%) |
| aisingapore/gemma4_e2b_cand2_nothink | 107 | 0.358 | 0.269 | 0.899 | 0.963 | 13/107 (12%) | 0.100 | 11/107 (10%) | 8/107 (7%) |
| aisingapore/Qwen-SEA-LION-v4.5-27B_nothink | 107 | **0.614** | **0.579** | 0.878 | 0.961 | **54/107 (50%)** | **0.455** | **49/107 (46%)** | **43/107 (40%)** |
| google/gemma-4-31B-it_nothink | 107 | 0.532 | 0.464 | **0.907** | **0.981** | 34/107 (32%) | 0.268 | 29/107 (27%) | 23/107 (21%) |
| google/gemma-4-E2B-it_nothink | 107 | 0.339 | 0.242 | 0.905 | 0.973 | 9/107 (8%) | 0.075 | 8/107 (7%) | 7/107 (7%) |
| google/gemma-4-E4B-it_nothink | 107 | 0.328 | 0.224 | 0.845 | 0.971 | 10/107 (9%) | 0.078 | 8/107 (7%) | 7/107 (7%) |
| Qwen/Qwen3.5-27B_nothink | 107 | 0.544 | 0.504 | 0.809 | 0.962 | 47/107 (44%) | 0.374 | 39/107 (36%) | 34/107 (32%) |
| Qwen/Qwen3.6-27B_nothink | 107 | 0.582 | 0.540 | 0.835 | 0.978 | 51/107 (48%) | 0.396 | 44/107 (41%) | 32/107 (30%) |

Metric definitions (from the claw-eval paper):
- **Avg Score** — mean task score across all 3 trials (0–1); missing trials padded with 0
- **Completion** — task objective satisfaction weighted by rubric item importance
- **Robustness** — fraction of injected-error tool types that subsequently recovered (1.0 if no errors injected)
- **Safety** — multiplicative gate penalising policy violations; avoidance of harmful/unauthorised actions
- **Pass@1** — tasks where ≥1 of 3 trials scored ≥ 0.75 (optimistic; upper bound on capability)
- **Pass^1** — per-trial pass rate: fraction of all task-trial pairs scoring ≥ 0.75
- **Pass^2** — tasks where ≥2 of 3 trials scored ≥ 0.75
- **Pass^3** — tasks where all 3 trials scored ≥ 0.75 (strict reliability; primary leaderboard metric)

Tasks with fewer than 3 graded trials or error traces are listed as anomalies in the score_summary output. Re-run with `--continue` to fill in missing trials.

> **Known limitation — officeqa tasks T074–T085 (context overflow):**
> These 12 tasks pass large OCR documents (~778 KB / ~195K tokens for T085) as tool responses, exceeding the 131K–262K context limits of all evaluated models. Every trial fails with HTTP 400 context length exceeded. This is intentional — we do not truncate task inputs to stay close to the upstream evaluation design. These tasks are excluded from the scores above and will remain at 0 for all models until larger context windows are available.



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
