# Travel Agentic RL：面向旅行规划的 Tool-Use Agent 冷启动与强化学习对齐

本项目复现并实现了一个面向旅行规划场景的 **Agentic RL / Tool-Use Agent** 系统。项目以旅行问答、路线规划、天气查询、周边检索、航班/火车票查询等真实工具调用任务为核心，通过 **数据扩展 → 强模型轨迹蒸馏 → 多阶段数据清洗 → SFT 冷启动 → 工具闭环推理 → GRPO 强化学习对齐 → LLM-as-Judge 评估** 的完整流程，训练一个能够自主选择工具、执行多轮工具调用并生成最终旅行方案的模型。

本项目基座模型为 **Qwen3-4B-Instruct-2507**，训练框架采用 **ms-swift**，推理与 RL rollout 阶段使用 **vLLM** 加速生成。核心目标不是让模型简单回答旅行问题，而是让模型具备完整的 Agent 行为能力：判断任务是否需要工具、选择合适工具、生成结构化 tool call、读取工具返回、继续规划下一步，并在信息充分后输出规范的最终答案。

---

## 0. 实验设备与运行环境

本项目实验主要在 AutoDL 云服务器环境中完成，使用单卡高显存 GPU 进行 SFT 冷启动训练、LoRA 合并、模型推理测试以及后续 GRPO / RL 代码复现与实验分析。

### 硬件环境

| 项目 | 配置 |
|---|---|
| 平台 | AutoDL |
| GPU | NVIDIA RTX PRO 6000 |
| GPU 数量 | 1 张 |
| 显存 | 高显存单卡环境 |
| 主要用途 | Qwen3-4B LoRA SFT、模型推理、工具调用测试、vLLM / GRPO 代码复现 |

本项目所使用的数据规模相对较小，SFT 阶段约一千余条高质量 Agent 轨迹样本，因此单张 RTX PRO 6000 已经能够满足 Qwen3-4B-Instruct-2507 的 LoRA SFT 训练需求。实验中未采用多卡分布式训练，而是使用单卡完成主要训练与推理流程。

### 软件环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Linux |
| Python | 3.10 |
| Conda 环境 | `travel_rl` |
| 训练框架 | ms-swift |
| 基座模型 | Qwen3-4B-Instruct-2507 |
| 微调方式 | LoRA SFT |
| 推理框架 | Transformers / vLLM |
| 实验记录 | Weights & Biases |
| 工具接口 | 高德地图 API、Firecrawl API、OpenAI-compatible LLM API |

---

## 1. 项目整体流程

完整实验流程如下：

```text
Seed Queries
    ↓
DeepSeek 数据扩展
    ↓
data/train_expansion.jsonl
    ↓
MiniMax-M2.7 Agent Rollout 蒸馏
    ↓
saved/train_expansion/*.json
    ↓
三级数据清洗
    ├── simple_filter.py       规则过滤
    ├── format_filter.py       格式清洗
    └── quality_filter.py      LLM-as-Judge 质量过滤
    ↓
data/final/sft_train.jsonl
 data/final/sft_val.jsonl
 data/final/rl.jsonl
    ↓
Qwen3-4B LoRA SFT 冷启动
    ↓
LoRA Merge
    ↓
SFT Agent Tool-Loop 推理测试
    ↓
GRPO / Agentic RL 训练
    ↓
最终模型推理与 LLM Judge 对比评估
```

项目完整覆盖了 Agentic RL 中最重要的几个环节：

1. **数据构造**：从少量旅行 query 出发，扩展为更丰富的旅行任务。
2. **轨迹蒸馏**：使用强模型作为 teacher，生成多轮工具调用轨迹。
3. **数据清洗**：规则过滤、格式修复、LLM-as-Judge 三层质量控制。
4. **SFT 冷启动**：让 Qwen 学会基本 tool-use 格式和停止条件。
5. **真实工具推理**：模型输出 `<tool_call>` 后由程序真实调用工具。
6. **GRPO 对齐**：通过奖励函数优化工具调用合理性、答案完整性和格式稳定性。
7. **自动评估**：使用 LLM Judge 对 baseline、SFT、GRPO 模型进行对比。

---

## 2. 项目目录结构

```text
travel_agentic_rl_rebuild/
├── configs/
│   └── sft_lora.yaml
│
├── data/
│   ├── train_seed.jsonl
│   ├── train_expansion.jsonl
│   ├── clean/
│   └── final/
│       ├── sft_train.jsonl
│       ├── sft_val.jsonl
│       ├── rl.jsonl
│       └── test_final.jsonl
│
├── data_clean/
│   ├── simple_filter.py
│   ├── format_filter.py
│   ├── quality_filter.py
│   └── build_final_datasets.py
│
├── tools/
│   ├── tool_web_search.py
│   ├── tool_visit.py
│   ├── tool_weather.py
│   ├── tool_poi_search.py
│   ├── tool_around_search.py
│   ├── tool_route_planning.py
│   ├── tool_transport.py
│   └── tool_train_ticket.py
│
├── ms-swift/
│   └── examples/train/grpo/plugin/
│       ├── tooluse_multi_turn_scheduler.py
│       └── tooluse_reward_parser_aligned.py
│
├── scripts/
│   ├── download_qwen.sh
│   ├── train_sft.sh
│   ├── merge_lora.sh
│   ├── infer_sft.sh
│   ├── rollout_rl.sh
│   └── train_rl.sh
│
├── saved/
│   └── output/
│       ├── qwen3_4b_sft_lora/
│       ├── qwen3_4b_sft_merged_2310/
│       └── grpo_parser_aligned_run/
│
├── outputs/
│   ├── output_qwen3_4b_baseline_all.jsonl
│   ├── output_qwen3_4b_sft_2310_all.jsonl
│   └── output_qwen3_4b_grpo_epoch150.jsonl
│
├── prompt.py
├── data_expansion.py
├── data_distill.py
├── run_tool_loop_infer.py
├── llm_judge.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. 环境配置

### 3.1 创建 Conda 环境

```bash
conda create -n travel_rl python=3.10 -y
conda activate travel_rl
```

如果服务器 shell 没有初始化 conda，可以执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate travel_rl
```

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

本项目使用本地源码版 `ms-swift`：

```bash
cd ms-swift
pip install -e .
cd ..
```

如训练时缺少 LLM 相关依赖，可以安装：

```bash
cd ms-swift
pip install -e ".[llm]"
cd ..
```

检查 `swift` 是否可用：

```bash
which swift
swift --help
python -c "import swift; print(swift.__file__)"
```

---

## 4. API 与环境变量

项目依赖多个外部服务，包括 OpenAI-compatible LLM API、Firecrawl、高德地图等。请在项目根目录创建 `.env` 文件：

```bash
cp .env.example .env
```

`.env.example` 示例：

```bash
# 通用 LLM / Judge
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.example.com/v1
LLM_MODEL_ID=deepseek-chat
JUDGE_MODEL_ID=deepseek-chat

# Agent rollout 使用的模型服务
AGENT_API_KEY=your_agent_api_key
AGENT_BASE_URL=https://api.minimax.io/v1
AGENT_MODEL_ID=MiniMax-M2.7

# Firecrawl 搜索
FIRECRAWL_API_KEY=your_firecrawl_key

# 高德地图
AMAP_MAPS_API_KEY=your_amap_key

# W&B
WANDB_PROJECT=travel-agentic-rl
WANDB_MODE=online
```

注意：真实 API Key 不要提交到 GitHub。`.env`、模型权重、训练 checkpoint、原始 rollout 文件均应加入 `.gitignore`。

---

## 5. 工具系统

本项目的 Agent 可以调用以下工具：

| 工具名称 | 文件 | 作用 |
|---|---|---|
| `search` | `tools/tool_web_search.py` | 网页搜索，获取 URL、标题和摘要 |
| `visit` | `tools/tool_visit.py` | 访问网页并抽取页面内容 |
| `weather_search` | `tools/tool_weather.py` | 查询城市天气 |
| `poi_search` | `tools/tool_poi_search.py` | 查询地点 POI 和经纬度 |
| `around_search` | `tools/tool_around_search.py` | 查询某坐标附近的餐厅、景点、商圈等 |
| `route_planning` | `tools/tool_route_planning.py` | 查询两点之间步行、驾车、公交路线 |
| `flights_search` | `tools/tool_transport.py` | 查询航班信息 |
| `train_tickets_search` | `tools/tool_train_ticket.py` | 查询火车票信息 |

Agent 的工具调用格式统一为：

```json
[
  {
    "name": "poi_search",
    "arguments": {
      "keywords": "故宫",
      "city": "北京"
    }
  }
]
```

模型在推理时需要输出：

```text
<tool_call>
[
  {"name": "poi_search", "arguments": {"keywords": "故宫", "city": "北京"}}
]
</tool_call>
```

程序执行工具后，会将结果封装为：

```text
<tool_response>
工具返回结果
</tool_response>
```

模型最终必须输出：

```text
<answer>
最终旅行方案
</answer>
```

---

## 6. 数据构造流程

### 6.1 Seed Query

初始数据位于：

```text
data/train_seed.jsonl
```

每条数据是一个旅行相关问题，例如：

```json
{"query": "北京三日游怎么安排？"}
```

### 6.2 DeepSeek 数据扩展

使用 `data_expansion.py` 对 seed query 做语义扩展，生成更多不同类型的旅行任务：

```bash
python data_expansion.py \
  --input data/train_seed.jsonl \
  --output data/train_expansion.jsonl \
  --model deepseek-chat
```

扩展后的问题覆盖：

- 城市多日游规划
- 周边美食/酒店/景点查询
- 两地交通路线
- 天气相关出行建议
- 航班与火车票查询
- 景点开放时间与门票预约
- 商圈、餐厅、停车场、目的地周边检索

---

## 7. 强模型 Agent Rollout 蒸馏

### 7.1 蒸馏目标

`data_distill.py` 使用强模型 `MiniMax-M2.7` 作为 teacher，结合真实工具环境生成多轮 Agent 轨迹。每条轨迹包含：

```text
system prompt
user question
assistant <think> + <tool_call>
user <tool_response>
assistant <think> + <tool_call>
user <tool_response>
assistant <answer>
```

这些 teacher 轨迹后续用于训练 Qwen3-4B，使其模仿强模型的工具调用策略。

### 7.2 运行数据蒸馏

```bash
python data_distill.py \
  --input data/train_expansion.jsonl \
  --output-dir saved/train_expansion \
  --model MiniMax-M2.7 \
  --max-tool-call 13
```

蒸馏输出为：

```text
saved/train_expansion/*.json
```

每个文件是一条完整 rollout 轨迹，包含：

- `question`
- `answer`
- `messages`
- `prediction`
- `termination`
- `stats`

---

## 8. 数据清洗

Teacher rollout 中存在工具失败、格式异常、无答案、英文输出、工具调用过多等问题，因此项目采用三层清洗流程。

### 8.1 第一层：规则过滤 `simple_filter.py`

作用：过滤明显不可用样本，并合并为 jsonl。

```bash
python data_clean/simple_filter.py \
  --input-dir saved/train_expansion \
  --output data/clean/train.jsonl \
  --max-turns 13 \
  --min-answer-len 100
```

过滤规则包括：

- `termination != answer`
- 没有工具调用
- 工具调用轮数过多
- 最终答案过短
- 输出全英文
- 缺失 messages / answer

清洗统计示例：

```text
input files: 1831
kept: 1708
dropped: 123
```

### 8.2 第二层：格式清洗 `format_filter.py`

作用：规范 tool call 和 answer 格式。

```bash
python data_clean/format_filter.py \
  --input data/clean/train.jsonl \
  --output data/clean/train_clean_openai_toolcalls.jsonl \
  --report data/clean/train_clean_openai_toolcalls_report.json \
  --filter-mode empty_and_invalid
```

主要处理：

- 统一 system prompt
- 规范 `<tool_call>` JSON 格式
- 删除无意义 assistant chatter
- 规范 `<answer>` 位置
- 对齐 tool_call 与 tool_response 数量
- 过滤无最终答案样本

格式清洗后统计示例：

```json
{
  "input_rows": 1708,
  "converted_rows": 1708,
  "filtered_rows": 5,
  "kept_rows": 1703,
  "tool_calls_total": 11292,
  "tool_response_messages": 11292,
  "tool_call_parse_fail_blocks": 0,
  "rows_call_tool_mismatch": 0
}
```

### 8.3 第三层：LLM-as-Judge `quality_filter.py`

作用：使用 LLM 从任务相关性、答案完整性、事实安全性、工具使用合理性、格式质量五个维度评估样本质量。

```bash
python data_clean/quality_filter.py \
  --input data/clean/train_clean_openai_toolcalls.jsonl \
  --output data/clean/train_clean_judged_full.jsonl \
  --summary-output data/clean/train_clean_judged_summary.json \
  --filtered-output data/clean/train_clean_judged_filtered.jsonl \
  --model deepseek-chat \
  --workers 5 \
  --keep-verdicts pass,borderline
```

Judge 输出示例：

```json
{
  "verdict": "pass",
  "overall_score": 8.5,
  "dimension_scores": {
    "task_relevance": 9,
    "completeness": 8,
    "factual_safety": 8,
    "tool_use_reasonableness": 9,
    "format_quality": 9
  },
  "reasons": "答案紧扣问题，工具调用合理，格式规范。"
}
```

最终保留：

```text
pass + borderline = 高质量训练样本
fail = 删除
```

---

## 9. 构造最终训练集

使用 `build_final_datasets.py` 将清洗后的高质量样本切分为 SFT、验证集和 RL 数据：

```bash
python data_clean/build_final_datasets.py \
  --input data/clean/train_clean_judged_filtered.jsonl \
  --output-dir data/final \
  --rl-size 200 \
  --val-ratio 0.04 \
  --seed 42
```

输出：

```text
data/final/sft_train.jsonl
data/final/sft_val.jsonl
data/final/rl.jsonl
data/final/test_final.jsonl
```

数据划分示例：

```text
高质量样本总数：1674
SFT train：约 1416
SFT val：约 58
RL prompts：200
```

---

## 10. SFT 数据格式

SFT 数据是一条条完整 Agent 轨迹，每行是一个 JSON 对象：

```json
{
  "id": "sample_id",
  "conversations": [
    {
      "role": "system",
      "content": "你是一名专业的旅行规划 Agent..."
    },
    {
      "role": "user",
      "content": "吉利汽车即墨东部直营店附近有没有必胜客？怎么走？"
    },
    {
      "role": "assistant",
      "content": "<think>我需要先查询起点位置。</think>\n\n<tool_call>...</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>工具返回结果...</tool_response>"
    },
    {
      "role": "assistant",
      "content": "<answer>附近有必胜客，路线如下...</answer>"
    }
  ]
}
```

训练时模型主要学习 `assistant` 的输出，包括：

- 何时调用工具
- 调用哪个工具
- 如何生成 JSON tool call
- 如何基于 tool response 继续推理
- 何时停止工具调用
- 如何输出 `<answer>`

SFT 本质是行为克隆：模型模仿强模型生成的高质量 Agent 轨迹。

---

## 11. Qwen3-4B LoRA SFT 冷启动

### 11.1 下载基座模型

```bash
bash scripts/download_qwen.sh
```

模型路径：

```text
pretrained/Qwen/Qwen3-4B-Instruct-2507
```

### 11.2 SFT 配置

`scripts/train_sft.sh` 采用 ms-swift 启动 LoRA SFT：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/train_sft.sh
```

核心参数：

```text
model: Qwen3-4B-Instruct-2507
tuner_type: lora
lora_rank: 128
lora_alpha: 256
max_length: 32768
num_train_epochs: 2
learning_rate: 8e-6
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.1
batch_size: 1
gradient_checkpointing: true
torch_dtype: bfloat16
```

训练目标：

```text
用户问题
    ↓
<think>
    ↓
<tool_call>
    ↓
<tool_response>
    ↓
继续工具调用或输出 <answer>
```

### 11.3 训练结果

训练输出目录：

```text
saved/output/qwen3_4b_sft_lora/v0-20260709-234044/
```

最终 checkpoint：

```text
checkpoint-2310
```

验证集 `eval_loss` 从约 `0.97` 持续下降并收敛到约 `0.585`，说明模型稳定学习了 Agent 工具调用轨迹格式。

---

## 12. LoRA 合并

SFT 输出的是 LoRA adapter，需要合并回基座模型：

```bash
ADAPTER_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_lora/v0-20260709-234044/checkpoint-2310 \
OUTPUT_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_merged_2310 \
bash scripts/merge_lora.sh
```

合并后模型目录：

```text
saved/output/qwen3_4b_sft_merged_2310/
```

其中包含完整模型权重、tokenizer 和 config，可以直接用于后续推理、rollout 和 RL。

---

## 13. SFT 工具闭环推理测试

### 13.1 推理脚本关系

```text
scripts/infer_sft.sh
    ↓ 调用
run_tool_loop_infer.py
```

`run_tool_loop_infer.py` 是真正执行 Agent 推理循环的核心程序：

```text
读取测试问题
    ↓
加载 SFT merged model
    ↓
模型生成 <tool_call>
    ↓
解析并执行真实工具
    ↓
返回 <tool_response>
    ↓
模型继续推理
    ↓
直到输出 <answer>
```

`infer_sft.sh` 是启动脚本，用于指定模型路径、测试集路径、工具目录和输出路径。

### 13.2 运行推理

```bash
MODEL_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_merged_2310 \
DATASET_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/data/final/test_final.jsonl \
OUTPUT_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/outputs/output_qwen3_4b_sft_2310_all.jsonl \
NUM_SAMPLES=58 \
bash scripts/infer_sft.sh
```

输出文件每行是一条完整推理轨迹：

```json
{
  "sample_idx": 0,
  "id": "sample_id",
  "status": "answer",
  "tool_calls": 3,
  "final_answer": "最终答案...",
  "final_response": "<answer>最终答案...</answer>",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<tool_call>...</tool_call>"},
    {"role": "user", "content": "<tool_response>...</tool_response>"},
    {"role": "assistant", "content": "<answer>...</answer>"}
  ]
}
```

状态字段说明：

| status | 含义 |
|---|---|
| `answer` | 成功输出最终答案 |
| `max_turns` | 达到最大工具调用轮数仍未结束 |
| `no_tool_no_answer` | 未输出工具调用，也未输出答案 |
| `error` | 工具或程序执行异常 |

---

## 14. GRPO / Agentic RL 训练

SFT 使模型学会工具调用格式，而 GRPO 进一步优化工具调用策略、停止条件和最终答案质量。

### 14.1 RL 阶段整体结构

```text
SFT merged model
    ↓
rollout_rl.sh 启动 vLLM rollout server
    ↓
train_rl.sh 启动 GRPO trainer
    ↓
tooluse_multi_turn_scheduler.py 执行多轮工具交互
    ↓
tooluse_reward_parser_aligned.py 给轨迹打 reward
    ↓
GRPO 更新模型
```

### 14.2 vLLM Rollout Server

`rollout_rl.sh` 启动 vLLM 推理服务：

```bash
bash scripts/rollout_rl.sh
```

核心参数：

```bash
swift rollout \
  --model "${MODEL_PATH}" \
  --external_plugins "${SCHEDULER_PLUGIN}" \
  --multi_turn_scheduler travel_tool_loop \
  --vllm_use_async_engine true \
  --vllm_max_model_len 50000 \
  --vllm_tensor_parallel_size 2 \
  --vllm_gpu_memory_utilization 0.8 \
  --max_turns 13
```

vLLM 的作用是为 GRPO 训练高速生成多条 completion。它本身不执行工具，工具循环由 `tooluse_multi_turn_scheduler.py` 负责。

### 14.3 Multi-Turn Scheduler

`tooluse_multi_turn_scheduler.py` 是 RL 阶段的工具调用调度器，负责：

```text
模型生成 <tool_call>
    ↓
解析 tool call JSON
    ↓
调用 tools/ 下真实工具
    ↓
封装 <tool_response>
    ↓
继续让 vLLM 生成下一轮
    ↓
直到 <answer> 或 max_turns
```

它相当于训练阶段版本的 `run_tool_loop_infer.py`。

### 14.4 Reward Plugin

`tooluse_reward_parser_aligned.py` 是 GRPO 的奖励函数插件，负责对每条 rollout 轨迹打分。

奖励由多个部分组成：

| 奖励项 | 作用 |
|---|---|
| answer tag reward | 是否正确输出 `<answer>` |
| tool schema reward | `<tool_call>` JSON 是否可解析 |
| process reward | 是否合理执行工具流程 |
| stage reward | 是否在正确阶段调用工具或输出答案 |
| efficiency reward | 是否避免重复调用、无效调用和过度调用 |
| LLM Judge reward | 使用大模型评价最终答案质量 |

Reward 采用 curriculum 设计：

```text
前期：更重格式、工具调用可解析性
中期：更重工具流程与停止条件
后期：更重最终答案质量和任务完成度
```

### 14.5 启动 GRPO 训练

```bash
bash scripts/train_rl.sh
```

核心命令：

```bash
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --ref_model "${REF_MODEL_PATH}" \
  --dataset "${RL_DATASET}" \
  --external_plugins "${REWARD_PLUGIN}" "${SCHEDULER_PLUGIN}" \
  --reward_funcs external_parser_aligned_curriculum_reward \
  --use_vllm true \
  --vllm_mode server \
  --multi_turn_scheduler travel_tool_loop
```

GRPO 每次从 `data/final/rl.jsonl` 取一个问题，采样多条候选轨迹，对同组轨迹计算 reward，并根据相对优势更新模型。

---

## 15. 推理与评估

### 15.1 Baseline 推理

使用原始 Qwen3-4B-Instruct 模型进行 Agent 推理：

```bash
MODEL_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/pretrained/Qwen/Qwen3-4B-Instruct-2507 \
DATASET_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/data/final/test_final.jsonl \
OUTPUT_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/outputs/output_qwen3_4b_baseline_all.jsonl \
NUM_SAMPLES=80 \
bash scripts/infer_sft.sh
```

### 15.2 SFT 模型推理

```bash
MODEL_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_merged_2310 \
DATASET_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/data/final/test_final.jsonl \
OUTPUT_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/outputs/output_qwen3_4b_sft_2310_all.jsonl \
NUM_SAMPLES=80 \
bash scripts/infer_sft.sh
```

### 15.3 GRPO 模型推理

```bash
MODEL_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/grpo_parser_aligned_run/checkpoint-150 \
DATASET_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/data/final/test_final.jsonl \
OUTPUT_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/outputs/output_qwen3_4b_grpo_epoch150.jsonl \
NUM_SAMPLES=80 \
bash scripts/infer_sft.sh
```

### 15.4 统计推理成功率

```bash
python - <<'PY'
import json
from collections import Counter

path = "outputs/output_qwen3_4b_sft_2310_all.jsonl"
status = Counter()
tool_calls = []
answer_lens = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        status[row.get("status")] += 1
        tool_calls.append(row.get("tool_calls", 0))
        answer_lens.append(len(row.get("final_answer") or ""))

print("status:", status)
print("avg_tool_calls:", sum(tool_calls) / max(1, len(tool_calls)))
print("avg_answer_len:", sum(answer_lens) / max(1, len(answer_lens)))
PY
```

### 15.5 LLM-as-Judge 对比评估

使用 `llm_judge.py` 对不同模型输出进行成对比较：

```bash
python llm_judge.py \
  --input-a outputs/output_qwen3_4b_baseline_all.jsonl \
  --input-b outputs/output_qwen3_4b_sft_2310_all.jsonl \
  --name-a baseline \
  --name-b sft \
  --output outputs/sft_vs_baseline.txt \
  --model deepseek-chat
```

GRPO 对比：

```bash
python llm_judge.py \
  --input-a outputs/output_qwen3_4b_sft_2310_all.jsonl \
  --input-b outputs/output_qwen3_4b_grpo_epoch150.jsonl \
  --name-a sft \
  --name-b grpo \
  --output outputs/rl_vs_sft.txt \
  --model deepseek-chat
```

---

## 16. 实验结果

实验在 80 条测试样本上进行了多组模型对比，结果如下：

| 对比实验 | 模型 A 平均分 | 模型 B 平均分 | 结论 |
|---|---:|---:|---|
| SFT vs Baseline | Baseline: 6.2263 | SFT: 6.5075 | SFT 优于原始基座模型 |
| GRPO vs SFT | SFT: 6.3131 | GRPO: 6.8519 | GRPO 进一步提升 Agent 任务表现 |
| GRPO vs Baseline | Baseline: 6.1181 | GRPO: 6.9419 | GRPO 相对基座提升明显 |
| GRPO vs Qwen3-14B | Qwen3-14B: 5.8606 | GRPO-4B: 6.8756 | 经过 Agentic RL 的 4B 模型优于未对齐 14B 模型 |

实验说明：

1. **SFT 冷启动有效**：模型初步学会了工具调用格式和旅行规划回答模式。
2. **GRPO 带来进一步提升**：强化学习优化了工具调用合理性、停止条件和最终答案完整度。
3. **Agentic 对齐比单纯增大模型更重要**：经过任务对齐的 4B 模型在该场景下优于未专门训练的 14B 模型。

---

## 17. 关键观察

### 17.1 SFT 阶段效果

SFT 后模型已经具备基本 Agent 能力：

- 能够根据旅行问题主动生成 `<tool_call>`
- 能调用 POI、路线、天气、搜索等工具
- 能基于 `<tool_response>` 继续规划下一步
- 多数样本能最终输出 `<answer>`

但 SFT 仍存在一些典型问题：

- 部分样本工具调用过多
- 部分样本达到 `max_turns`
- 对停止条件掌握不稳定
- 工具失败时有时会重复补查

### 17.2 GRPO 阶段效果

GRPO 主要改善：

- 减少重复工具调用
- 提高最终 answer 成功率
- 提高 tool call JSON 稳定性
- 提高答案完整性和可执行性
- 降低 max_turns 失败率
- 提升工具选择与任务需求的一致性

---

## 18. 常用命令汇总

### 数据清洗

```bash
python data_clean/simple_filter.py \
  --input-dir saved/train_expansion \
  --output data/clean/train.jsonl \
  --max-turns 13 \
  --min-answer-len 100

python data_clean/format_filter.py \
  --input data/clean/train.jsonl \
  --output data/clean/train_clean_openai_toolcalls.jsonl \
  --report data/clean/train_clean_openai_toolcalls_report.json \
  --filter-mode empty_and_invalid

python data_clean/quality_filter.py \
  --input data/clean/train_clean_openai_toolcalls.jsonl \
  --output data/clean/train_clean_judged_full.jsonl \
  --summary-output data/clean/train_clean_judged_summary.json \
  --filtered-output data/clean/train_clean_judged_filtered.jsonl \
  --model deepseek-chat \
  --workers 5
```

### SFT 训练

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/train_sft.sh
```

### LoRA 合并

```bash
ADAPTER_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_lora/v0-20260709-234044/checkpoint-2310 \
OUTPUT_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_merged_2310 \
bash scripts/merge_lora.sh
```

### SFT 推理

```bash
MODEL_DIR=/root/autodl-tmp/travel_agentic_rl_rebuild/saved/output/qwen3_4b_sft_merged_2310 \
DATASET_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/data/final/test_final.jsonl \
OUTPUT_PATH=/root/autodl-tmp/travel_agentic_rl_rebuild/outputs/output_qwen3_4b_sft_2310_all.jsonl \
NUM_SAMPLES=80 \
bash scripts/infer_sft.sh
```

### GRPO Rollout Server

```bash
bash scripts/rollout_rl.sh
```

### GRPO 训练

```bash
bash scripts/train_rl.sh
```

---

## 19. 隐私与 GitHub 提交注意事项

不要提交以下内容：

```text
.env
*.key
saved/output/
pretrained/
saved/train_expansion/
data/clean/
data/final/
outputs/
wandb/
__pycache__/
```

推荐 `.gitignore`：

```gitignore
.env
*.key
__pycache__/
*.pyc
wandb/
pretrained/
saved/
outputs/
data/clean/
data/final/
*.safetensors
*.bin
*.pt
*.pth
```

如果工具返回中包含 API Key，需要在数据清洗阶段进行 mask，避免泄漏。

---

## 20. 项目总结

本项目完整复现了一个旅行规划场景下的 Agentic RL 系统。项目重点不只是训练一个问答模型，而是训练一个能够真实调用工具的 Agent。

最终实现了：

1. 从 seed query 扩展旅行任务。
2. 使用强模型生成多轮工具调用轨迹。
3. 通过规则、格式、LLM Judge 三层清洗构造高质量训练数据。
4. 使用 Qwen3-4B 进行 LoRA SFT 冷启动。
5. 合并 LoRA 并实现真实工具闭环推理。
6. 使用 vLLM rollout server 和 GRPO 进行 Agentic RL 对齐。
7. 通过 LLM-as-Judge 评估 baseline、SFT、GRPO 模型效果。

实验结果表明：

- SFT 能显著提升模型的工具调用格式与任务完成能力。
- GRPO 能进一步提升工具使用合理性和最终答案质量。
- 在特定 Agent 场景下，经过任务对齐的小模型可以超过未对齐的大模型。

该项目展示了一条完整的 Tool-Use Agent 训练路线：

```text
强模型蒸馏轨迹
    ↓
SFT 冷启动学格式与行为
    ↓
真实工具环境 rollout
    ↓
Reward 约束工具使用与答案质量
    ↓
GRPO 强化学习优化 Agent 策略
```

这也是当前大模型 Agent 后训练中非常重要的一类工程范式。
