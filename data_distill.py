# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 蒸馏推理路径和答案
# --------------------------------------------

import sys
import os
import json
import time
import hashlib
import asyncio
import datetime
from pathlib import Path
from typing import Any

import json_repair
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI
from qwen_agent.agents.fncall_agent import FnCallAgent


# =========================
# 1. 项目路径与环境变量
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


from utils import logging as agent_logging
from tools.tool_web_search import WebSearch
from tools.tool_visit import Visit
from tools.tool_weather import WeatherSearch
from tools.tool_transport import FlightsSearch
from tools.tool_train_ticket import TrainTicketsSearch
from tools.tool_route_planning import RoutePlanning
from tools.tool_poi_search import POISearch
from tools.tool_around_search import AroundSearch
from prompt import AGENTIC_SYSTEM_PROMPT


logger = agent_logging.get_logger(__name__)


# =========================
# 2. 工具函数
# =========================

def today_date() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


def to_plain_dict(obj: Any) -> Any:
    """
    将 OpenAI / Pydantic 返回对象递归转成普通 dict/list。
    这样后续 json.dump 和 tool_calls 解析更稳定。
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()

    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_plain_dict(x) for x in obj]

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return {
                k: to_plain_dict(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    return obj


def extract_answer_from_text(text: str) -> str | None:
    """
    从一整段文本里提取 <answer>...</answer>。
    """
    if not text:
        return None

    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()

    return None


def extract_answer_from_messages(messages: list[dict]) -> str | None:
    """
    从所有 assistant messages 拼接后的文本里提取最终答案。

    为什么需要这个函数？
    因为长答案可能会被模型拆成多轮输出：
    第 N 轮：<answer> 开始，但没有 </answer>
    第 N+1 轮：继续输出，并补上 </answer>

    如果只检查当前一轮 content，就会误判为没有答案。
    """
    assistant_texts = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            assistant_texts.append(content)

    full_text = "\n".join(assistant_texts)

    return extract_answer_from_text(full_text)


def safe_json_loads(text: str) -> Any:
    """
    解析模型返回的 arguments。
    优先用标准 json.loads；
    失败后用 json_repair 尝试修复。
    """
    try:
        return json.loads(text)
    except Exception:
        return json_repair.loads(text)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =========================
# 3. 多轮 Tool-Use Agent
# =========================

class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        max_tokens_per_turn: int | None = None,
        max_llm_calls_per_run: int | None = None,
        max_total_tokens: int | None = None,
    ):
        """
        max_tokens_per_turn:
            每一轮模型最多生成多少 token。
            之前你的答案很长，被截断了，所以这里默认调大一点。

        max_llm_calls_per_run:
            每个问题最多允许调用几轮模型。
            工具调用复杂时可以适当调大。

        max_total_tokens:
            当前粗略上下文限制。
        """
        self.max_tokens_per_turn = max_tokens_per_turn or int(
            os.getenv("MAX_TOKENS_PER_TURN", "8192")
        )
        self.max_llm_calls_per_run = max_llm_calls_per_run or int(
            os.getenv("MAX_LLM_CALLS_PER_RUN", "12")
        )
        self.max_total_tokens = max_total_tokens or int(
            os.getenv("MAX_TOTAL_TOKENS", "80000")
        )

        self.max_total_tokens_before_finishing = int(self.max_total_tokens * 0.8)

        # 真实工具对象
        self.tool_class = [
            Visit(),
            WebSearch(),
            WeatherSearch(),
            FlightsSearch(),
            TrainTicketsSearch(),
            RoutePlanning(),
            POISearch(),
            AroundSearch(),
        ]

        # 给程序执行工具用
        self.tool_map = {tool.name: tool for tool in self.tool_class}

        # 给大模型看的 tools schema
        self.tools = []
        for tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            self.tools.append(
                {
                    "type": "function",
                    "function": tool.function,
                }
            )

    def count_tokens(self, messages: list[dict]) -> int:
        """
        源码里通常会用 tokenizer 精确计算。
        当前复现阶段用粗略估算即可，避免必须下载 tokenizer。
        """
        total = 0

        for msg in messages:
            if isinstance(msg, BaseModel):
                msg = msg.model_dump()

            content = msg.get("content", "")
            total += len(str(content)) // 2

        return total

    async def call_server(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
        max_attempts: int = 2,
    ):
        """
        调用 Agent 模型，例如 MiniMax-M2.7。

        注意：
        有些 OpenAI-compatible 平台支持 max_completion_tokens；
        有些只支持 max_tokens。
        所以这里做了兼容 fallback。
        """
        attempts = 0
        last_error = None

        while attempts < max_attempts:
            try:
                try:
                    completion = await client.chat.completions.create(
                        model=os.environ["AGENT_MODEL_ID"],
                        messages=messages,
                        tools=self.tools,
                        max_completion_tokens=self.max_tokens_per_turn,
                    )
                except Exception as e:
                    error_text = str(e)

                    if "max_completion_tokens" in error_text:
                        completion = await client.chat.completions.create(
                            model=os.environ["AGENT_MODEL_ID"],
                            messages=messages,
                            tools=self.tools,
                            max_tokens=self.max_tokens_per_turn,
                        )
                    else:
                        raise e

                message = completion.choices[0].message
                assert message, "Error: LLM response is empty."

                return completion, message

            except Exception as e:
                attempts += 1
                last_error = e

                logger.warning(
                    f"LLM call_server failed at attempt "
                    f"{attempts}/{max_attempts}: {e}"
                )

                await asyncio.sleep(2)

        raise RuntimeError(
            f"Failed to get response from LLM after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    async def custom_call_tool(self, tool_call: dict, **kwargs) -> str:
        """
        真正执行工具。

        模型返回：
        {
            "name": "weather_search",
            "arguments": {"city": "成都"}
        }

        程序执行：
        await WeatherSearch().call({"city": "成都"})
        """
        tool_name = tool_call["name"]
        tool_args = tool_call.get("arguments", {})

        if tool_name in self.tool_map:
            raw_result = await self.tool_map[tool_name].call(tool_args, **kwargs)
            return str(raw_result)

        return f"Error: Tool {tool_name} not found"

    def parse_tool_calls(self, message_dict: dict) -> list[dict]:
        """
        解析 assistant message 中的 tool_calls。
        """
        raw_tool_calls = message_dict.get("tool_calls") or []
        parsed_tool_calls = []

        for item in raw_tool_calls:
            try:
                item = to_plain_dict(item)

                function = item["function"]
                arguments = function.get("arguments", "{}")

                if isinstance(arguments, str):
                    arguments = safe_json_loads(arguments)

                parsed_tool_calls.append(
                    {
                        "id": item.get("id"),
                        "name": function["name"],
                        "arguments": arguments,
                    }
                )

            except Exception as e:
                logger.warning(f"Parse tool_call failed: {item}, error={e}")

        return parsed_tool_calls

    async def run_agent(
        self,
        data: dict,
        client: AsyncOpenAI,
        save_path: str | None = None,
    ) -> dict:
        """
        核心 ReAct / Tool-Use Loop。

        流程：
        user question
        ↓
        assistant 生成 tool_calls
        ↓
        Python 执行工具
        ↓
        tool result 写回 messages
        ↓
        assistant 继续推理
        ↓
        直到出现完整 <answer>...</answer>
        """
        start_time = time.time()

        qid = data["qid"]
        question = data["question"]
        answer = data.get("answer", "")

        system_prompt = AGENTIC_SYSTEM_PROMPT
        system_prompt = system_prompt.replace("__CURRENT_DATE__", today_date())
        system_prompt = system_prompt.replace(
            "__MAX_TOOL_CALL__",
            str(self.max_llm_calls_per_run),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        stats = {"turns": 0}
        completions = []

        num_llm_calls_available = self.max_llm_calls_per_run
        termination = "answer not found"
        prediction = "No answer found."

        while num_llm_calls_available > 0:
            # 防止单样本卡死
            if time.time() - start_time > 10 * 60:
                termination = "No answer found after 10mins"
                prediction = "No answer found after 10mins"
                break

            stats["turns"] += 1
            num_llm_calls_available -= 1

            logger.info(
                f"[QID={qid}] Round {stats['turns']}, "
                f"remaining llm calls: {num_llm_calls_available}"
            )

            completion, message = await self.call_server(client, messages)
            completions.append(completion)

            message_dict = to_plain_dict(message)

            # 保存 assistant message
            messages.append(message_dict)

            content = message_dict.get("content") or ""
            if content:
                logger.info(f"[ASSISTANT] {content[:1000]}")

            # 关键修复：
            # 不只检查当前 content，而是检查所有 assistant messages 拼接后是否已有完整 answer。
            extracted_answer = extract_answer_from_messages(messages)
            if extracted_answer:
                prediction = extracted_answer
                termination = "answer"
                logger.info(f"[QID={qid}] Found final answer.")
                break

            # 解析并执行工具
            parsed_tool_calls = self.parse_tool_calls(message_dict)

            if parsed_tool_calls:
                tasks = [
                    self.custom_call_tool(tool_call)
                    for tool_call in parsed_tool_calls
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for tool_call, result in zip(parsed_tool_calls, results):
                    if isinstance(result, Exception):
                        result = f"Fetch {tool_call} response failed: {result}"

                    logger.info(f"[TOOL] execute function call: {tool_call}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": str(result),
                        }
                    )

                    tool_name = tool_call["name"]
                    stats[tool_name] = stats.get(tool_name, 0) + 1

            else:
                # 没有 tool_calls，也没有完整 answer。
                # 如果模型已经开始 <answer> 但没结束，下一轮让它继续并闭合。
                assistant_all = "\n".join(
                    (m.get("content") or "")
                    for m in messages
                    if m.get("role") == "assistant"
                )

                if "<answer>" in assistant_all and "</answer>" not in assistant_all:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous final answer was cut off. "
                                "Continue exactly from where you stopped and make sure to end with </answer>. "
                                "Do not call tools anymore."
                            ),
                        }
                    )
                else:
                    # 否则提醒模型如果信息足够就必须给最终答案
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "If you have collected enough information, stop calling tools and provide "
                                "the final answer now. The final answer must be wrapped in "
                                "<answer>...</answer>."
                            ),
                        }
                    )

            # 上下文过长时，强制要求停止工具调用并输出答案
            token_count = self.count_tokens(messages)
            logger.debug(f"[QID={qid}] token count: {token_count}")

            if token_count > self.max_total_tokens_before_finishing:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum context length. "
                            "Stop making tool calls and provide your final answer now. "
                            "The final answer must be wrapped in <answer>...</answer>."
                        ),
                    }
                )

        # 循环结束后再次尝试从完整 messages 里提取答案
        extracted_answer = extract_answer_from_messages(messages)
        if extracted_answer:
            prediction = extracted_answer
            termination = "answer"
        else:
            prediction = "No answer found."
            if num_llm_calls_available <= 0:
                termination = "exceed available llm calls"
            else:
                termination = "answer not found"

        result = {
            "qid": qid,
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "stats": stats,
        }

        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            out_file = save_dir / f"{qid}.json"
            save_json(result, out_file)

            logger.info(f"Result dumped to {out_file}")

        return result

    async def make_trajectory(
        self,
        data: dict | list[dict],
        client: AsyncOpenAI,
        save_path: str | None = None,
    ):
        """
        支持单条数据，也支持多条数据。
        当前版本是顺序跑，稳定优先。
        """
        if isinstance(data, dict):
            return await self.run_agent(data, client, save_path=save_path)

        results = []

        for item in data:
            try:
                result = await self.run_agent(item, client, save_path=save_path)
                results.append(result)

            except Exception as e:
                logger.warning(
                    f"rollout question failed: {item.get('question')}, error={e}"
                )

                fail_result = {
                    "qid": item.get("qid", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "messages": [],
                    "prediction": "",
                    "termination": f"runtime error: {e}",
                    "stats": {},
                }

                results.append(fail_result)

        return results


# =========================
# 4. 外部调用入口
# =========================

async def arun_episode(data, save_path: str | None = None):
    agent = MultiTurnReactAgent()

    client = AsyncOpenAI(
        base_url=os.getenv("AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("AGENT_API_KEY") or os.getenv("OPENAI_API_KEY"),
        timeout=120,
        max_retries=2,
    )

    result = await agent.make_trajectory(
        data=data,
        client=client,
        save_path=save_path,
    )

    return result


# =========================
# 5. 主程序：先单样本测试
# =========================

if __name__ == "__main__":
    # 推荐先跑这个复杂样本，验证 flights/weather/poi/route/search 是否都能正常闭环
    data = {
        "qid": "unit_test_chengdu_full_trip_001",
        "question": (
            "我从新加坡出发，计划在2026年7月12日到2026年7月14日去成都三日游，"
            "一个人出行，打算住在春熙路/太古里附近。"
            "请帮我生成一份完整出行规划。"
            "规划需要包括：出发城市到成都的交通建议、到达成都后的市内交通、"
            "每天上午/下午/晚上的行程安排、每段从A到B的推荐出行方式、"
            "经典景点、美食安排、住宿区域建议、天气提醒、门票/预约提醒、"
            "以及如果下雨或太热的备选方案。"
            "我想兼顾大熊猫基地、武侯祠、锦里、宽窄巷子、文殊院、杜甫草堂、"
            "春熙路/太古里和地道成都美食。"
            "请先调用工具查询天气、关键地点位置、必要路线、出发到成都的交通信息和美食信息，"
            "不要只凭常识回答。最终答案必须放在 <answer>...</answer> 中。"
        ),
    }

    asyncio.run(arun_episode(data, "saved/tmp"))

    # =========================
    # 批量生成示例：单样本成功后再打开
    # =========================
    """
    for filename in ["train_expansion.jsonl", "test.jsonl"]
    # for filename in ["rl_buchong_left_316.jsonl"]:
        start = time.time()
        data = []
        phase = filename.split(".")[0]
        save_path = f"saved/{phase}/"
        with open(f"data/{filename}") as fd:
            for line in fd:
                info = json.loads(line)
                question = info["question"]
                # info["qid"] = hashlib.md5(question.encode('utf-8')).hexdigest()
                data.append(info)
        asyncio.run(arun_episode(data, save_path))
        end = time.time()
        print(f"{filename} rollout cost: ", end - start)
    """