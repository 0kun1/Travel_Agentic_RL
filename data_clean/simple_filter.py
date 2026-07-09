# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 第一级规则清洗 + rollout JSON 转 jsonl
# --------------------------------------------

import argparse
import json
import re
from pathlib import Path
from collections import Counter

try:
    from langdetect import detect
except Exception:
    detect = None


API_KEY_PATTERNS = [
    re.compile(r'("key"\s*:\s*")[^"]+(")'),
    re.compile(r"('key'\s*:\s*')[^']+(')"),
]


def mask_api_keys(text: str) -> str:
    """
    清理工具返回里可能泄露的 API key。
    例如 around_search 返回 No POI data available for {'key': 'xxx', ...}
    """
    if not isinstance(text, str):
        text = str(text)

    for pattern in API_KEY_PATTERNS:
        text = pattern.sub(r"\1***MASKED***\2", text)

    return text


def get_assistant_reasoning(message: dict) -> str:
    """
    把 reasoning_details 转成 <think>...</think>。
    如果没有 reasoning_details，则尝试使用 reasoning_content。
    """
    reasoning = ""

    reasoning_details = message.get("reasoning_details")

    if isinstance(reasoning_details, list):
        parts = []
        for item in reasoning_details:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
        reasoning = "\n".join(parts).strip()

    if not reasoning:
        reasoning = str(message.get("reasoning_content", "") or "").strip()

    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n"

    return ""


def is_english(text: str) -> bool:
    if detect is None:
        return False

    try:
        return detect(text) == "en"
    except Exception:
        return False


def get_last_assistant_content(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "") or "")
    return ""


def should_drop(info: dict, max_turns: int, min_answer_len: int) -> tuple[bool, str]:
    """
    第一级简单规则过滤。
    """
    termination = info.get("termination", "")
    if termination != "answer":
        return True, f"bad_termination:{termination}"

    messages = info.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return True, "empty_messages"

    stats = dict(info.get("stats", {}) or {})
    turns = stats.get("turns", 0)
    tool_calls = sum(v for k, v in stats.items() if k != "turns" and isinstance(v, int))

    if turns > max_turns:
        return True, "too_many_turns"

    if tool_calls == 0:
        return True, "no_tool_calls"

    final_text = info.get("prediction") or get_last_assistant_content(messages)

    if not isinstance(final_text, str):
        final_text = str(final_text)

    if "exceeds the limit" in final_text or "reached the maximum" in final_text:
        return True, "limit_warning"

    if len(final_text.strip()) < min_answer_len:
        return True, "too_short"

    if is_english(final_text):
        return True, "english_answer"

    return False, "keep"


def convert_messages_to_conversations(messages: list[dict], max_tool_call_text: str = "13") -> list[dict]:
    """
    将 OpenAI tool calling 格式的 messages 转成项目后续清洗使用的 conversations 格式。

    assistant tool_calls:
        -> <think>...</think>
           <tool_call>[...]</tool_call>

    tool response:
        -> role=user
           <tool_response>...</tool_response>
    """
    conversations = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            content = str(message.get("content", "") or "")
            content = re.sub(
                r"最大可调用\s*\d+\s*轮工具",
                f"最大可调用{max_tool_call_text}轮工具",
                content,
            )
            conversations.append(
                {
                    "role": "system",
                    "content": content,
                }
            )

        elif role == "user":
            conversations.append(
                {
                    "role": "user",
                    "content": str(message.get("content", "") or ""),
                }
            )

        elif role == "assistant":
            reasoning_content = get_assistant_reasoning(message)
            tool_calls = message.get("tool_calls")

            if isinstance(tool_calls, list) and tool_calls:
                content = json.dumps(tool_calls, ensure_ascii=False)
                conversations.append(
                    {
                        "role": "assistant",
                        "content": f"{reasoning_content}<tool_call>\n{content}\n</tool_call>",
                    }
                )
            else:
                content = str(message.get("content", "") or "").strip()
                if content:
                    conversations.append(
                        {
                            "role": "assistant",
                            "content": f"{reasoning_content}{content}",
                        }
                    )

        elif role == "tool":
            tool_content = mask_api_keys(str(message.get("content", "") or ""))
            conversations.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{tool_content}\n</tool_response>",
                }
            )

    return conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="saved/train_expansion")
    parser.add_argument("--output", type=str, default="data/clean/train.jsonl")
    parser.add_argument("--max-turns", type=int, default=13)
    parser.add_argument("--min-answer-len", type=int, default=100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))

    reason_counter = Counter()
    tool_counter = Counter()

    kept = 0
    dropped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for p in files:
            try:
                info = json.load(open(p, encoding="utf-8"))
            except Exception as e:
                dropped += 1
                reason_counter["bad_json"] += 1
                print(f"[bad_json] {p}: {e}")
                continue

            should, reason = should_drop(
                info,
                max_turns=args.max_turns,
                min_answer_len=args.min_answer_len,
            )
            reason_counter[reason] += 1

            if should:
                dropped += 1
                continue

            stats = info.get("stats", {}) or {}
            for k, v in stats.items():
                if k != "turns" and isinstance(v, int):
                    tool_counter[k] += v

            qid = info.get("qid") or p.stem
            messages = info["messages"]

            conversations = convert_messages_to_conversations(
                messages,
                max_tool_call_text=str(args.max_turns),
            )

            row = {
                "id": qid,
                "conversations": conversations,
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print("input files:", len(files))
    print("kept:", kept)
    print("dropped:", dropped)
    print("drop/keep reasons:", reason_counter)
    print("tool calls:", tool_counter)
    print("output:", output_path)


if __name__ == "__main__":
    main()