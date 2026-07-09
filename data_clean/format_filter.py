# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 格式清洗
# 输入:  data/clean/train.jsonl
# 输出:  data/clean/train_clean_openai_toolcalls.jsonl
# --------------------------------------------

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# 允许从项目根目录导入 prompt.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prompt import COLDSTART_SYSTEM_PROMPT


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


DEFAULT_FAIL_PATTERNS = [
    "Error fetching",
    "Failed to read page",
    "could not be accessed",
    "no information is available",
    "No results found for query",
    "Timeout or error",
    "Invalid request format",
    "Fetch",
    "failed",
]


API_KEY_PATTERNS = [
    re.compile(r'("key"\s*:\s*")[^"]+(")'),
    re.compile(r"('key'\s*:\s*')[^']+(')"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mask_api_keys(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    for pattern in API_KEY_PATTERNS:
        text = pattern.sub(r"\1***MASKED***\2", text)

    return text


def extract_date_and_max_calls(system_content: str) -> tuple[str, str]:
    date_match = re.search(r"当前日期：([^\n]+)", system_content)
    max_match = re.search(r"最大可调用\s*(\d+)\s*轮工具", system_content)

    cur_date = date_match.group(1).strip() if date_match else "__CURRENT_DATE__"
    max_calls = max_match.group(1).strip() if max_match else "13"

    return cur_date, max_calls


def normalize_system_prompt(system_content: str) -> str:
    """
    将蒸馏阶段的 system prompt 统一成冷启动 SFT 使用的 system prompt。
    """
    cur_date, max_calls = extract_date_and_max_calls(system_content)

    return (
        COLDSTART_SYSTEM_PROMPT
        .replace("__CURRENT_DATE__", cur_date)
        .replace("__MAX_TOOL_CALL__", max_calls)
        .strip()
    )


def extract_think(content: str) -> str:
    match = THINK_RE.search(content)

    if not match:
        return ""

    think = match.group(1).strip()

    if not think:
        return ""

    return f"<think>\n{think}\n</think>\n\n"


def clean_answer_content(content: str) -> str:
    """
    只保留 <answer>...</answer> 中的内容。
    """
    match = ANSWER_RE.search(content)

    if not match:
        return ""

    answer = match.group(1).strip()

    if not answer:
        return ""

    return f"<answer>\n{answer}\n</answer>"


def try_json_loads(text: str) -> Any:
    """
    尝试解析 JSON。
    这里不引入 json_repair，避免清洗脚本依赖过多。
    如果前面 simple_filter 正常，这里通常是标准 JSON。
    """
    return json.loads(text)


def normalize_arguments(args: Any) -> Any:
    """
    tool arguments 有时是字符串形式的 JSON：
    "{\"address\": \"xxx\"}"

    这里尽量转成 dict。
    """
    if isinstance(args, str):
        args = args.strip()
        if not args:
            return {}

        try:
            return json.loads(args)
        except Exception:
            return args

    return args


def normalize_one_tool_call(raw_call: dict[str, Any]) -> dict[str, Any] | None:
    """
    兼容两类格式：

    1. OpenAI 原生格式：
       {
         "id": "...",
         "function": {
           "name": "poi_search",
           "arguments": "{\"address\":\"xxx\"}"
         },
         "type": "function"
       }

    2. 简化格式：
       {
         "name": "poi_search",
         "arguments": {"address": "xxx"}
       }

    输出统一为：
       {
         "name": "poi_search",
         "arguments": {"address": "xxx"}
       }
    """
    if not isinstance(raw_call, dict):
        return None

    if "function" in raw_call:
        fn = raw_call.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments", {})
    else:
        name = raw_call.get("name")
        args = raw_call.get("arguments", {})

    if not isinstance(name, str) or not name:
        return None

    args = normalize_arguments(args)

    return {
        "name": name,
        "arguments": args,
    }


def parse_tool_calls_from_content(content: str) -> tuple[list[dict[str, Any]], int]:
    """
    从 <tool_call>...</tool_call> 中解析工具调用。
    """
    all_calls = []
    parse_fail_blocks = 0

    blocks = TOOL_CALL_BLOCK_RE.findall(content)

    for block in blocks:
        block = block.strip()

        try:
            parsed = try_json_loads(block)
        except Exception:
            parse_fail_blocks += 1
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            parse_fail_blocks += 1
            continue

        for raw_call in parsed:
            normalized = normalize_one_tool_call(raw_call)
            if normalized:
                all_calls.append(normalized)

    return all_calls, parse_fail_blocks


def clean_tool_call_content(content: str) -> tuple[str, int, int]:
    """
    将 assistant 中的 tool_call 统一规范成：

    <tool_call>
    [
      {"name": "...", "arguments": {...}}
    ]
    </tool_call>
    """
    think = extract_think(content)

    tool_calls, parse_fail_blocks = parse_tool_calls_from_content(content)

    if not tool_calls:
        return "", 0, parse_fail_blocks

    tool_call_json = json.dumps(tool_calls, ensure_ascii=False, indent=2)

    cleaned = f"{think}<tool_call>\n{tool_call_json}\n</tool_call>"

    return cleaned, len(tool_calls), parse_fail_blocks


def clean_tool_response_content(content: str) -> str:
    """
    规范 tool_response。
    """
    match = TOOL_RESPONSE_RE.search(content)

    if match:
        tool_content = match.group(1).strip()
    else:
        tool_content = content.strip()

    tool_content = mask_api_keys(tool_content)

    return f"<tool_response>\n{tool_content}\n</tool_response>"


def has_empty_tool_response(conversations: list[dict[str, Any]]) -> bool:
    for msg in conversations:
        if msg.get("role") != "user":
            continue

        content = str(msg.get("content", ""))

        if "<tool_response>" not in content:
            continue

        match = TOOL_RESPONSE_RE.search(content)

        if not match:
            return True

        if not match.group(1).strip():
            return True

    return False


def has_failed_tool_response(
    conversations: list[dict[str, Any]],
    fail_patterns: list[str],
) -> bool:
    lower_patterns = [p.lower() for p in fail_patterns]

    for msg in conversations:
        if msg.get("role") != "user":
            continue

        content = str(msg.get("content", "")).lower()

        if "<tool_response>" not in content:
            continue

        if any(pattern.lower() in content for pattern in lower_patterns):
            return True

    return False


def has_invalid_final_answer(conversations: list[dict[str, Any]]) -> bool:
    """
    判断是否缺少有效最终答案。
    """
    for msg in reversed(conversations):
        if msg.get("role") != "assistant":
            continue

        content = str(msg.get("content", ""))

        match = ANSWER_RE.search(content)

        if match and match.group(1).strip():
            return False

    return True


def validate_conversation_order(conversations: list[dict[str, Any]]) -> dict[str, int]:
    """
    检查 tool_call 和 tool_response 数量是否大致匹配。
    """
    result = {
        "tool_call_messages": 0,
        "tool_calls_total": 0,
        "tool_response_messages": 0,
        "rows_call_tool_mismatch": 0,
        "has_answer": 0,
    }

    tool_calls_total = 0
    tool_responses_total = 0

    for msg in conversations:
        role = msg.get("role")
        content = str(msg.get("content", ""))

        if role == "assistant" and "<tool_call>" in content:
            calls, _ = parse_tool_calls_from_content(content)
            result["tool_call_messages"] += 1
            tool_calls_total += len(calls)

        if role == "user" and "<tool_response>" in content:
            result["tool_response_messages"] += 1
            tool_responses_total += 1

        if role == "assistant" and "<answer>" in content and "</answer>" in content:
            result["has_answer"] = 1

    result["tool_calls_total"] = tool_calls_total

    if tool_calls_total != tool_responses_total:
        result["rows_call_tool_mismatch"] = 1

    return result


def convert_and_clean_conversations(
    conversations: list[dict[str, Any]],
    normalize_system: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    对单条 conversations 做格式清洗。
    """
    cleaned = []

    stats = {
        "system_normalized": 0,
        "assistant_tool_call_messages": 0,
        "tool_calls_total": 0,
        "tool_response_messages": 0,
        "assistant_answer_messages": 0,
        "deleted_assistant_chatter": 0,
        "tool_call_parse_fail_blocks": 0,
    }

    for msg in conversations:
        role = msg.get("role")
        content = str(msg.get("content", "") or "")

        if role == "system":
            new_content = normalize_system_prompt(content) if normalize_system else content

            if new_content != content:
                stats["system_normalized"] += 1

            cleaned.append(
                {
                    "role": "system",
                    "content": new_content,
                }
            )
            continue

        if role == "user":
            if "<tool_response>" in content:
                cleaned.append(
                    {
                        "role": "user",
                        "content": clean_tool_response_content(content),
                    }
                )
                stats["tool_response_messages"] += 1
            else:
                cleaned.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
            continue

        if role == "assistant":
            # assistant 工具调用轮
            if "<tool_call>" in content:
                cleaned_tool_call, num_calls, parse_fail = clean_tool_call_content(content)
                stats["tool_call_parse_fail_blocks"] += parse_fail

                if cleaned_tool_call and num_calls > 0:
                    cleaned.append(
                        {
                            "role": "assistant",
                            "content": cleaned_tool_call,
                        }
                    )
                    stats["assistant_tool_call_messages"] += 1
                    stats["tool_calls_total"] += num_calls
                else:
                    stats["deleted_assistant_chatter"] += 1

                continue

            # assistant 最终答案轮
            if "<answer>" in content and "</answer>" in content:
                think = extract_think(content)
                answer_content = clean_answer_content(content)

                if answer_content:
                    cleaned.append(
                        {
                            "role": "assistant",
                            "content": f"{think}{answer_content}",
                        }
                    )
                    stats["assistant_answer_messages"] += 1
                else:
                    stats["deleted_assistant_chatter"] += 1

                continue

            # assistant 废话轮：没有 tool_call，也没有 answer
            stats["deleted_assistant_chatter"] += 1
            continue

    return cleaned, stats


def should_filter_row(
    conversations: list[dict[str, Any]],
    mode: str,
    fail_patterns: list[str],
) -> tuple[bool, dict[str, int]]:
    empty_tool = has_empty_tool_response(conversations)
    failed_tool = has_failed_tool_response(conversations, fail_patterns)
    invalid_answer = has_invalid_final_answer(conversations)

    if mode == "empty_only":
        should_filter = empty_tool
    elif mode == "empty_and_invalid":
        should_filter = empty_tool or invalid_answer
    elif mode == "strict_practical":
        should_filter = empty_tool or failed_tool or invalid_answer
    else:
        raise ValueError(f"Unknown filter mode: {mode}")

    reason = {
        "empty_tool_response": int(empty_tool),
        "failed_tool_response": int(failed_tool),
        "invalid_final_answer": int(invalid_answer),
    }

    return should_filter, reason


def clean_dataset(
    rows: list[dict[str, Any]],
    normalize_system: bool,
    filter_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept_rows = []

    aggregate = {
        "input_rows": len(rows),
        "converted_rows": 0,
        "filtered_rows": 0,
        "kept_rows": 0,

        "system_normalized": 0,
        "assistant_tool_call_messages": 0,
        "tool_calls_total": 0,
        "tool_response_messages": 0,
        "assistant_answer_messages": 0,
        "deleted_assistant_chatter": 0,
        "tool_call_parse_fail_blocks": 0,

        "rows_with_empty_tool_response": 0,
        "rows_with_failed_tool_response": 0,
        "rows_with_invalid_final_answer": 0,
        "rows_call_tool_mismatch": 0,
        "rows_no_answer": 0,
    }

    for row in rows:
        conversations = row.get("conversations", [])

        if not isinstance(conversations, list):
            aggregate["filtered_rows"] += 1
            aggregate["rows_with_invalid_final_answer"] += 1
            continue

        cleaned_conversations, stats = convert_and_clean_conversations(
            conversations,
            normalize_system=normalize_system,
        )

        aggregate["converted_rows"] += 1

        for key in [
            "system_normalized",
            "assistant_tool_call_messages",
            "tool_calls_total",
            "tool_response_messages",
            "assistant_answer_messages",
            "deleted_assistant_chatter",
            "tool_call_parse_fail_blocks",
        ]:
            aggregate[key] += stats.get(key, 0)

        should_filter, reason = should_filter_row(
            cleaned_conversations,
            mode=filter_mode,
            fail_patterns=DEFAULT_FAIL_PATTERNS,
        )

        aggregate["rows_with_empty_tool_response"] += reason["empty_tool_response"]
        aggregate["rows_with_failed_tool_response"] += reason["failed_tool_response"]
        aggregate["rows_with_invalid_final_answer"] += reason["invalid_final_answer"]

        validation = validate_conversation_order(cleaned_conversations)
        aggregate["rows_call_tool_mismatch"] += validation["rows_call_tool_mismatch"]

        if validation["has_answer"] == 0:
            aggregate["rows_no_answer"] += 1

        if should_filter:
            aggregate["filtered_rows"] += 1
            continue

        cleaned_row = {
            "id": row.get("id"),
            "conversations": cleaned_conversations,
        }

        kept_rows.append(cleaned_row)

    aggregate["kept_rows"] = len(kept_rows)

    report = {
        "stats": aggregate,
        "filter_mode": filter_mode,
        "fail_patterns": DEFAULT_FAIL_PATTERNS,
    }

    return kept_rows, report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="data/clean/train.jsonl",
        help="Input jsonl path.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/clean/train_clean_openai_toolcalls.jsonl",
        help="Output jsonl path.",
    )

    parser.add_argument(
        "--report",
        type=str,
        default="data/clean/train_clean_openai_toolcalls_report.json",
        help="Report json path.",
    )

    parser.add_argument(
        "--filter-mode",
        type=str,
        default="empty_and_invalid",
        choices=["empty_only", "empty_and_invalid", "strict_practical"],
        help="Filtering mode.",
    )

    parser.add_argument(
        "--keep-original-system",
        action="store_true",
        help="Do not normalize system prompt.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    rows = load_jsonl(input_path)

    kept_rows, report = clean_dataset(
        rows=rows,
        normalize_system=not args.keep_original_system,
        filter_mode=args.filter_mode,
    )

    write_jsonl(output_path, kept_rows)

    full_report = {
        "input": str(input_path),
        "output": str(output_path),
        "report": str(report_path),
        **report,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(json.dumps(full_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()