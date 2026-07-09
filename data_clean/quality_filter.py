# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: LLM-as-Judge 质量打分过滤
# 输入: data/clean/train_clean_openai_toolcalls.jsonl
# 输出:
#   1. train_clean_judged_full.jsonl       每条样本的打分结果
#   2. train_clean_judged_summary.json     总体统计
#   3. train_clean_judged_filtered.jsonl   保留 pass/borderline 的训练数据
# --------------------------------------------

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request, error

import json_repair
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from prompt import DATA_JUDGE_PROMPT


DIMENSION_KEYS = [
    "task_relevance",
    "completeness",
    "factual_safety",
    "tool_use_reasonableness",
    "format_quality",
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


def get_sample_id(row: dict[str, Any]) -> str:
    value = row.get("id")
    if value is None:
        value = row.get("qid")

    return str(value) if value is not None else ""


def extract_question(row: dict[str, Any]) -> str:
    """
    从 conversations 里取第一个普通 user 问题。
    注意：tool_response 也是 user role，但不是真正问题，要跳过。
    """
    if isinstance(row.get("question"), str):
        return row["question"].strip()

    for msg in row.get("conversations", []):
        if msg.get("role") != "user":
            continue

        content = str(msg.get("content", "")).strip()

        if not content:
            continue

        if "<tool_response>" in content:
            continue

        return content

    return ""


def extract_answer(conversations: list[dict[str, Any]]) -> str:
    """
    从最后一个 assistant 的 <answer>...</answer> 中提取最终答案。
    """
    for msg in reversed(conversations):
        if msg.get("role") != "assistant":
            continue

        content = str(msg.get("content", ""))

        if "<answer>" in content and "</answer>" in content:
            return content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()

    return ""


def parse_tool_call_block(block: str) -> list[dict[str, Any]]:
    """
    解析 <tool_call>...</tool_call> 中的 JSON。
    format_filter.py 输出的通常是 list：
    [
      {"name": "poi_search", "arguments": {...}}
    ]
    """
    block = block.strip()

    try:
        parsed = json.loads(block)
    except Exception:
        parsed = json_repair.loads(block)

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    valid_calls = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        name = item.get("name")

        # 兼容旧 OpenAI function 格式
        if not name and "function" in item:
            fn = item.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments", {})
        else:
            args = item.get("arguments", {})

        if not name:
            continue

        valid_calls.append(
            {
                "name": name,
                "arguments": args,
            }
        )

    return valid_calls


def summarize_tool_trajectory(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    给 Judge 的工具轨迹摘要。
    不把完整 tool_response 都塞给 Judge，避免 prompt 太长。
    """
    tool_names = []
    tool_call_messages = []
    failed_tool_messages = 0
    num_tool_response = 0

    fail_markers = [
        "error fetching",
        "failed",
        "timeout",
        "no results found",
        "invalid request",
        "could not be accessed",
        "no poi data available",
        "api response error",
    ]

    for msg in conversations:
        role = msg.get("role")
        content = str(msg.get("content", "") or "")

        if role == "assistant":
            blocks = re.findall(
                r"<tool_call>\s*(.*?)\s*</tool_call>",
                content,
                flags=re.S,
            )

            for block in blocks:
                calls = parse_tool_call_block(block)

                for call in calls:
                    tool_names.append(call["name"])

                tool_call_messages.append(calls)

        elif role == "user" and "<tool_response>" in content:
            num_tool_response += 1

            lowered = content.lower()
            if any(marker in lowered for marker in fail_markers):
                failed_tool_messages += 1

    return {
        "tool_names": tool_names,
        "num_tool_call": len(tool_names),
        "num_tool_response": num_tool_response,
        "unique_tool_count": len(set(tool_names)),
        "first_tool": tool_names[0] if tool_names else "",
        "failed_tool_messages": failed_tool_messages,
        "tool_call_path": tool_call_messages[:20],
    }


def build_prompt(question: str, answer: str, trajectory: dict[str, Any]) -> str:
    trajectory_str = json.dumps(trajectory, ensure_ascii=False, indent=2)

    return (
        DATA_JUDGE_PROMPT
        .replace("__QUESTION__", question)
        .replace("__ANSWER__", answer)
        .replace("__TRAJECTORY__", trajectory_str)
    )


def extract_json_object(text: str) -> str:
    """
    从模型输出中抽取 JSON 对象。
    """
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    left = text.find("{")
    right = text.rfind("}")

    if left >= 0 and right >= left:
        return text[left:right + 1]

    return text


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    """
    使用 OpenAI-compatible / DeepSeek 风格接口调用 Judge 模型。
    """
    base_url = base_url.rstrip("/")

    if base_url.endswith("/v1"):
        url = base_url + "/chat/completions"
    else:
        url = base_url + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict data quality judge. Return JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 2048,
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = ""

    for attempt in range(retries):
        try:
            req = request.Request(
                url=url,
                data=body,
                headers=headers,
                method="POST",
            )

            with request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")

            raw_json = json.loads(raw_body)

            content = (
                raw_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )

            json_text = extract_json_object(content)

            try:
                judge = json.loads(json_text)
            except Exception:
                judge = json_repair.loads(json_text)

            return {
                "ok": True,
                "judge": judge,
                "raw_content": content,
            }

        except error.HTTPError as exc:
            last_error = exc.read().decode("utf-8", errors="replace")

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries - 1:
            time.sleep(1 + attempt)

    return {
        "ok": False,
        "error": last_error,
    }


def normalize_judge(judge: dict[str, Any]) -> dict[str, Any]:
    """
    兜底规范 Judge 输出，防止模型漏字段。
    """
    if not isinstance(judge, dict):
        judge = {}

    verdict = str(judge.get("verdict", "fail")).lower()

    if verdict not in {"pass", "borderline", "fail"}:
        verdict = "fail"

    try:
        overall_score = float(judge.get("overall_score", 0))
    except Exception:
        overall_score = 0.0

    dimension_scores = judge.get("dimension_scores", {})

    if not isinstance(dimension_scores, dict):
        dimension_scores = {}

    fixed_dims = {}

    for key in DIMENSION_KEYS:
        try:
            fixed_dims[key] = float(dimension_scores.get(key, 0))
        except Exception:
            fixed_dims[key] = 0.0

    reasons = judge.get("reasons", "")

    return {
        "verdict": verdict,
        "overall_score": overall_score,
        "dimension_scores": fixed_dims,
        "reasons": reasons,
    }


def judge_one(
    row: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    conversations = row.get("conversations", [])
    sample_id = get_sample_id(row)
    question = extract_question(row)
    answer = extract_answer(conversations)
    trajectory = summarize_tool_trajectory(conversations)

    prompt = build_prompt(
        question=question,
        answer=answer,
        trajectory=trajectory,
    )

    result = call_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        timeout=timeout,
        retries=retries,
    )

    output = {
        "id": sample_id,
        "question": question,
        "tool_trajectory_summary": trajectory,
    }

    if result.get("ok"):
        judge = normalize_judge(result.get("judge", {}))
        output.update(
            {
                "ok": True,
                "judge": judge,
                "raw_content": result.get("raw_content", ""),
            }
        )
    else:
        output.update(
            {
                "ok": False,
                "error": result.get("error", ""),
            }
        )

    return output


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [item for item in results if item.get("ok")]
    failed_results = [item for item in results if not item.get("ok")]

    summary: dict[str, Any] = {
        "total": len(results),
        "success": len(ok_results),
        "failed": len(failed_results),
    }

    if not ok_results:
        return summary

    verdict_counts: dict[str, int] = {}
    overall_scores = []
    dimension_sums = {key: 0.0 for key in DIMENSION_KEYS}

    for item in ok_results:
        judge = item.get("judge", {})
        verdict = str(judge.get("verdict", "unknown"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        try:
            overall_scores.append(float(judge.get("overall_score", 0)))
        except Exception:
            overall_scores.append(0.0)

        dims = judge.get("dimension_scores", {}) or {}

        for key in DIMENSION_KEYS:
            try:
                dimension_sums[key] += float(dims.get(key, 0))
            except Exception:
                dimension_sums[key] += 0.0

    summary["overall_avg"] = round(sum(overall_scores) / len(overall_scores), 4)

    summary["verdict_distribution"] = verdict_counts

    summary["dimension_avg"] = {
        key: round(value / len(ok_results), 4)
        for key, value in dimension_sums.items()
    }

    return summary


def export_filtered(
    input_rows: list[dict[str, Any]],
    judged_results: list[dict[str, Any]],
    output_path: Path,
    keep_verdicts: set[str],
) -> int:
    id_to_row = {
        get_sample_id(row): row
        for row in input_rows
    }

    kept_rows = []

    for item in judged_results:
        if not item.get("ok"):
            continue

        verdict = str(item.get("judge", {}).get("verdict", "")).lower()

        if verdict not in keep_verdicts:
            continue

        row = id_to_row.get(str(item.get("id")))

        if row:
            # 也可以把 judge 结果附加回样本里，方便后续分析
            row_with_judge = dict(row)
            row_with_judge["judge"] = item.get("judge", {})
            kept_rows.append(row_with_judge)

    write_jsonl(output_path, kept_rows)

    return len(kept_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge for cleaned travel Agent training data."
    )

    parser.add_argument(
        "--input",
        default="data/clean/train_clean_openai_toolcalls.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/clean/train_clean_judged_full.jsonl",
    )

    parser.add_argument(
        "--summary-output",
        default="data/clean/train_clean_judged_summary.json",
    )

    parser.add_argument(
        "--filtered-output",
        default="data/clean/train_clean_judged_filtered.jsonl",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
    )

    parser.add_argument(
        "--model",
        default=os.getenv("JUDGE_MODEL_ID", ""),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Judge first N samples only. 0 means all.",
    )

    parser.add_argument(
        "--keep-verdicts",
        default="pass,borderline",
        help="Comma-separated verdicts to keep in filtered output.",
    )

    args = parser.parse_args()

    if not args.base_url:
        raise ValueError("Missing OPENAI_BASE_URL or --base-url.")

    if not args.api_key:
        raise ValueError("Missing OPENAI_API_KEY or --api-key.")

    if not args.model:
        raise ValueError("Missing JUDGE_MODEL_ID or --model.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    filtered_path = Path(args.filtered_output)

    rows = load_jsonl(input_path)

    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    print(f"Loaded rows: {len(rows)}")
    print(f"Judge model: {args.model}")
    print(f"Workers: {args.workers}")

    results = []

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [
            executor.submit(
                judge_one,
                row,
                args.base_url,
                args.api_key,
                args.model,
                args.timeout,
                args.retries,
            )
            for row in rows
        ]

        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(
                    {
                        "ok": False,
                        "error": f"future error: {e}",
                    }
                )

            if idx % 20 == 0 or idx == len(futures):
                print(f"Progress: {idx}/{len(futures)}")

    write_jsonl(output_path, results)

    summary = summarize_results(results)

    keep_verdicts = {
        item.strip().lower()
        for item in args.keep_verdicts.split(",")
        if item.strip()
    }

    kept_count = export_filtered(
        input_rows=rows,
        judged_results=results,
        output_path=filtered_path,
        keep_verdicts=keep_verdicts,
    )

    summary["filtered_kept"] = kept_count
    summary["filtered_output"] = str(filtered_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()