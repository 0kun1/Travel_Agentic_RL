# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: SFT / GRPO 模型多轮工具调用推理
#
# 核心流程:
# 1. 读取 test_final.jsonl 中的 system + user
# 2. 模型生成 assistant
# 3. 解析 <tool_call>
# 4. 执行对应工具
# 5. 追加 <tool_response>
# 6. 继续推理直到 <answer>
# --------------------------------------------

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


TOOL_CLASS_MAP = {
    "search": ("tool_web_search.py", "WebSearch"),
    "visit": ("tool_visit.py", "Visit"),
    "poi_search": ("tool_poi_search.py", "POISearch"),
    "weather_search": ("tool_weather.py", "WeatherSearch"),
    "around_search": ("tool_around_search.py", "AroundSearch"),
    "route_planning": ("tool_route_planning.py", "RoutePlanning"),
    "train_tickets_search": ("tool_train_ticket.py", "TrainTicketsSearch"),
    "flights_search": ("tool_transport.py", "FlightsSearch"),
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    m = re.search(pattern, text or "", flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def has_final_answer(text: str) -> bool:
    return bool(re.search(r"<\s*answer\s*>.*?<\s*/\s*answer\s*>", text or "", flags=re.S | re.I))


def has_answer_start(text: str) -> bool:
    return bool(re.search(r"<\s*answer\s*>", text or "", flags=re.I))


def extract_answer_text(text: str) -> str:
    return extract_tag_block(text, "answer")


def has_partial_tool_call(text: str) -> bool:
    text = text or ""
    return "<tool_call>" in text and "</tool_call>" not in text


def safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # 兜底：如果安装了 json_repair，就尝试修复
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        return None


def normalize_tool_calls(payload: Any) -> List[Dict[str, Any]]:
    """
    把各种可能的工具调用格式统一成：
    [{"name": "...", "arguments": {...}}]
    """
    if payload is None:
        return []

    if isinstance(payload, dict):
        if isinstance(payload.get("tool_calls"), list):
            return normalize_tool_calls(payload["tool_calls"])

        if "name" in payload:
            args = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(args, str):
                loaded = safe_json_loads(args)
                args = loaded if isinstance(loaded, dict) else {"raw_arguments": args}
            if not isinstance(args, dict):
                args = {"raw_arguments": str(args)}
            return [{"name": payload.get("name"), "arguments": args}]

        if "function" in payload:
            fn_obj = payload.get("function")
            if isinstance(fn_obj, dict):
                name = fn_obj.get("name")
                args = fn_obj.get("arguments") or fn_obj.get("parameters") or {}
            else:
                name = str(fn_obj)
                args = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(args, str):
                loaded = safe_json_loads(args)
                args = loaded if isinstance(loaded, dict) else {"raw_arguments": args}
            if not isinstance(args, dict):
                args = {"raw_arguments": str(args)}
            return [{"name": name, "arguments": args}]

        # 兼容 {"search": {...}}
        if len(payload) == 1:
            key = next(iter(payload.keys()))
            if key in TOOL_CLASS_MAP:
                val = payload[key]
                return [{"name": key, "arguments": val if isinstance(val, dict) else {"raw_arguments": str(val)}}]

        return []

    if isinstance(payload, list):
        calls = []
        for item in payload:
            calls.extend(normalize_tool_calls(item))
        return calls

    return []


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    从模型输出中解析 <tool_call>...</tool_call>。
    老师源码里也有这个函数，是整个 Agent Loop 的关键。
    """
    text = (text or "").strip()

    blocks = re.findall(r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>", text, flags=re.S | re.I)
    if not blocks:
        return []

    all_calls = []
    for block in blocks:
        payload = safe_json_loads(block)
        all_calls.extend(normalize_tool_calls(payload))

    return all_calls


def extract_call(call: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    name = call.get("name")
    args = call.get("arguments") or {}

    if isinstance(args, str):
        loaded = safe_json_loads(args)
        args = loaded if isinstance(loaded, dict) else {"raw_arguments": args}

    if not isinstance(args, dict):
        args = {"raw_arguments": str(args)}

    return name, args


def load_tool(tools_dir: str, tool_name: str):
    if tool_name not in TOOL_CLASS_MAP:
        return None

    module_rel, cls_name = TOOL_CLASS_MAP[tool_name]
    module_path = Path(tools_dir) / module_rel

    if not module_path.exists():
        raise FileNotFoundError(f"Tool file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"tool_module_{tool_name}", str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    cls = getattr(module, cls_name)
    return cls()


def is_lnglat(text: str) -> bool:
    return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*", text or ""))


def extract_first_lnglat(text: str) -> Optional[str]:
    m = re.search(r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?", text or "")
    if not m:
        return None
    return m.group(0).replace(" ", "")


async def retry_route_with_poi_if_needed(args, call_args: Dict[str, Any]) -> Optional[str]:
    """
    老师源码里有同样逻辑：
    如果 route_planning 的 origin / destination 不是经纬度，
    就先用 poi_search 转成经纬度后再重试路线规划。
    """
    origin = str(call_args.get("origin", ""))
    destination = str(call_args.get("destination", ""))

    if not origin or not destination:
        return None

    if is_lnglat(origin) and is_lnglat(destination):
        return None

    try:
        poi_tool = load_tool(args.tools_dir, "poi_search")
        route_tool = load_tool(args.tools_dir, "route_planning")

        origin_poi, dest_poi = await asyncio.gather(
            poi_tool.call({"address": origin}),
            poi_tool.call({"address": destination}),
        )

        origin_lnglat = extract_first_lnglat(str(origin_poi))
        dest_lnglat = extract_first_lnglat(str(dest_poi))

        if not origin_lnglat or not dest_lnglat:
            return None

        retry_args = dict(call_args)
        retry_args["origin"] = origin_lnglat
        retry_args["destination"] = dest_lnglat

        return await route_tool.call(retry_args)

    except Exception:
        return None


def get_initial_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    conversations = row.get("conversations") or row.get("messages") or []
    if len(conversations) < 2:
        raise ValueError("Each sample needs at least system + user messages.")

    # 推理测试时只取 system + 第一个用户问题，不把标准答案喂给模型
    return [
        {
            "role": conversations[0]["role"],
            "content": conversations[0]["content"],
        },
        {
            "role": conversations[1]["role"],
            "content": conversations[1]["content"],
        },
    ]


def infer_once(model, tokenizer, messages: List[Dict[str, str]], args) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


async def execute_tool_calls(args, tool_calls: List[Dict[str, Any]], sample_idx: int, turn: int) -> str:
    tool_outputs = []

    for i, call in enumerate(tool_calls, start=1):
        fn, call_args = extract_call(call)

        if not fn:
            tool_result = "TOOL_ERROR: missing tool name"

        elif fn not in TOOL_CLASS_MAP:
            tool_result = f"TOOL_ERROR: unsupported tool `{fn}`"

        else:
            try:
                tool = load_tool(args.tools_dir, fn)
                tool_result = await tool.call(call_args)

                if fn == "route_planning":
                    retry_result = await retry_route_with_poi_if_needed(args, call_args)
                    if retry_result:
                        tool_result = retry_result

            except Exception as e:
                tool_result = f"TOOL_ERROR: {type(e).__name__}: {e}"

        tool_outputs.append(
            {
                "tool": fn,
                "arguments": call_args,
                "result": str(tool_result)[: args.tool_response_max_chars],
            }
        )

    return "<tool_response>\n" + json.dumps(tool_outputs, ensure_ascii=False, indent=2) + "\n</tool_response>"


async def run_single_sample(args, model, tokenizer, row: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    messages = get_initial_messages(row)

    total_tool_calls = 0
    status = "max_turns"
    final_answer = ""
    final_response = ""

    saw_tool_response = False
    no_tool_no_answer_retries = 0
    same_tool_call_count = 0
    previous_tool_signature = ""

    for turn in range(1, args.max_turns + 1):
        response = infer_once(model, tokenizer, messages, args)
        final_response = response

        if not args.quiet:
            print(f"\n===== SAMPLE {sample_idx} | TURN {turn} | ASSISTANT =====")
            print(response[: args.print_chars])

        messages.append({"role": "assistant", "content": response})

        # 如果还没工具返回就提前 answer，可以强制它先用工具
        if args.tool_first_enforce and has_answer_start(response) and not saw_tool_response:
            messages.append(
                {
                    "role": "user",
                    "content": "请先通过 <tool_call> 调用至少一个工具，并在收到工具结果后再输出最终 <answer>。",
                }
            )
            continue

        if has_final_answer(response):
            status = "answer"
            final_answer = extract_answer_text(response)
            break

        tool_calls = parse_tool_calls(response)

        if not tool_calls and has_partial_tool_call(response):
            messages.append(
                {
                    "role": "user",
                    "content": "你的工具调用不完整。请只输出完整且可解析的 <tool_call>...</tool_call>。",
                }
            )
            continue

        if not tool_calls:
            no_tool_no_answer_retries += 1
            if no_tool_no_answer_retries <= args.max_no_tool_no_answer_retries:
                if saw_tool_response:
                    messages.append(
                        {
                            "role": "user",
                            "content": "请基于已有工具结果输出最终答案，并严格使用 <answer>...</answer>。",
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "请优先调用合适工具获取事实信息，并严格输出 <tool_call>...</tool_call>。",
                        }
                    )
                continue

            status = "no_tool_no_answer"
            break

        no_tool_no_answer_retries = 0

        current_signature = json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
        if current_signature == previous_tool_signature:
            same_tool_call_count += 1
        else:
            same_tool_call_count = 0
            previous_tool_signature = current_signature

        if same_tool_call_count >= args.max_same_tool_call_rounds:
            messages.append(
                {
                    "role": "user",
                    "content": "不要重复相同工具调用，请基于已有信息输出最终 <answer>...</answer>。",
                }
            )
            continue

        total_tool_calls += len(tool_calls)

        tool_response = await execute_tool_calls(args, tool_calls, sample_idx, turn)
        saw_tool_response = True

        if not args.quiet:
            print(f"\n----- TOOL RESPONSE -----")
            print(tool_response[: args.print_chars])

        messages.append({"role": "user", "content": tool_response})

    return {
        "sample_idx": sample_idx,
        "id": row.get("id"),
        "status": status,
        "tool_calls": total_tool_calls,
        "final_answer": final_answer,
        "final_response": final_response,
        "messages": messages if args.save_full_messages else None,
    }


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


async def main_async(args):
    rows = load_jsonl(args.dataset_path)

    start = args.start_idx
    end = min(len(rows), start + args.num_samples)

    print("========== INFER CONFIG ==========")
    print("model_dir:", args.model_dir)
    print("dataset_path:", args.dataset_path)
    print("tools_dir:", args.tools_dir)
    print("start_idx:", start)
    print("end_idx:", end)
    print("output_path:", args.output_path)
    print("==================================")

    model, tokenizer = load_model_and_tokenizer(args)

    outputs = []
    for idx in range(start, end):
        try:
            result = await run_single_sample(args, model, tokenizer, rows[idx], idx)
        except Exception as e:
            result = {
                "sample_idx": idx,
                "id": rows[idx].get("id"),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }

        outputs.append(result)
        write_jsonl(args.output_path, outputs)

    print(f"[DONE] saved to {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tools_dir", type=str, required=True)

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)

    parser.add_argument("--max_turns", type=int, default=13)
    parser.add_argument("--max_new_tokens", type=int, default=3000)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument("--tool_first_enforce", action="store_true")
    parser.add_argument("--max_no_tool_no_answer_retries", type=int, default=2)
    parser.add_argument("--max_same_tool_call_rounds", type=int, default=3)
    parser.add_argument("--tool_response_max_chars", type=int, default=5000)

    parser.add_argument("--save_full_messages", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print_chars", type=int, default=2000)

    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = str(Path(args.tools_dir).resolve().parent)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()