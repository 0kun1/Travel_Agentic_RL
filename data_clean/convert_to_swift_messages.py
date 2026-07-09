# -*- coding: utf-8 -*-
"""
把 data/final/*.jsonl 中的 conversations 格式
转换成 ms-swift 推荐的 messages 格式。

输入示例:
{
  "id": "...",
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

输出示例:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

import argparse
import json
from pathlib import Path


VALID_ROLES = {"system", "user", "assistant"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_one(row: dict) -> dict:
    messages = row.get("messages") or row.get("conversations")

    if not isinstance(messages, list):
        raise ValueError(f"Missing messages/conversations field in row: {row.keys()}")

    new_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # 兼容 sharegpt 老格式
        if role is None:
            role = msg.get("from")
        if content is None:
            content = msg.get("value")

        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")

        if content is None:
            content = ""

        new_messages.append(
            {
                "role": role,
                "content": str(content),
            }
        )

    return {"messages": new_messages}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 jsonl 文件，例如 data/final/sft_train.jsonl",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 jsonl 文件，例如 data/swift/sft_train.jsonl",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)

    converted = []
    for row in rows:
        converted.append(convert_one(row))

    write_jsonl(output_path, converted)

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"rows: {len(converted)}")


if __name__ == "__main__":
    main()