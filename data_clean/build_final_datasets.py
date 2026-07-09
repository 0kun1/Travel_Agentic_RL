# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 构造最终训练数据集
#
# 输入:
#   data/clean/train_clean_judged_filtered.jsonl
#
# 输出:
#   data/final/train_clean_judged_shuffle_filtered.jsonl
#   data/final/sft.jsonl
#   data/final/sft_train.jsonl
#   data/final/sft_val.jsonl
#   data/final/rl.jsonl
# --------------------------------------------

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_unused_fields(row: dict) -> dict:
    """
    quality_filter.py 输出的样本里可能带有 judge 字段。
    judge 字段只用于质量分析，不参与训练，所以这里删除。
    """
    row = dict(row)
    row.pop("judge", None)
    return row


def split_dataset(
    rows: list[dict],
    rl_size: int,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    将总数据切分成：
    - sft_rows
    - sft_train_rows
    - sft_val_rows
    - rl_rows
    """
    if not rows:
        raise ValueError("Input rows is empty.")

    if rl_size <= 0:
        rl_size = max(1, int(len(rows) * 0.15))

    if rl_size >= len(rows):
        raise ValueError(
            f"rl_size={rl_size} must be smaller than total rows={len(rows)}"
        )

    sft_rows = rows[:-rl_size]
    rl_rows = rows[-rl_size:]

    val_size = max(1, int(len(sft_rows) * val_ratio))

    if val_size >= len(sft_rows):
        raise ValueError(
            f"val_size={val_size} must be smaller than sft rows={len(sft_rows)}"
        )

    sft_train_rows = sft_rows[:-val_size]
    sft_val_rows = sft_rows[-val_size:]

    return sft_rows, sft_train_rows, sft_val_rows, rl_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="data/clean/train_clean_judged_filtered.jsonl",
        help="LLM-as-Judge 过滤后的输入数据。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/final",
        help="最终数据集输出目录。",
    )

    parser.add_argument(
        "--rl-size",
        type=int,
        default=200,
        help="切分给 RL/GRPO 的样本数量。",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.04,
        help="SFT 验证集比例。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证每次切分结果一致。",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    rows = [remove_unused_fields(row) for row in rows]

    print(f"input: {input_path}")
    print(f"total rows before shuffle: {len(rows)}")

    random.seed(args.seed)
    random.shuffle(rows)

    shuffle_path = output_dir / "train_clean_judged_shuffle_filtered.jsonl"
    write_jsonl(shuffle_path, rows)

    sft_rows, sft_train_rows, sft_val_rows, rl_rows = split_dataset(
        rows=rows,
        rl_size=args.rl_size,
        val_ratio=args.val_ratio,
    )

    write_jsonl(output_dir / "sft.jsonl", sft_rows)
    write_jsonl(output_dir / "sft_train.jsonl", sft_train_rows)
    write_jsonl(output_dir / "sft_val.jsonl", sft_val_rows)
    write_jsonl(output_dir / "rl.jsonl", rl_rows)

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total": len(rows),
        "sft": len(sft_rows),
        "sft_train": len(sft_train_rows),
        "sft_val": len(sft_val_rows),
        "rl": len(rl_rows),
        "rl_size": args.rl_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "files": {
            "shuffle": str(shuffle_path),
            "sft": str(output_dir / "sft.jsonl"),
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "sft_val": str(output_dir / "sft_val.jsonl"),
            "rl": str(output_dir / "rl.jsonl"),
        },
    }

    summary_path = output_dir / "dataset_split_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()