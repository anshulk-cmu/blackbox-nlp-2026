"""Build the ground-truth addition dataset: all (a,b) in [0,99]^2 -> 10,000 rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

MIN_OPERAND = 0
MAX_OPERAND = 99


def count_carries(a, b):
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        col = (a % 10) + (b % 10) + carry
        carry = 1 if col >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def build_rows():
    rows = []
    pid = 0
    for a in range(MIN_OPERAND, MAX_OPERAND + 1):
        for b in range(MIN_OPERAND, MAX_OPERAND + 1):
            rows.append({"id": pid, "a": a, "b": b, "sum": a + b, "n_carries": count_carries(a, b)})
            pid += 1
    return rows


def self_check(rows):
    n = (MAX_OPERAND - MIN_OPERAND + 1) ** 2
    assert len(rows) == n, f"expected {n} rows, got {len(rows)}"
    assert len({r["id"] for r in rows}) == n, "ids not unique"
    assert len({(r["a"], r["b"]) for r in rows}) == n, "duplicate (a, b) pairs"
    for r in rows:
        assert r["sum"] == r["a"] + r["b"], f"wrong sum at id {r['id']}"
        assert 0 <= r["n_carries"] <= 2, f"bad carry count at id {r['id']}"


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "a", "b", "sum", "n_carries"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_out = repo_root / "data" / "ground_truth" / "addition_2digit.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    rows = build_rows()
    self_check(rows)
    write_csv(rows, args.out)

    sums = [r["sum"] for r in rows]
    carries = [r["n_carries"] for r in rows]
    carry_dist = {k: carries.count(k) for k in sorted(set(carries))}
    print(f"Wrote {len(rows):,} problems -> {args.out}")
    print(f"  answer range : {min(sums)}..{max(sums)}  carries: {carry_dist}")


if __name__ == "__main__":
    main()
