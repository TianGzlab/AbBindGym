import argparse
import csv
import json
import re
from pathlib import Path


def as_float(value):
    if value is None:
        return None
    return float(value)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def find_metric(data, names):
    lowered = {str(key).lower(): value for key, value in data.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def compute_ranking_score(data):
    ranking_score = find_metric(data, ("ranking_score", "rankingscore"))
    if ranking_score is not None:
        return float(ranking_score)

    iptm = as_float(find_metric(data, ("iptm", "iptM", "ipTM")))
    ptm = as_float(find_metric(data, ("ptm", "pTM")))
    fraction_disordered = as_float(find_metric(data, ("fraction_disordered", "fractionDisordered")))
    has_clash = find_metric(data, ("has_clash", "hasClash"))
    if has_clash is not None:
        has_clash = 1.0 if bool(has_clash) else 0.0

    if None in (iptm, ptm, fraction_disordered, has_clash):
        return None
    return 0.8 * iptm + 0.2 * ptm + 0.5 * fraction_disordered - 100.0 * has_clash


def candidate_id(path, data, mode, regex):
    if mode == "name":
        value = str(data.get("name") or path.stem)
    elif mode == "file":
        value = path.stem
    else:
        value = path.parent.name

    if regex:
        match = re.search(regex, value)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return value


def iter_prediction_files(input_dir):
    for path in Path(input_dir).rglob("*.json"):
        name = path.name.lower()
        if "summary" in name or "confidence" in name or "ranking" in name:
            yield path


def collect_predictions(input_dir, id_mode, regex):
    grouped = {}
    for path in iter_prediction_files(input_dir):
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        ranking_score = compute_ranking_score(data)
        if ranking_score is None:
            continue
        iptm = as_float(find_metric(data, ("iptm", "iptM", "ipTM")) or 0.0)
        ptm = as_float(find_metric(data, ("ptm", "pTM")) or 0.0)
        cid = candidate_id(path, data, id_mode, regex)
        row = {
            "candidate_id": cid,
            "prediction_file": str(path),
            "ranking_score": ranking_score,
            "iptm": iptm,
            "ptm": ptm,
        }
        current = grouped.get(cid)
        if current is None or (ranking_score, iptm) > (current["ranking_score"], current["iptm"]):
            grouped[cid] = row
    return grouped


def write_ranked(rows, output_csv, top_n):
    ranked = sorted(rows, key=lambda row: (row["ranking_score"], row["iptm"]), reverse=True)
    if top_n:
        ranked = ranked[:top_n]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "candidate_id", "ranking_score", "iptm", "ptm", "prediction_file"]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": row["candidate_id"],
                    "ranking_score": f"{row['ranking_score']:.8f}",
                    "iptm": f"{row['iptm']:.8f}",
                    "ptm": f"{row['ptm']:.8f}",
                    "prediction_file": row["prediction_file"],
                }
            )
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Rank AF3 candidate predictions by representative ranking score.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--candidate-id-from", choices=("parent", "name", "file"), default="parent")
    parser.add_argument("--candidate-regex", default=None)
    args = parser.parse_args()

    grouped = collect_predictions(args.input_dir, args.candidate_id_from, args.candidate_regex)
    if not grouped:
        raise ValueError("no AF3 summary/confidence JSON files with ranking metrics were found")

    ranked = write_ranked(grouped.values(), args.output_csv, args.top_n)
    best = ranked[0]
    print(
        f"wrote={len(ranked)} output_csv={args.output_csv} "
        f"best_candidate={best['candidate_id']} best_ranking_score={best['ranking_score']:.4f} "
        f"best_iptm={best['iptm']:.4f}"
    )


if __name__ == "__main__":
    main()
