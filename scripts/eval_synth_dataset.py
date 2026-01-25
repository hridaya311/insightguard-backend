import json, argparse, statistics
from collections import Counter

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    n = 0
    row_counts = []
    anom_counts = []
    mover_vars = []
    categories = Counter()

    for ex in iter_jsonl(args.path):
        n += 1
        table = ex.get("table", [])
        row_counts.append(len(table))
        for r in table:
            categories[r.get("category","UNKNOWN")] += 1

        anoms = ex.get("features", {}).get("anomaly_candidates", [])
        anom_counts.append(len(anoms))

        movers = ex.get("features", {}).get("top_movers", [])
        for m in movers:
            v = m.get("variance")
            if isinstance(v, (int, float)):
                mover_vars.append(abs(v))

        # basic schema checks
        assert "gold" in ex and "executive_summary" in ex["gold"]
        assert "features" in ex and "top_movers" in ex["features"]

    print("Examples:", n)
    print("Rows/table: min=%d p50=%d p90=%d max=%d" % (
        min(row_counts), int(statistics.median(row_counts)),
        sorted(row_counts)[int(0.9*len(row_counts))-1], max(row_counts)
    ))
    print("Anomaly candidates/table: avg=%.2f max=%d" % (sum(anom_counts)/len(anom_counts), max(anom_counts)))
    if mover_vars:
        mover_vars_sorted = sorted(mover_vars)
        print("Abs variance (top movers): p50=%.2f p90=%.2f max=%.2f" % (
            mover_vars_sorted[len(mover_vars_sorted)//2],
            mover_vars_sorted[int(0.9*len(mover_vars_sorted))-1],
            mover_vars_sorted[-1]
        ))
    topcats = categories.most_common(10)
    print("Top categories:", topcats)

if __name__ == "__main__":
    main()
