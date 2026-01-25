import json, argparse, re
from typing import Dict, Any, List, Tuple

REQ_KEYS = ["executive_summary","key_drivers","risks","recommendations"]

def iter_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_table_categories(prompt: str) -> List[str]:
    cats=[]
    in_table=False
    for line in prompt.splitlines():
        if line.strip().startswith("Variance table"):
            in_table=True
            continue
        if in_table:
            if not line.strip():
                break
            m=re.match(r"-\s*(.*?):\s*actual=", line.strip())
            if m:
                cats.append(m.group(1).strip())
    return cats

def top_mover_categories(prompt: str) -> List[str]:
    cats=[]
    in_movers=False
    for line in prompt.splitlines():
        if line.strip()=="Top movers:":
            in_movers=True
            continue
        if in_movers:
            if not line.strip() or line.strip().startswith("Anomaly candidates"):
                break
            m=re.match(r"-\s*(.*?):\s*variance=", line.strip())
            if m:
                cats.append(m.group(1).strip())
    return cats

def score_one(prompt: str, output_text: str) -> Dict[str, Any]:
    result = {"ok": False, "errors": [], "score": 0}

    # 1) parse JSON
    try:
        out = json.loads(output_text)
    except Exception as e:
        result["errors"].append(f"invalid_json:{e}")
        return result

    # 2) required keys
    missing=[k for k in REQ_KEYS if k not in out]
    if missing:
        result["errors"].append(f"missing_keys:{missing}")
    else:
        result["score"] += 2

    # 3) category grounding
    table_cats=set(extract_table_categories(prompt))
    used=set()

    # from key_drivers list
    kd=out.get("key_drivers", [])
    if isinstance(kd, list):
        for item in kd:
            if isinstance(item, dict) and "category" in item and item["category"]:
                used.add(str(item["category"]))

    # also scan executive_summary text for any table category names (cheap heuristic)
    summary = out.get("executive_summary","")
    if isinstance(summary,str):
        for c in table_cats:
            if c in summary:
                used.add(c)

    extraneous=[c for c in used if c not in table_cats]
    if extraneous:
        result["errors"].append(f"extraneous_categories:{extraneous}")
    else:
        result["score"] += 2

    # 4) mentions top movers
    movers=set(top_mover_categories(prompt)[:5])
    overlap=len(used.intersection(movers))
    if overlap >= 2:
        result["score"] += 2
    else:
        result["errors"].append(f"low_top_mover_overlap:{overlap}")

    result["ok"] = (len(result["errors"]) == 0)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prompts_path", required=True)
    ap.add_argument("--use_target", action="store_true",
                    help="Evaluate the dataset target field as if it were a model output")
    args=ap.parse_args()

    n=0
    ok=0
    scores=[]
    err_counts={}
    for row in iter_jsonl(args.prompts_path):
        n += 1
        prompt=row["prompt"]
        output = row["target"] if args.use_target else row.get("model_output","")
        r = score_one(prompt, output)
        scores.append(r["score"])
        if r["ok"]:
            ok += 1
        else:
            for e in r["errors"]:
                err_counts[e.split(":")[0]] = err_counts.get(e.split(":")[0],0) + 1

    print("Evaluated:", n)
    print("Pass rate:", f"{ok}/{n} ({(ok/n*100 if n else 0):.1f}%)")
    print("Avg score:", sum(scores)/len(scores) if scores else 0)
    if err_counts:
        print("Top errors:")
        for k,v in sorted(err_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print("-", k, v)

if __name__=="__main__":
    main()
