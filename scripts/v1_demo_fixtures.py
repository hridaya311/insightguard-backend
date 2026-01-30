import json
import urllib.request

BASE_URL = "http://localhost:8000"

FIXTURES = [
    {
        "name": "PASS - coherent narrative + deltas explained",
        "expect_status": 200,
        "payload": {
            "tenant_id": "demo-tenant",
            "baseline": {"period": "Q1-2026", "metrics": {"revenue": 120000000, "gross_margin": 0.62, "operating_margin": 0.28}},
            "projection": {"period": "Q2-2026", "metrics": {"revenue": 138000000, "gross_margin": 0.62, "operating_margin": 0.28}},
            "assumptions": {
                "cost_structure": "mixed",
                "narrative": "Revenue increased due to volume growth and pricing. Operating margin remained flat due to increased investments that offset operating leverage."
            }
        }
    },
    {
        "name": "FAIL - unreconciled deltas (revenue move not explained)",
        "expect_status": 422,
        "payload": {
            "tenant_id": "demo-tenant",
            "baseline": {"period": "Q1-2026", "metrics": {"revenue": 100000000, "gross_margin": 0.62, "operating_margin": 0.28}},
            "projection": {"period": "Q2-2026", "metrics": {"revenue": 120000000, "gross_margin": 0.62, "operating_margin": 0.28}},
            "assumptions": {
                "cost_structure": "mixed",
                "narrative": "We expect continued execution with stable costs and operational focus."
            }
        }
    },
    {
        "name": "FAIL - narrative numeric misalignment (claims margin expansion, OM down)",
        "expect_status": 422,
        "payload": {
            "tenant_id": "demo-tenant",
            "baseline": {"period": "Q1-2026", "metrics": {"revenue": 120000000, "gross_margin": 0.62, "operating_margin": 0.30}},
            "projection": {"period": "Q2-2026", "metrics": {"revenue": 130000000, "gross_margin": 0.62, "operating_margin": 0.29}},
            "assumptions": {
                "cost_structure": "mixed",
                "narrative": "We expect margin expansion driven by operating leverage and cost discipline."
            }
        }
    },
    {
        "name": "FAIL - baseline constraint violation (OM up while GM flat/down, no opex lever)",
        "expect_status": 422,
        "payload": {
            "tenant_id": "demo-tenant",
            "baseline": {"period": "Q1-2026", "metrics": {"revenue": 120000000, "gross_margin": 0.62, "operating_margin": 0.28}},
            "projection": {"period": "Q2-2026", "metrics": {"revenue": 120000000, "gross_margin": 0.61, "operating_margin": 0.29}},
            "assumptions": {
                "cost_structure": "mixed",
                "narrative": "We expect improved profitability driven by execution and focus."
            }
        }
    },
]

def post_json(path: str, payload: dict):
    url = BASE_URL + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:
        return 0, {"error": str(e)}

def main():
    print("InsightGuard v1 demo fixtures")
    print("=" * 32)

    ok = 0
    for fx in FIXTURES:
        status, body = post_json("/run-verified-v1", fx["payload"])
        passed = (status == fx["expect_status"])
        verdict = body.get("verdict") if isinstance(body, dict) else None
        failures = body.get("failures", []) if isinstance(body, dict) else []
        codes = [f.get("code") for f in failures if isinstance(f, dict)]

        print(f"\n{fx['name']}")
        print(f"  expected HTTP {fx['expect_status']}, got {status} -> {'OK' if passed else 'MISMATCH'}")
        print(f"  verdict: {verdict}")
        if codes:
            print(f"  failure codes: {codes}")
        report = (body.get("report") or {}) if isinstance(body, dict) else {}
        loc = report.get("location")
        if loc:
            print(f"  report: {loc}")

        if passed:
            ok += 1

    print("\n" + "=" * 32)
    print(f"Summary: {ok}/{len(FIXTURES)} fixtures matched expected outcomes")

if __name__ == "__main__":
    main()
