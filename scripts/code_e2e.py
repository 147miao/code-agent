"""code-agent self E2E: call real /api/ask, validate citation paths/lines."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"
CASES = ROOT / "tests" / "code_e2e_cases.json"


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    client = httpx.Client(timeout=180, trust_env=False)
    repos = client.get(f"{API}/api/repos").json()["repos"]
    if not repos:
        sys.exit("no repo indexed")
    repo_id = repos[0]["repo_id"]
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        resp = client.post(f"{API}/api/ask", json={"question": case["question"], "repo_id": repo_id})
        ms = round((time.perf_counter() - t0) * 1000)
        data = resp.json() if resp.status_code == 200 else {}
        cites = data.get("citations") or []
        valid = 0
        hit = False
        for cit in cites:
            path = cit.get("path") or ""
            start = cit.get("start_line")
            end = cit.get("end_line")
            in_range = isinstance(start, int) and isinstance(end, int) and 1 <= start <= end
            if path and (ROOT / path).exists() and in_range:
                valid += 1
            if any(exp in path for exp in case["expect_paths"]):
                hit = True
        rows.append(
            {
                "q": case["question"],
                "status": resp.status_code,
                "refusal": bool(data.get("refusal")),
                "cites": len(cites),
                "valid": valid,
                "hit": hit,
                "ms": ms,
            }
        )
        label = f"[{len(rows)}/{len(cases)}] {case['question'][:30]}"
        print(f"{label} status={resp.status_code} cites={len(cites)} valid={valid} hit={hit} {ms}ms")
    total_cites = sum(r["cites"] for r in rows)
    valid_cites = sum(r["valid"] for r in rows)
    report = {
        "n": len(rows),
        "refusal_count": sum(1 for r in rows if r["refusal"]),
        "citation_valid_rate": round(valid_cites / max(1, total_cites), 3),
        "expected_path_hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 3),
        "p50_ms": int(statistics.median(r["ms"] for r in rows)),
        "avg_ms": round(statistics.mean(r["ms"] for r in rows)),
    }
    print(json.dumps(report, ensure_ascii=False))
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "code_e2e.md").write_text("# code-agent self E2E\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # noqa: E501
    (out / "code_e2e.json").write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")  # noqa: E501


if __name__ == "__main__":
    main()
