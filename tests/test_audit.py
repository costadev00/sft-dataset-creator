from __future__ import annotations

import json

from sft_dataset_creator.audit import create_audit_sample, score_audit
from sft_dataset_creator.engine import execute_plan
from sft_dataset_creator.planner import build_plan


def test_audit_sample_and_score(project_config, tmp_path) -> None:
    run_dir = tmp_path / "run"
    plan = build_plan(project_config, run_dir)
    execute_plan(plan, project_config, run_dir=run_dir)
    review_path = create_audit_sample(run_dir, size=3, seed=9)
    rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all("system_verdict" not in row for row in rows)
    for row in rows:
        row["human_verdict"] = "accept"
    review_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = score_audit(run_dir)
    assert report["sample_size"] == 3
    assert report["human_accept_rate"] == 1.0
    assert report["meets_reject_recall_target"] is True
