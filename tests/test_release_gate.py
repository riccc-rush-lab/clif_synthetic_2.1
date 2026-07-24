"""Tests for the release gate (U25, R4/R28).

The gate is the mechanical enforcement of the PhysioNet/Rush DUA blocker: a
public release must be blocked until a human records a compliance
acknowledgment. These tests prove it blocks by default and only passes on a
complete, approved, non-placeholder record.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from release_gate import check_acknowledgment, main, parse_acknowledgment  # noqa: E402

_GOOD = """# Compliance acknowledgment

- Reviewer: Jane Doe, Research Compliance Officer
- Date: 2026-07-24
- Decision: approved
- Scope: demo_output/ n=100 synthetic dataset and tagged public release
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "COMPLIANCE_ACK.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_acknowledgment_blocks_release(tmp_path: Path) -> None:
    problems = check_acknowledgment(tmp_path / "COMPLIANCE_ACK.md")
    assert problems
    assert "not found" in problems[0]


def test_complete_acknowledgment_passes(tmp_path: Path) -> None:
    assert check_acknowledgment(_write(tmp_path, _GOOD)) == []


def test_unfilled_template_blocks_release(tmp_path: Path) -> None:
    template = _GOOD.replace("Jane Doe, Research Compliance Officer", "<full name and role>")
    problems = check_acknowledgment(_write(tmp_path, template))
    assert any("placeholder" in p for p in problems)


def test_non_approved_decision_blocks_release(tmp_path: Path) -> None:
    problems = check_acknowledgment(_write(tmp_path, _GOOD.replace("approved", "rejected")))
    assert any("not 'approved'" in p for p in problems)


def test_missing_field_blocks_release(tmp_path: Path) -> None:
    without_scope = "\n".join(ln for ln in _GOOD.splitlines() if not ln.startswith("- Scope"))
    problems = check_acknowledgment(_write(tmp_path, without_scope))
    assert any("missing required field: scope" in p for p in problems)


def test_malformed_date_blocks_release(tmp_path: Path) -> None:
    problems = check_acknowledgment(_write(tmp_path, _GOOD.replace("2026-07-24", "July 2026")))
    assert any("ISO YYYY-MM-DD" in p for p in problems)


def test_parse_extracts_all_required_fields() -> None:
    fields = parse_acknowledgment(_GOOD)
    assert set(fields) == {"reviewer", "date", "decision", "scope"}
    assert fields["decision"] == "approved"


def test_main_exit_codes(tmp_path: Path) -> None:
    assert main(["--ack", str(tmp_path / "nope.md")]) == 1
    assert main(["--ack", str(_write(tmp_path, _GOOD))]) == 0


def test_repo_release_is_blocked_until_a_real_ack_is_recorded() -> None:
    # The committed repo intentionally ships only the template, so a release is
    # blocked until a human records the real acknowledgment.
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "COMPLIANCE_ACK.template.md").exists()
    problems = check_acknowledgment(repo_root / "COMPLIANCE_ACK.md")
    assert problems, "release gate must block until COMPLIANCE_ACK.md is recorded"
