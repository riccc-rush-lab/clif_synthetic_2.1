"""Tests for the release gate (U25, R4/R28).

The gate is the mechanical enforcement of the credentialed-data and Rush DUA blocker: a
public release must be blocked until a human records a compliance
acknowledgment. These tests prove it blocks by default and only passes on a
complete, approved, non-placeholder record.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from release_gate import (  # noqa: E402
    ACK_ENV_VAR,
    check_ack_text,
    main,
    parse_acknowledgment,
)

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


def test_missing_acknowledgment_blocks_release(tmp_path: Path, monkeypatch) -> None:
    # No file and no env var → the default (no --skip-if-absent) blocks.
    monkeypatch.delenv(ACK_ENV_VAR, raising=False)
    assert main(["--ack", str(tmp_path / "nope.md")]) == 1


def test_complete_acknowledgment_passes() -> None:
    assert check_ack_text(_GOOD) == []


def test_unfilled_template_blocks_release() -> None:
    template = _GOOD.replace("Jane Doe, Research Compliance Officer", "<full name and role>")
    assert any("placeholder" in p for p in check_ack_text(template))


def test_non_approved_decision_blocks_release() -> None:
    assert any("not 'approved'" in p for p in check_ack_text(_GOOD.replace("approved", "rejected")))


def test_missing_field_blocks_release() -> None:
    without_scope = "\n".join(ln for ln in _GOOD.splitlines() if not ln.startswith("- Scope"))
    assert any("missing required field: scope" in p for p in check_ack_text(without_scope))


def test_malformed_date_blocks_release() -> None:
    problems = check_ack_text(_GOOD.replace("2026-07-24", "July 2026"))
    assert any("ISO YYYY-MM-DD" in p for p in problems)


def test_parse_extracts_all_required_fields() -> None:
    fields = parse_acknowledgment(_GOOD)
    assert set(fields) == {"reviewer", "date", "decision", "scope"}
    assert fields["decision"] == "approved"


def test_main_exit_codes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ACK_ENV_VAR, raising=False)
    assert main(["--ack", str(tmp_path / "nope.md")]) == 1
    assert main(["--ack", str(_write(tmp_path, _GOOD))]) == 0


def test_skip_if_absent_passes_when_no_ack_available(tmp_path: Path, monkeypatch) -> None:
    # CI path: no local file, no secret → the gate skips (exit 0) so it does not
    # block a release it cannot verify; the local run stays authoritative.
    monkeypatch.delenv(ACK_ENV_VAR, raising=False)
    assert main(["--ack", str(tmp_path / "nope.md"), "--skip-if-absent"]) == 0


def test_env_var_supplies_acknowledgment(tmp_path: Path, monkeypatch) -> None:
    # CI enforcement path: a repo secret provides the acknowledgment via env var.
    monkeypatch.setenv(ACK_ENV_VAR, _GOOD)
    assert main(["--ack", str(tmp_path / "nope.md"), "--skip-if-absent"]) == 0
    # A present-but-invalid secret still blocks, even with --skip-if-absent.
    monkeypatch.setenv(ACK_ENV_VAR, _GOOD.replace("approved", "rejected"))
    assert main(["--ack", str(tmp_path / "nope.md"), "--skip-if-absent"]) == 1


def test_repo_ships_only_a_placeholder_template_the_gate_rejects() -> None:
    # The committed repo ships only the template, which the gate must reject as a
    # placeholder. (Checks the template directly rather than the live
    # COMPLIANCE_ACK.md, so a valid *local, uncommitted* approval does not flip
    # this test — the real acknowledgment is deliberately never committed.)
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "COMPLIANCE_ACK.template.md"
    assert template.exists()
    assert check_ack_text(template.read_text(encoding="utf-8")), (
        "the template must be rejected as a placeholder"
    )


def test_recorded_acknowledgment_is_never_committed() -> None:
    # The real acknowledgment may carry reviewer identity and is a local-only
    # record; it must never be tracked in git (enforced via .gitignore).
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "COMPLIANCE_ACK.md"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout.strip() == "", "COMPLIANCE_ACK.md must never be committed"
