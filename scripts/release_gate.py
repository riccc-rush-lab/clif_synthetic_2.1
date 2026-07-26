#!/usr/bin/env python3
"""Release gate — block a public release until compliance acknowledgment is recorded.

The parameter pack is fitted over MIMIC-IV-Ext-CLIF, which is governed by a
PhysioNet **credentialed** data use agreement. Fitting locally to an aggregate
pack is permitted; *publishing* a derived artifact (the demo dataset, the pack,
a tagged public release) is gated on a recorded human acknowledgment from
PhysioNet/MIMIC and Rush research compliance.

This script makes that gate mechanical instead of remembered. It exits nonzero
unless a compliance acknowledgment is present and records all required fields with
non-placeholder values. The acknowledgment is sourced, in order, from:

1. the ``--ack`` file (default ``COMPLIANCE_ACK.md`` at the repo root), or
2. the ``CLIFFORGE_COMPLIANCE_ACK`` environment variable (its full text) — used in
   CI, where the file is deliberately absent (it can carry a reviewer's identity
   and is gitignored). Wire a repository **secret** into this env var to enforce
   the gate in CI without committing the acknowledgment.

Because the acknowledgment is intentionally local-only, ``--skip-if-absent`` lets
CI **pass** when neither source is available (no file, no secret): CI cannot
verify a decision it was never given, so the local run — where the file lives — is
authoritative. A local invocation without ``--skip-if-absent`` still fails on a
missing acknowledgment, so publishing from a workstation stays gated.

    python scripts/release_gate.py [--ack PATH] [--skip-if-absent]

Required fields (``Key: value`` lines, case-insensitive keys):

* ``Reviewer`` — who performed the compliance review
* ``Date`` — ISO ``YYYY-MM-DD`` date of the review
* ``Decision`` — must be ``approved`` for the gate to pass
* ``Scope`` — what the acknowledgment covers

A missing file, a missing field, an unparseable date, an unfilled template
placeholder, or any decision other than ``approved`` blocks the release. The
gate deliberately cannot be satisfied by this repository's own automation — the
whole point is that a human records the decision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

#: Environment variable carrying the acknowledgment text (wire a repo secret here
#: to enforce the gate in CI without committing the file).
ACK_ENV_VAR = "CLIFFORGE_COMPLIANCE_ACK"

REQUIRED_FIELDS = ("reviewer", "date", "decision", "scope")
#: Substrings that mean the template was committed without being filled in.
PLACEHOLDER_MARKERS = ("<", "TODO", "TBD", "FILL", "XXX", "PLACEHOLDER")
_FIELD_RE = re.compile(r"^\s*[-*]?\s*\*{0,2}([A-Za-z ]+?)\*{0,2}\s*:\s*(.+?)\s*$")


def parse_acknowledgment(text: str) -> dict[str, str]:
    """Extract ``Key: value`` fields from the acknowledgment file."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = _FIELD_RE.match(line)
        if match:
            key = match.group(1).strip().lower()
            if key in REQUIRED_FIELDS and key not in fields:
                fields[key] = match.group(2).strip()
    return fields


def resolve_ack_text(path: Path) -> str | None:
    """Acknowledgment text from the ``--ack`` file, else the env var, else ``None``."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    env_text = os.environ.get(ACK_ENV_VAR)
    return env_text if env_text and env_text.strip() else None


def check_ack_text(text: str) -> list[str]:
    """Return a list of problems with the acknowledgment text; empty means it passes."""
    fields = parse_acknowledgment(text)

    problems = [f"missing required field: {name}" for name in REQUIRED_FIELDS if name not in fields]
    for name, value in fields.items():
        if not value or any(marker in value.upper() for marker in PLACEHOLDER_MARKERS):
            problems.append(
                f"field {name!r} is unfilled or still a template placeholder: {value!r}"
            )

    if "date" in fields and not any(p.startswith("field 'date'") for p in problems):
        try:
            dt.date.fromisoformat(fields["date"])
        except ValueError:
            problems.append(f"field 'date' is not an ISO YYYY-MM-DD date: {fields['date']!r}")

    if "decision" in fields and fields["decision"].strip().lower() != "approved":
        problems.append(f"decision is {fields['decision']!r}, not 'approved' — release is blocked")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block release without compliance acknowledgment.")
    parser.add_argument("--ack", default="COMPLIANCE_ACK.md", help="Path to the acknowledgment.")
    parser.add_argument(
        "--skip-if-absent",
        action="store_true",
        help="Pass (exit 0) when no acknowledgment is available from the file or the "
        f"{ACK_ENV_VAR} env var — for CI, where the local-only file is absent and no "
        "secret is configured. A present-but-invalid acknowledgment still fails.",
    )
    args = parser.parse_args(argv)

    text = resolve_ack_text(Path(args.ack))
    if text is None:
        if args.skip_if_absent:
            print(
                f"Release gate skipped: no acknowledgment file ({args.ack}) or "
                f"{ACK_ENV_VAR} secret in this environment. The gate is enforced where "
                "the acknowledgment lives (locally, or via the CI secret)."
            )
            return 0
        problems = [
            f"{args.ack} not found (and no {ACK_ENV_VAR} env var). A public release "
            "requires a recorded PhysioNet/MIMIC + Rush compliance acknowledgment."
        ]
    else:
        problems = check_ack_text(text)
    if problems:
        print("RELEASE BLOCKED — compliance acknowledgment is not satisfied:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRecord the PhysioNet/MIMIC + Rush review in COMPLIANCE_ACK.md "
            "(see COMPLIANCE_ACK.template.md) before publishing.",
            file=sys.stderr,
        )
        return 1
    print("Release gate passed: compliance acknowledgment is recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
