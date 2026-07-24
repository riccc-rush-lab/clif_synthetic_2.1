# Compliance acknowledgment (template)

Copy this file to `COMPLIANCE_ACK.md` and fill every field before publishing a
release. `scripts/release_gate.py` blocks tagging/publishing until the real file
exists with all fields completed and `Decision: approved`.

**This must be completed by a human who actually performed the review.** It is a
compliance record, not a formality — do not pre-fill it, and do not let tooling
generate it.

## Why this gate exists

CLIFForge's parameter pack is fitted over MIMIC-IV-Ext-CLIF, governed by a
PhysioNet **credentialed** data use agreement. Fitting locally to an aggregate
pack is permitted. **Publishing** a derived artifact — the demo dataset, the
parameter pack, or a tagged public release — requires confirmed PhysioNet/MIMIC
and Rush research-compliance sign-off.

## Record

- Reviewer: <full name and role of the person who performed the review>
- Date: <YYYY-MM-DD>
- Decision: <approved | rejected>
- Scope: <exactly what is cleared for publication, e.g. "demo_output/ n=100 synthetic dataset and tagged public release; parameter pack NOT included">

## Review notes

<Summarize what was checked: that no row-level real record is present in the
published artifact, that the fit stage emitted aggregate parameters only, that
the minimum-cell-count suppression held, and the PhysioNet/Rush determination
relied upon.>
