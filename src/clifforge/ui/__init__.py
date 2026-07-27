"""Optional Streamlit UI for CLIFForge (the Cohort Designer).

The UI is a thin front-end over the same generation pipeline the CLI uses
(``variants.spec_to_pack`` -> ``generate.orchestrator.generate_dataset``); it
introduces no new generation logic. Launch it with ``clif-forge ui`` (requires
the ``ui`` extra: ``pip install "clifforge[ui]"``).
"""
