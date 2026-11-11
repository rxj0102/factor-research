.PHONY: reproduce render check test

reproduce:  ## real-price factor study from data/snapshot/, refresh README
	uv run python scripts/real_factor_study.py
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	uv run python -m pytest -q
