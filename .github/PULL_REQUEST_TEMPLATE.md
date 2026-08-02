## Summary

Describe what changed and why.

## CNC safety impact

- [ ] No CNC safety behavior changes
- [ ] Safety behavior changed and the rationale, evidence, and residual risk are documented

## Validation

- [ ] `uv run pytest`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/rotarycam`
- [ ] `git diff --check`
- [ ] User-facing behavior and public APIs are documented
- [ ] UI changes were checked offscreen and screenshots are attached when useful

## Machine verification

- [ ] This change does not claim generic G-code compatibility with a real controller
- [ ] Any changed machine limits or commands identify the exact verified evidence source
