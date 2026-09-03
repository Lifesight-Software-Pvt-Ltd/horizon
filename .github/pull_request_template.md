## What this changes

<!-- One or two sentences. Link the issue this came from. -->

Closes #

## Why

<!--
For a bug fix: what actually broke, and how you diagnosed it.
For a feature: what it makes possible that was not possible before.
-->

## How you verified it

<!-- What you ran and what it showed --- not what you intended to happen. -->

- [ ] `pytest --cov=horizon_forecaster` passes locally, on Python <!-- version -->
- [ ] `black .` and `flake8` are clean
- [ ] New behaviour has real tests (a bug fix has a test that fails without it)
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`, if a user would notice
- [ ] `docs/` updated, if this changes documented behaviour or a design decision
- [ ] Verified against a real dataset, if this touches feature engineering or
      model behaviour <!-- delete if not applicable -->

## Notes for the reviewer

<!--
Anything you are unsure about, decisions you went back and forth on, or
anything you could not test. CI is triggered manually by a maintainer, so say
if there is a platform or Python version you could not cover.
-->
