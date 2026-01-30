# InsightGuard (v1)

InsightGuard is used right before financial results or projections are approved externally to prevent flawed reasoning from being signed off as fact.

## What it does

Given a baseline period, a projection period, and an assumptions narrative, InsightGuard produces:

- **PASS**: no internal reasoning conflicts detected that justify blocking external approval
- **FAIL**: explicit reasoning conflicts detected; external approval should not proceed until corrected or explicitly justified

InsightGuard is deterministic and fail-closed by design.

## When to use it (moment of truth)

Run InsightGuard immediately before:
- CFO / VP Finance sign-off
- board package finalization
- investor or lender reporting approval
- external forecast / guidance approval

## What a FAIL means (and why it matters)

A FAIL is not “the forecast is wrong.”
A FAIL means the reasoning chain is not coherent enough to approve externally without correcting the issue or attaching an explicit justification.

Common real-world impact of signing off anyway:
- misstatements and retractions
- credibility loss with board / investors
- audit flags and rework
- incorrect downstream decisions (hiring, spend, capital allocation)

InsightGuard makes these reasoning risks explicit, before they become external facts.

## What is verified vs not verified

Verified (v1):
- verification integrity (required inputs present)
- material deltas are explicitly explained in the narrative
- narrative direction matches numeric direction
- baseline constraint consistency
- logical consistency checks (deterministic rules)

Not verified (v1):
- market demand realism
- competitive dynamics
- management intent or strategy
- macroeconomic conditions

## API

### POST /run-verified-v1

- Returns 200 with verdict PASS on success
- Returns 422 with verdict FAIL and failure reasons on verification failure
- Always writes a tenant-isolated job record and verification report to S3

## Run the demo fixtures

Start the API:

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Run fixtures:

python scripts/v1_demo_fixtures.py

Expected: 4/4 fixtures matched expected outcomes
