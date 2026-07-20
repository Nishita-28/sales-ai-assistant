# Restricted Claims -- superseded

This file is no longer read by the Admin page or by `app/claim_checker.py`.

The restricted-claims policy now lives in [`restricted_claims.yaml`](restricted_claims.yaml)
in this same folder, in a structured form (category, trigger keywords,
whether a match is always blocked regardless of retrieved evidence, and a
human-readable note) that the assistant actually loads and enforces at
runtime -- edit it through the Admin page's "Restricted Claims" tab.

The original prose version of this policy, from before that change, is
kept for reference at
[`restricted_claims.superseded.md`](restricted_claims.superseded.md).
