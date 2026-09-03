# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| < 1.0 | No |

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting: go to the
repository's **Security** tab and choose *Report a vulnerability*. Please
include the affected version, the Python version and OS, what an attacker could
do with it, and a reproduction if you have one.

We will acknowledge your report and let you know whether we can reproduce it
and what we intend to do. If a fix is needed, we will credit you in the release
notes unless you would rather we did not.

## Known dependency vulnerabilities

`neuralprophet==0.8.0` constrains `torch` and `pytorch-lightning` to versions
carrying 24 known CVEs. This is a consequence of NeuralProphet's own upper
bounds rather than a choice, and it is documented in `CHANGELOG.md` under
"Still deferred". The alternative available today is a NeuralProphet
pre-release with breaking API changes and no coverage against this code, so the
pin stands until a vetted release can be adopted. If your deployment cannot
accept those CVEs, run without NeuralProphet — pass `models=` excluding it, and
the other four models are unaffected.

Please do not file new reports for these specific CVEs.
