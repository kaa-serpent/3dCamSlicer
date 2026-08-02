# Security Policy

## Supported versions

RotaryCAM is currently pre-1.0 software. Security fixes are applied to the latest version on
the `main` branch; older revisions are not supported.

## Reporting a vulnerability

Do not open a public issue for a security-sensitive finding. Use GitHub's private
vulnerability reporting form in the repository's **Security** tab so the maintainer can
investigate before disclosure.

Include, when available:

- the affected revision or version;
- a minimal reproduction;
- expected and observed behavior;
- potential CNC, file-processing, or local-system impact;
- suggested mitigations or fixes.

You should receive an initial acknowledgement through GitHub within seven days. Timelines for
validation and remediation depend on severity and reproducibility. Please allow time for a fix
before public disclosure.

## CNC safety reports

Unexpected tool motion, unsafe rapid moves, validation bypasses, incorrect machine limits,
and export from an unverified profile should be treated as security-sensitive safety issues.
Never test a report first on a live machine: reproduce it in simulation and use a dry-run with
appropriate emergency-stop precautions only after review.
