# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x (latest release) | yes |
| earlier | no |

## Reporting a Vulnerability

Please report vulnerabilities privately via GitHub's
[Report a vulnerability](https://github.com/wlazlod/probcal/security/advisories/new)
form — do not open a public issue for security problems. You will get an
acknowledgement within a few days; fixes ship as a patch release of the latest
version. There is no bug bounty.

probcal's runtime surface is deliberately small (numpy only, no network or file
I/O at import or call time), but reports about anything in the package, its
build, or its release pipeline are welcome.
