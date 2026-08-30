# Security Policy

## Supported versions

Only the latest release line receives fixes. A security fix ships as a patch
release of the current minor (for example, a report against 0.3.0 is fixed in
0.3.x); earlier minors are not patched. Check the
[releases page](https://github.com/wlazlod/probcal/releases) or PyPI for the
current version.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's
[Report a vulnerability](https://github.com/wlazlod/probcal/security/advisories/new)
form. Do not open a public issue for security problems. You will get an
acknowledgement within a few days. There is no bug bounty.

## Scope

probcal's runtime depends on numpy and the standard library only, and makes no
network requests. The only file I/O is explicit and caller-directed:
`to_json(path)` / `from_json(path)` on calibrators and monitors, and
`validation_report(path=...)`. Serialized artifacts are plain JSON; the package
never unpickles. Reports about anything in the package, its optional
integrations, its build, or its release pipeline are welcome.
