**Status:** beta on PyPI. Until 1.0, an API-breaking change bumps the minor
version; a change that only alters computed numbers may ship in any release.
Both are listed in the changelog with the reasoning and keep an escape hatch
where the old behaviour had legitimate uses. A deprecated symbol warns with a
`DeprecationWarning` for at least one minor release before removal, naming its
replacement. Serialized artifacts carry a stronger promise: every 0.x release
reads schema 1, pinned by golden files in CI.
