"""Optional integrations with external scorecard/recourse tooling.

Each submodule imports its dependency lazily behind a clear ``ImportError``
naming the extra; ``import probcal`` never touches any of them.
"""

__all__ = ["optbinning"]
