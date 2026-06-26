"""
The target job description the demo ranks for: the challenge's "Senior AI Engineer,
Founding Team" role.

The signal scorer is tuned to THIS role's must-haves (encoded in src/config.py), so the
demo ranks for this one role honestly rather than implying it adapts to arbitrary JDs -
that would require a different, weaker keyword ranker.
"""
from src.jd import JD_QUERY

DEFAULT_PRESET = "Senior AI Engineer (challenge JD)"
PRESETS = {DEFAULT_PRESET: JD_QUERY}
