"""KCD-157: numbers spoken at a pace a caller can write down.

A phone number read as ten digits in one unbroken run is what makes a
caller ask twice. People dictate them in chunks, with a beat between
chunks -- "98765 ... 43210". This module owns that chunking, language-
neutrally (each language's own speller turns a chunk into words); the
verbalizers in bn_normalize.py and speech_norm.py join the chunks with a
standalone comma, which every FastPitch checkpoint here renders as a short
prosodic break.

Kept separate so the grouping rule exists in exactly one place and is
table-tested once, not re-derived per language.
"""

from __future__ import annotations

# The separator between spoken groups. Standalone (spaces both sides), not
# attached to a word, so anything that walks the spoken text token by token
# (tests/test_digit_fidelity.py's round trip) simply skips it.
GROUP_SEPARATOR = " , "

_MOBILE_LEN = 10


def phone_groups(digits: str) -> list[str]:
    """A 10-digit Indian mobile number is dictated 5 + 5. Anything else is
    chunked in fours, with the remainder folded into the last group so no
    group is a lonely single digit."""
    if len(digits) == _MOBILE_LEN:
        return [digits[:5], digits[5:]]
    if len(digits) <= 4:
        return [digits]
    groups = [digits[i : i + 4] for i in range(0, len(digits), 4)]
    if len(groups[-1]) < 2 and len(groups) > 1:
        last = groups.pop()
        groups[-1] += last
    return groups


def id_groups(identifier: str) -> list[str]:
    """A confirmation ID such as KCD-20261005-3F9A2B1C is read a segment at
    a time (the hyphens are the natural break), and a long segment is
    halved so no single run exceeds four characters."""
    groups: list[str] = []
    for segment in identifier.split("-"):
        if not segment:
            continue
        if len(segment) <= 4:
            groups.append(segment)
        else:
            groups.extend(segment[i : i + 4] for i in range(0, len(segment), 4))
    return groups


def speak_grouped(groups: list[str], spell) -> str:
    """`spell` turns one group into its spoken words for one language."""
    return GROUP_SEPARATOR.join(spell(g) for g in groups if g)
