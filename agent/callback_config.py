"""Whether this deployment will take callback requests at all.

Ported from `merged_code/agent/callback_config.py`. Its own tiny module, for the
reason the original gave: agent/callback_request.py stays a pure decision module
with no environment access of its own, the same discipline the other guards here
hold themselves to, so the one piece of configuration lives apart from it.

A deploying clinic with nobody to make the calls turns this off rather than
having the agent promise a callback that will never happen.
"""

from __future__ import annotations

import os

# Accepts the spellings of "off" a person actually types in an env file. Unset
# means ON, so an existing deployment behaves as it did before this was added.
CALLBACKS_ENABLED = os.environ.get("CALLBACKS_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
    "disabled",
)
