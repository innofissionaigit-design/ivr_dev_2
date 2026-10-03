"""KCD-454: "no field label, colon or bracket is ever spoken."

A tiny, deliberately dumb check -- the story's own acceptance criterion
is "an automated check fails a build containing a spoken punctuation
artefact", not a particular implementation. Kept separate from
reply_templates.py itself so the same check can run over EVERY reply a
template function can produce, from a test, without needing a parser.
"""

from __future__ import annotations

import ast

# "：" (fullwidth colon) alongside the ASCII one for the same reason
# reply_templates.py itself has already been caught shipping "、" (a
# fullwidth comma) where "," was meant -- a stray fullwidth punctuation
# character is a real, recurring class of mistake in this file, not a
# hypothetical one.
SPOKEN_PUNCTUATION_ARTIFACTS = (":", "：", "[", "]", "{", "}")


def find_artifacts(text: str) -> list[str]:
    return [ch for ch in SPOKEN_PUNCTUATION_ARTIFACTS if ch in text]


def has_spoken_artifact(text: str) -> bool:
    return bool(find_artifacts(text))


def _docstring_line_numbers(tree: ast.Module) -> set[int]:
    """Module/class/function docstrings are prose ABOUT the code, never
    spoken to a caller -- excluded so a colon in an explanatory comment
    ("KCD-454: ...") is not mistaken for a spoken artefact."""
    lines = set()
    candidates = [tree] + [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in candidates:
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            lines.add(body[0].value.lineno)
    return lines


def _regex_argument_ids(tree: ast.Module) -> set[int]:
    """id() of every Constant node passed directly as an argument to
    re.compile/match/search/sub/fullmatch/findall -- a regex CHARACTER
    CLASS like "[A-Za-z]" is source code, never spoken to a caller, and
    "[" / "]" there is not the KCD-454 artefact this module looks for."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("compile", "match", "search", "sub", "fullmatch", "findall", "split"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        ids.add(id(arg))
    return ids


def scan_source_for_artifacts(source: str) -> list[tuple[int, str, str]]:
    """(line_number, offending_characters, the_literal_text) for every
    string literal in `source` -- f-string text segment, plain string, or
    dict value -- containing a spoken punctuation artefact, skipping
    docstrings and regex patterns. Scans SOURCE rather than calling each
    function, so every branch is covered without needing fixture data
    for each one."""
    tree = ast.parse(source)
    skip_lines = _docstring_line_numbers(tree)
    skip_ids = _regex_argument_ids(tree)
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.lineno not in skip_lines
            and id(node) not in skip_ids
        ):
            artifacts = find_artifacts(node.value)
            if artifacts:
                hits.append((node.lineno, "".join(artifacts), node.value))
    return hits
