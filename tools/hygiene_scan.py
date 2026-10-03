"""Static hygiene scan for the mistakes that hide in plain sight in tuples, lists and defaults.

    python tools/hygiene_scan.py            # report; exit 1 if anything is found

Every rule is a real Python trap, not a style preference:

  mutable-default        def f(x=[]) / def f(x={}) / def f(x=set()): one object shared by every call.
  dataclass-mutable      a dataclass field defaulting to a list/dict/set: the same trap (a ValueError at import
                         for the literal forms, silent sharing for a call like dict()).
  str-in-str             `x in ("abc")` or `x in ("a" "b")`: the parentheses do not make a tuple, so it is a
                         substring test ("b" in ("abc") is True). A one-element tuple needs a trailing comma.
  startswith-list        s.startswith([...]) / .endswith([...]) : a TypeError at run time (it takes a str or a tuple).
  isinstance-list        isinstance(x, [A, B]) : a TypeError (it takes a tuple).
  implicit-concat-item   a list/tuple/set display with adjacent string literals and no comma ("a" "b"): two items
                         silently fused into one; reported only when a display mixes such a fusion with commas.
  tuple-of-one-str       a module-level constant `X = ("abc")` : a str, not a one-element tuple.
  pydantic-mutable       a pydantic/BaseModel field defaulting to a bare [] / {} (pydantic copies it, but the
                         intent is clearer and safer as Field(default_factory=...)).

Run over every .py file in the repository (tests included) except vendored/cache directories.
"""

from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    "static",
}


def _parenthesised(lines: list[str], node: ast.Constant) -> bool:
    """True when the string literal `node` is wrapped in parentheses on its own line: `("abc")`. In the AST that is a
    plain str, indistinguishable from `"abc"` -- which is exactly the mistake when a tuple was meant."""
    line = lines[node.lineno - 1]
    before, after = (
        line[: node.col_offset].rstrip(),
        line[node.end_col_offset :].lstrip() if node.end_lineno == node.lineno else "",
    )
    return before.endswith("(") and after.startswith(")") and node.end_lineno == node.lineno


def _is_mutable_literal(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "dict", "set", "defaultdict", "deque"}
        and not node.args
        and not node.keywords
    )


def _decorated_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        name = (
            d.func.id
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
            else d.id
            if isinstance(d, ast.Name)
            else d.attr
            if isinstance(d, ast.Attribute)
            else ""
        )
        if name == "dataclass":
            return True
    return False


def _is_pydantic(cls: ast.ClassDef) -> bool:
    return any(
        (isinstance(b, ast.Name) and b.id == "BaseModel") or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
        for b in cls.bases
    )


def scan_source(source: str, path: str = "<src>") -> list[tuple[str, int, str, str]]:
    """[(path, line, rule, detail)] for one file's source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:  # a file that does not parse is itself a finding
        return [(path, e.lineno or 0, "syntax", str(e))]
    lines = source.splitlines()
    out: list[tuple[str, int, str, str]] = []

    def add(node, rule, detail):
        out.append((path, getattr(node, "lineno", 0), rule, detail))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for d in defaults:
                if _is_mutable_literal(d):
                    add(
                        d,
                        "mutable-default",
                        "a mutable default argument is shared by every call",
                    )
        elif isinstance(node, ast.ClassDef):
            dc, pyd = _decorated_dataclass(node), _is_pydantic(node)
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None and _is_mutable_literal(stmt.value):
                    if dc:
                        add(
                            stmt,
                            "dataclass-mutable",
                            f"{ast.unparse(stmt.target)} needs field(default_factory=...)",
                        )
                    elif pyd:
                        add(
                            stmt,
                            "pydantic-mutable",
                            f"{ast.unparse(stmt.target)} needs Field(default_factory=...)",
                        )
        elif isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if (
                    isinstance(op, (ast.In, ast.NotIn))
                    and isinstance(comp, ast.Constant)
                    and isinstance(comp.value, str)
                    and len(comp.value) > 1
                    and _parenthesised(lines, comp)
                ):
                    add(
                        comp,
                        "str-in-str",
                        f"`in ({comp.value!r})` is a substring test, not a one-element tuple (add a comma)",
                    )
        elif isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.isupper()
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and _parenthesised(lines, node.value)
            ):
                add(
                    node,
                    "tuple-of-one-str",
                    f"{node.targets[0].id} = ({node.value.value!r}) is a str, not a tuple (add a comma)",
                )
        elif isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in {"startswith", "endswith"}
                and node.args
                and isinstance(node.args[0], ast.List)
            ):
                add(
                    node,
                    "startswith-list",
                    f".{f.attr}([...]) raises TypeError; pass a tuple",
                )
            if (
                isinstance(f, ast.Name)
                and f.id == "isinstance"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.List)
            ):
                add(
                    node,
                    "isinstance-list",
                    "isinstance(x, [...]) raises TypeError; pass a tuple",
                )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) >= 3:
            # fused literals: an element whose source spans two adjacent string literals (implicit concatenation) in a
            # display that otherwise separates its items with commas -- the classic missing comma.
            for el in node.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str) and el.end_lineno == el.lineno:
                    seg = ast.get_source_segment(source, el) or ""
                    if seg.count('"') >= 4 and seg.startswith('"') and '" "' in seg:
                        add(
                            el,
                            "implicit-concat-item",
                            f"{seg[:50]} looks like two items with a missing comma",
                        )
    return out


def scan_tree(root: str = ROOT) -> list[tuple[str, int, str, str]]:
    found: list[tuple[str, int, str, str]] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(base, name)
                with open(path, encoding="utf-8") as f:
                    found.extend(scan_source(f.read(), os.path.relpath(path, root)))
    return found


if __name__ == "__main__":
    findings = scan_tree()
    for path, line, rule, detail in sorted(findings):
        print(f"{path}:{line}: [{rule}] {detail}")
    print(f"{len(findings)} finding(s)")
    sys.exit(1 if findings else 0)
