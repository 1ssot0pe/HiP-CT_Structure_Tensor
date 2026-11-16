"""Static import check for the whole repository, without running anything.

Stage scripts reach each other and `core` through sys.path, so a renamed or
deleted function only fails when that one script is next run. This catches it
up front: every `from core.X import a, b` and every import of a repo-local
module must resolve to a file that defines those names at top level.

    python tools/check_imports.py .        # exit 1 on any problem
"""
import ast
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
files = [root / p for p in subprocess.run(
    ["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True,
    check=True).stdout.split("\n") if p and (root / p).exists()]

# module name -> file. Dotted for packages (core.x), bare for script folders.
mods = {}
for f in files:
    rel = f.relative_to(root).with_suffix("")
    parts = rel.parts
    if parts[0] == "core":
        mods[".".join(parts)] = f
        if parts[-1] == "__init__":
            mods["core"] = f
    else:
        mods.setdefault(parts[-1], f)


def top_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names, star = set(), False

    def visit(body):
        nonlocal star
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tg:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    if a.name == "*":
                        star = True
                    names.add(a.asname or a.name.split(".")[0])
            for fld in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(n, fld, None)
                if isinstance(sub, list) and not isinstance(
                        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(sub)
    visit(tree.body)
    return names, star


cache, problems, checked = {}, [], 0
for f in files:
    try:
        tree = ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        problems.append(f"{f.relative_to(root)}: SYNTAX {e}")
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            target = mods.get(n.module)
            if target is None:
                if n.module.startswith("core"):
                    problems.append(f"{f.relative_to(root)}:{n.lineno}: no module {n.module}")
                continue
            if target not in cache:
                cache[target] = top_names(target)
            defined, star = cache[target]
            for a in n.names:
                checked += 1
                sub = f"{n.module}.{a.name}"
                if a.name != "*" and not star and a.name not in defined and sub not in mods:
                    problems.append(f"{f.relative_to(root)}:{n.lineno}: "
                                    f"{n.module} has no top-level '{a.name}'")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("core") and a.name not in mods:
                    problems.append(f"{f.relative_to(root)}:{n.lineno}: no module {a.name}")
                checked += 1

print(f"{len(files)} files, {checked} local import names checked, "
      f"{len(problems)} problems")
for p in problems:
    print("  " + p)
sys.exit(1 if problems else 0)
