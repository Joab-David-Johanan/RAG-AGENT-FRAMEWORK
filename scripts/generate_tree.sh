python - <<'EOF'
import os

EXCLUDE_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "node_modules",
    ".next",
}

EXCLUDE_SUFFIXES = (
    ".egg-info",
    ".coverage",
    ".code-workspace",
)

def should_exclude(name: str) -> bool:
    if name in EXCLUDE_NAMES:
        return True
    return any(name.endswith(s) for s in EXCLUDE_SUFFIXES)

def walk(path, prefix=""):
    entries = sorted(
        e for e in os.listdir(path)
        if not should_exclude(e)
    )

    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    ordered = dirs + files

    for i, entry in enumerate(ordered):
        full = os.path.join(path, entry)
        last = i == len(ordered) - 1

        connector = "└── " if last else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(full):
            walk(full, prefix + ("    " if last else "│   "))

project_root = os.path.basename(os.getcwd())

print(project_root)
walk(".")
EOF
