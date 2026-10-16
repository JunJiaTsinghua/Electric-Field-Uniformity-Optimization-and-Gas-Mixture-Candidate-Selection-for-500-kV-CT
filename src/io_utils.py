"""Write analysis outputs without changing the supplied project files."""
from pathlib import Path
import json


def output_directory(root, out, name):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or any(out == root / d or (root / d) in out.parents
                          for d in ("data", "models", "config", "src", "docs")):
        raise ValueError("Choose an output directory separate from the supplied source files")
    destination = out / name
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
