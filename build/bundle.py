#!/usr/bin/env python3
"""Builds a single-file distributable script for Siril's Scripts menu.

Embeds modular packages as clean, readable multi-line Python source blocks
with condensed docstrings so the final bundled file reads like normal Python.
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "dg_patch_tool"
OUTPUT_PATH = REPO_ROOT / "dist" / "DG_Patch_Tool.py"
ROOT_OUTPUT_PATH = REPO_ROOT / "DG_Patch_Tool.py"
EXCLUDE_DIR_NAMES = {"tests", "__pycache__"}

SIRIL_SCRIPTS_DIRS = [
    Path("/Users/daiangan/siril/scripts"),
    Path.home() / "Siril" / "scripts",
]

_PREAMBLE = '''#!/usr/bin/env python3
# Script: DG_Patch_Tool
# Title: DG_Patch_Tool
# Description: External patch and heal tool for starless image cleanup in Siril
# Author: Daian Gan
"""DG_Patch_Tool — standalone script for Siril.
Version: {version}

Author: Daian Gan
Website: https://daiangan.com
"""

import sys
import importlib.abc
import importlib.util

# Ensure required packages in Siril's Python environment
try:
    import sirilpy
    if hasattr(sirilpy, "ensure_installed"):
        sirilpy.ensure_installed("PyQt6", "opencv-contrib-python", "numpy", "scipy")
except Exception:
    pass

_MODULE_SOURCES = {{}}
_PACKAGE_NAMES = {packages!r}


class _EmbeddedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _MODULE_SOURCES:
            return None
        return importlib.util.spec_from_loader(
            fullname, self, is_package=fullname in _PACKAGE_NAMES
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = _MODULE_SOURCES[module.__name__]
        exec(compile(source, "<" + module.__name__ + ">", "exec"), module.__dict__)


sys.meta_path.insert(0, _EmbeddedFinder())
'''

_POSTAMBLE = '''
from dg_patch_tool.app import run

if __name__ == "__main__":
    sys.exit(run())
'''


def condense_docstrings(source: str) -> str:
    """Condenses multi-paragraph docstrings down to their first paragraph."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    doc_spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                expr = node.body[0]
                doc_node = expr.value
                doc_spans.append(
                    (doc_node.lineno, doc_node.end_lineno, doc_node.col_offset, doc_node.value)
                )

    if not doc_spans:
        return source

    # Sort descending by start lineno so line replacements do not invalidate earlier offsets
    doc_spans.sort(key=lambda x: x[0], reverse=True)
    lines = source.splitlines()

    for start_line, end_line, col_offset, doc_text in doc_spans:
        paragraphs = re.split(r"\n\s*\n", doc_text.strip())
        first_para = paragraphs[0].strip()
        indent = " " * col_offset
        if "\n" in first_para:
            para_formatted = f"\n{indent}".join(line.strip() for line in first_para.splitlines())
            condensed = f'{indent}"""\n{indent}{para_formatted}\n{indent}"""'
        else:
            condensed = f'{indent}"""{first_para}"""'

        lines[start_line - 1 : end_line] = [condensed]

    result = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    try:
        ast.parse(result)
        return result
    except SyntaxError:
        return source


def _module_name_and_kind(path: Path) -> tuple[str, bool]:
    rel = path.relative_to(PACKAGE_ROOT.parent)
    parts = list(rel.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


def collect_modules() -> tuple[dict[str, str], list[str]]:
    modules: dict[str, str] = {}
    packages: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel_parts = path.relative_to(PACKAGE_ROOT).parts
        if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
            continue
        name, is_package = _module_name_and_kind(path)
        modules[name] = path.read_text(encoding="utf-8")
        if is_package:
            packages.append(name)

    return modules, packages


def _extract_version(modules: dict[str, str]) -> str:
    match = re.search(r'__version__\s*=\s*"([^"]+)"', modules["dg_patch_tool"])
    if match is None:
        return "1.0.0"
    return match.group(1)


def format_module_block(name: str, source: str) -> str:
    """Formats module source as readable multi-line string without repr escaping."""
    cleaned = condense_docstrings(source).strip()
    separator = "# " + "=" * 76
    return f"""
{separator}
# Module: {name}
{separator}
_MODULE_SOURCES["{name}"] = r\'\'\'
{cleaned}
\'\'\'
"""


def build() -> Path:
    modules, packages = collect_modules()
    version = _extract_version(modules)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = _PREAMBLE.format(version=version, packages=packages)
    module_blocks = "".join(format_module_block(name, modules[name]) for name in sorted(modules))
    full_content = header + module_blocks + _POSTAMBLE

    OUTPUT_PATH.write_text(full_content, encoding="utf-8")
    ROOT_OUTPUT_PATH.write_text(full_content, encoding="utf-8")

    # Clean up legacy space-separated files if present
    for old_path in [REPO_ROOT / "dist" / "DG Patch Tool.py", REPO_ROOT / "DG Patch Tool.py"]:
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass

    return OUTPUT_PATH


def copy_to_siril_scripts(output_path: Path) -> list[Path]:
    """Copies the built script into detected Siril scripts folders."""
    copied: list[Path] = []
    for scripts_dir in SIRIL_SCRIPTS_DIRS:
        if scripts_dir.is_dir():
            old_script = scripts_dir / "DG Patch Tool.py"
            if old_script.exists():
                try:
                    old_script.unlink()
                except Exception:
                    pass
            dest = scripts_dir / output_path.name
            shutil.copy2(output_path, dest)
            copied.append(dest)
    return copied


if __name__ == "__main__":
    output = build()
    size_kb = output.stat().st_size / 1024
    print(f"Built single-file bundle: {output} ({size_kb:.1f} KB)")
    copied_list = copy_to_siril_scripts(output)
    for c in copied_list:
        print(f"Copied to Siril scripts folder: {c}")
