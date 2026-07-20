"""Sphinx configuration for the PyMeshIt documentation."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_version() -> str:
    """Read the package version without importing the GUI or build backend."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not determine the PyMeshIt version")
    return match.group(1)


project = "PyMeshIt"
author = "Waqas Hussain"
release = project_version()
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# API pages intentionally import only the headless module. Mocking the compiled
# scientific stack keeps documentation builds fast and avoids initializing Qt.
autodoc_mock_imports = [
    "PySide6",
    "matplotlib",
    "netCDF4",
    "numpy",
    "pyvista",
    "pyvistaqt",
    "scipy",
    "tetgen",
    "triangle",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True
numpydoc_show_class_members = False

myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "furo"
html_title = f"PyMeshIt {release}"
html_theme_options = {
    "source_repository": "https://github.com/waqashussain117/PyMeshit/",
    "source_branch": "main",
    "source_directory": "docs/",
}

