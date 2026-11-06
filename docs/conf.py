"""Sphinx configuration for the PyMeshIt documentation."""

from pathlib import Path
import sys
import tomllib

from setuptools_scm import get_version


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_version() -> str:
    """Resolve the tag-based package version without importing the GUI."""
    with (ROOT / "pyproject.toml").open("rb") as source:
        fallback = tomllib.load(source)["tool"]["setuptools_scm"]["fallback_version"]
    return get_version(root=str(ROOT), fallback_root=str(ROOT), fallback_version=fallback)


project = "PyMeshIt"
author = "Waqas Hussain"
release = project_version()
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

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
nb_execution_mode = "off"
nb_merge_streams = True

html_theme = "furo"
html_title = f"PyMeshIt {release}"
html_theme_options = {
    "source_repository": "https://github.com/waqashussain117/PyMeshit/",
    "source_branch": "main",
    "source_directory": "docs/",
}
