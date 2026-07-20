# Installation

## Python package

Create a clean Python environment and install PyMeshIt from PyPI:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install and verify PyMeshIt:

```bash
python -m pip install --upgrade pip
python -m pip install pymeshit
python -c "import Pymeshit; print(Pymeshit.__version__)"
```

Pin a version when reproducibility matters:

```bash
python -m pip install "pymeshit==0.8.8"
```

## Linux system libraries

The Python API does not open the GUI during headless runs, but the current
package still includes Qt and VTK dependencies. On Ubuntu, install the common
runtime libraries if Qt or OpenGL imports fail:

```bash
sudo apt-get update
sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0 libegl1 libopengl0 libgl1
```

On a server without a display, set this before visualization or import checks:

```bash
export QT_QPA_PLATFORM=offscreen
```

## Development installation

To work on PyMeshIt itself:

```bash
git clone https://github.com/waqashussain117/PyMeshit.git
cd PyMeshit
python -m pip install -e .
```

The standalone application archives on the GitHub Releases page are separate
from the package installed by `pip`.

