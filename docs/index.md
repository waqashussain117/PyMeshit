# PyMeshIt documentation

PyMeshIt provides GUI-driven and headless Python workflows for creating
conforming surface meshes and tetrahedral volume meshes from geological
surfaces, faults, boundaries, and wells.

The headless API runs the meshing workflow without opening the GUI. It is
designed for automated geometry studies, parameter sweeps, simulation setup,
and Exodus export.

```{admonition} API status
:class: important
The headless API is experimental in the 0.8 release series. Pin the PyMeshIt
version used for production studies and review the release notes before
upgrading.
```

## Start here

- Follow the [installation guide](installation.md).
- Run the [surface-meshing quickstart](quickstart.md).
- Work through the [complete batch-workflow notebook](examples/headless_batch_workflow.ipynb).
- Read the [batch and POD guidance](user-guide/batch-and-pod.md) before
  comparing results from varying geometries.
- Consult the [headless API reference](api/index.rst) for exact signatures and
  defaults.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/input-data
user-guide/geometry-and-materials
user-guide/meshing-options
user-guide/results
user-guide/exodus-export
user-guide/batch-and-pod
troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

examples/headless_batch_workflow
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

## Project links

- [Source code](https://github.com/waqashussain117/PyMeshit)
- [PyPI package](https://pypi.org/project/Pymeshit/)
- [Issue tracker](https://github.com/waqashussain117/PyMeshit/issues)
- [Download the original batch-workflow notebook](https://github.com/waqashussain117/PyMeshit/blob/main/examples/headless_batch_workflow.ipynb)
