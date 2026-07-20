# Troubleshooting

## Import or display errors on Linux

Install the libraries listed in the [installation guide](installation.md). For
headless servers, try:

```bash
export QT_QPA_PLATFORM=offscreen
```

## A volume mesh is not generated

Inspect both the result flag and recorded failures:

```python
result = run_mesh_case(case)
print(result.ok)
print(result.failures)
```

Check that the selected surfaces form a closed, watertight PLC and that every
material seed lies inside its intended region. Use `generate_volume=False` to
inspect conforming surface meshes before running TetGen.

## Insufficient constraints

For a difficult non-benchmark case, try `constraint_mode="all"` while
diagnosing the geometry. In the standard intersection workflow,
`preserve_boundary_hulls=True` retains outer hulls while intersection curves
form internal constraints.

## Different node counts in batch cases

This is expected when geometry is remeshed. See
[Batch processing and POD studies](user-guide/batch-and-pod.md) for the
recommended common-reference interpolation workflow.

## Reporting a problem

Open an issue in the [PyMeshIt issue tracker](https://github.com/waqashussain117/PyMeshit/issues)
and include:

- PyMeshIt and Python versions;
- operating system;
- the complete traceback or `MeshResult.failures` value;
- relevant `MeshOptions`; and
- a minimal geometry case when it can be shared.

