Headless API reference
======================

The supported headless interface is available from both :mod:`Pymeshit` and
:mod:`Pymeshit.headless`. Importing from :mod:`Pymeshit` is recommended for
normal use.

Inputs
------

.. currentmodule:: Pymeshit.headless

.. autoclass:: SurfaceSpec
   :members:

.. autoclass:: WellSpec
   :members:

.. autoclass:: MaterialSpec
   :members:

.. autoclass:: MeshOptions
   :members:

.. autoclass:: MeshCase
   :members:

Execution
---------

.. autofunction:: read_points

.. autofunction:: run_mesh_case

.. autofunction:: generate_tetrahedral_mesh_from_surfaces

Results
-------

.. autoclass:: MeshResult
   :members:

