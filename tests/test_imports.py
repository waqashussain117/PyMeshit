def test_package_imports():
    import Pymeshit
    assert hasattr(Pymeshit, "main_wrapper")
    assert callable(Pymeshit.main_wrapper)


def test_version_string():
    from importlib.metadata import version
    import Pymeshit

    assert isinstance(Pymeshit.__version__, str)
    assert Pymeshit.__version__ != "0.0.0"
    assert Pymeshit.__version__ == version("Pymeshit")


def test_headless_api_is_exposed():
    import Pymeshit

    assert callable(Pymeshit.run_mesh_case)
    assert callable(Pymeshit.generate_tetrahedral_mesh_from_surfaces)


def test_headless_dataclasses_construct():
    from Pymeshit.headless import MeshCase, MeshOptions, SurfaceSpec

    case = MeshCase(
        surfaces=[
            SurfaceSpec(
                name="surface",
                points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                role="border",
            )
        ],
        options=MeshOptions(generate_volume=False),
    )

    assert case.surfaces[0].name == "surface"
    assert case.options.constraint_mode == "intersections"

