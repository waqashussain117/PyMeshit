def test_package_imports():
    import Pymeshit
    assert hasattr(Pymeshit, "main_wrapper")
    assert callable(Pymeshit.main_wrapper)


def test_version_string():
    import Pymeshit
    assert isinstance(Pymeshit.__version__, str)
    assert Pymeshit.__version__ != "0.0.0"


