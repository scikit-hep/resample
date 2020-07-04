def test_init_version():
    import resample  # noqa

    assert isinstance(resample.__version__, str)


def test_version():
    from resample import version

    assert isinstance(version.version, str)
