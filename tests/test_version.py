from urllib.request import urlopen
from pkg_resources import parse_version
import json


def test_local_against_pypi_version():
    from resample import __version__

    r = urlopen("https://pypi.org/pypi/resample/json")
    assert r.code == 200
    payload = r.read()
    releases = json.loads(payload)["releases"]

    # make sure version is up-to-date
    latest_pypi_version = max(parse_version(v) for v in releases)
    assert (
        parse_version(__version__) > latest_pypi_version
    ), "PyPI release found, version needs to be incremented"
