from urllib.request import urlopen
from pkg_resources import parse_version
import json


def test_local_against_pypi_version():
    from resample import __version__

    r = urlopen("https://pypi.org/pypi/resample/json")
    assert r.code == 200
    releases = json.loads(r.read())["releases"]

    # make sure version is up-to-date
    pypi_versions = [parse_version(v) for v in releases]
    assert parse_version(__version__) not in pypi_versions, "pypi version exists"
