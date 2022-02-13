import site
import sys
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import os
from pathlib import Path

# workaround to allow editable install as user
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]


def configuration():
    import numpy

    libs = ["npyrandom"]
    if os.name == "posix":
        libs.append("m")

    random_path = str(Path(numpy.get_include()).parent.parent / "random" / "lib")

    config = Configuration("resample")
    config.add_extension(
        "_ext",
        sources=["src/rcont.c", "src/resample/_ext.c"],
        libraries=libs,
        library_dirs=[random_path],
    )

    return config.todict()


setup(**configuration())
