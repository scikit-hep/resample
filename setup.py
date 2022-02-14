import site
import sys
import setuptools
import os
from pathlib import Path
import numpy

# workaround to allow editable install as user
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]


libs = ["npyrandom"]
if os.name == "posix":
    libs.append("m")

numpy_include = numpy.get_include()
random_path = str(Path(numpy_include).parent.parent / "random" / "lib")

ext = setuptools.Extension(
    "resample._ext",
    sources=["src/rcont.c", "src/resample/_ext.c"],
    include_dirs=[numpy_include],
    libraries=libs,
    library_dirs=[random_path],
)

setuptools.setup(ext_modules=[ext])
