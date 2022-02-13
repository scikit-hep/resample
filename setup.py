import site
import sys
import setuptools
import numpy
from numpy.distutils.extension import Extension

# workaround to allow editable install as user
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

mod = Extension(
    "resample._ext",
    sources=["src/resample/_ext.c", "src/rcond2.c"],
    include_dirs=[numpy.get_include()],
)

setuptools.setup(
    zip_safe=False,
    ext_modules=[mod],
)
