import site
import sys
import setuptools  # type: ignore

# workaround to allow editable install as user
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setuptools.setup()
