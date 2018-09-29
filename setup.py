from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="resample",
      version="0.21",
      description="Tools for randomization-based inference in Python",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="http://github.com/dsaxton/resample",
      author="Daniel Saxton",
      author_email="daniel.saxton@gmail.com",
      license="BSD-3-Clause",
      packages=["resample"],
      install_requires=["numpy >= 1.14.0", "scipy >= 1.1.0"],
      zip_safe=False)
