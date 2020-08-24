from os import path

from setuptools import setup


def open_local(fn):
    this_directory = path.abspath(path.dirname(__file__))
    return open(path.join(this_directory, fn), encoding="utf-8")


with open_local("README.md") as f:
    long_description = f.read()

with open_local("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

with open_local("resample/version.py") as f:
    exec(f.read())  # this sets `version`

setup(
    name="resample",
    version=version,
    description="Tools for randomization-based inference in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/resample-project/resample",
    author="Daniel Saxton and Hans Dembinski",
    license="BSD-3-Clause",
    install_requires=requirements,
    zip_safe=False,
    packages=["resample"]
)
