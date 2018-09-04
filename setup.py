from setuptools import setup

setup(name="resample",
      version="0.1.2",
      description="Tools for randomization-based inference in Python",
      url="http://github.com/dsaxton/resample",
      author="Daniel Saxton",
      author_email="daniel.saxton@gmail.com",
      license="BSD-3-Clause",
      packages=["resample"],
      install_requires=["numpy >= 1.14.0", "scipy >= 1.1.0"],
      zip_safe=False)
