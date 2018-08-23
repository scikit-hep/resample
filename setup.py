from setuptools import setup

setup(name="resample",
      version="0.1",
      description="Tools for bootstrapping and other resampling techniques.",
      url="http://github.com/dsaxton/resample",
      author="Daniel Saxton",
      author_email="daniel.saxton@gmail.com",
      license="BSD-3-Clause",
      packages=["resample"],
      install_requires=["numpy >= 1.9.0"],
      zip_safe=False)
