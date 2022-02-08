#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup, find_packages
from pathlib import Path

try: # for pip >= 10
	from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
	from pip.req import parse_requirements

pkg_name = "blob_detector"

cwd = Path(__file__).parent.resolve()
# Get __version__ variable
exec(open(str(cwd / pkg_name / '_version.py')).read())

install_requires = [line.strip() for line in open("requirements.txt").readlines()]


setup(
	name=pkg_name,
	python_requires=">3.7",
	version=__version__,
	description='Fine-tune framework based on chainer',
	author='Dimitri Korsch',
	author_email='korschdima@gmail.com',
	license='MIT License',
	packages=find_packages(),
	zip_safe=False,
	setup_requires=[],
	install_requires=install_requires,
    package_data={'': ['requirements.txt']},
    data_files=[('.',['requirements.txt'])],
    include_package_data=True,
)
