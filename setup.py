#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="mars-gym",
    version="0.1.0",
    license="MIT",
    description="Framework Code for the RecSys 2020 entitled 'MARS-Gym: A Gym framework to model, train, and evaluate recommendationsystems for marketplaces'.",
    long_description="%s\n%s" % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst"))
    ),
    author="MARS-Gym Authors",
    author_email="mars-gym@googlegroups.com",
    url="https://github.com/deeplearningbrasil/mars-gym",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    entry_points='''
        [console_scripts]
        mars-gym=mars_gym.cli:cli
    ''',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        # uncomment if you test on these interpreters:
        # "Programming Language :: Python :: Implementation :: IronPython",
        # "Programming Language :: Python :: Implementation :: Jython",
        # "Programming Language :: Python :: Implementation :: Stackless",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Documentation": "https://mars-gym.readthedocs.io/",
        "Changelog": "https://mars-gym.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/deeplearningbrasil/mars-gym/issues",
    },
    keywords=[
        # eg: "keyword1", "keyword2", "keyword3",
    ],
    python_requires=">=3.6",
    install_requires=[
        "luigi>=2.8,<3.0", "gym>=0.15,<1",
        "numpy>=1.17,<2", "scipy>=1.3,<2", "pandas>=0.25,<1", "pyspark>=2.4,<3",
        "matplotlib>=2.2,<3", "seaborn>=0.8,<1", "plotly>=4.4,<5", "streamlit==0.62",
        "torch==1.2", "torchbearer==0.5", "pytorch-nlp>=0.4",
        "scikit-learn>=0.21,<1", "imbalanced-learn>=0.4,<1", "tensorboardx>=1.6,<2",
        "tqdm<5", "requests>=2,<3", "diskcache>=3,<4", "psutil>=5,<6",
        "click>=7.0","docutils==0.15"
    ],
    extras_require={
        # eg:
        #   "rst": ["docutils>=0.11"],
        #   ":python_version=="2.6"": ["argparse"],
    },
)
