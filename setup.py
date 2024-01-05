from os import path
from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pospredict",
    description="Position prediction",
    version='0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Sragy/PositionPrediction",
    author="Sammy Ragy",
    packages=find_packages(),
    install_requires=["wheel",
                      "matplotlib",
                      "numpy"
                      "torch",
                      "nflows",
                      "sbi"
                      ],
    dependency_links=[],
)