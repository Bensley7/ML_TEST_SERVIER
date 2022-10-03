import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

try:
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8"
    ) as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []


setup(
    name="ML_TEST_SERVIER",
    version="0.0.1",
    author="Mohammed Benslimane",
    author_email="mde.benslimane@gmail.com",
    description="molecular properties classification",
    license="MIT",
    install_requires=REQUIRED,
    keywords="Smile",
    url="https://github.com/Bensley7/ML_TEST_SERVIER.git",
    packages=["ML_TEST_SERVIER"],
    entry_points={"console_scripts": ["servier=ML_TEST_SERVIER.main:cli"]},
    classifiers=[
        "License :: MIT",
    ],
)
