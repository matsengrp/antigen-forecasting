from setuptools import setup, find_packages

setup(
    name="antigentools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
    ],
)