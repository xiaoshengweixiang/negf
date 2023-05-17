from setuptools import setup, find_packages

setup(
    name='negf',
    version='0.0.1',
    description="A tool for calculating the quantum transport properties of condensed matter systems based on the principle of non-equilibrium Green's functions",
    author="Jinli Chen",
    author_email="1900011342@pku.edu.cn",
    packages=find_packages(),
    install_requires=[
        # list of dependencies
        'numpy',
        'matplotlib'
    ],
)
