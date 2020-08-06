from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="template for the team to use",
    author="aicoe-aiops",
    license="",
    install_requires=["click", "python-dotenv>=0.5.1"],
)
