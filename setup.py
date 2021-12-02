from setuptools import setup, find_packages
from pip._internal.req import parse_requirements


reqs = parse_requirements("./requirements.txt", session=False)

setup(
    version="1.0",
    name="neo-mayo",
    install_requires=reqs,
    packages=find_packages()
)