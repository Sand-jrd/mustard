from setuptools import setup, find_packages
from pip._internal.req import parse_requirements


reqs = parse_requirements("./requirements.txt", session=False)
try:
    reqs = [str(ir.req) for ir in reqs]
except:
    reqs = [str(ir.requirement) for ir in reqs]

setup(
    version="1.0",
    name="neo-mayo",
    install_requires=reqs,
    packages=find_packages()
)