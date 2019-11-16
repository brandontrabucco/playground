"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.0.0',
    'numpy',
    'mujoco-py',
    'gym[all]'
]


setup(
    name='playground',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('playground')],
    description='Deep Reinforcement Learning Algorithms.'
)
