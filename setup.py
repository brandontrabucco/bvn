from setuptools import find_packages
from setuptools import setup


PACKAGES = ['tensorflow', 'numpy', 'networkx']


setup(name='bvn',
      description='A Greedy Birkhoff-Von neumann Decomposition',
      license='MIT',
      version='0.1',
      zip_safe=True,
      include_package_data=True,
      packages=find_packages(),
      install_requires=PACKAGES)
