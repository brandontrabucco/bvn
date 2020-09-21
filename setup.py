from setuptools import setup


setup(
  name='bvn',
  packages=['bvn'],
  version='0.1',
  license='MIT',
  description='A Greedy Birkhoff-Von Neumann Decomposition',
  author='Brandon Trabucco',
  author_email='brandon@btrabucco.com',
  url='https://github.com/brandontrabucco/bvn',
  download_url='https://github.com/brandontrabucco/bvn/archive/v_01.tar.gz',
  keywords=['Permutation', 'Doubly Stochastic', 'Birkhoff-von Neumann', 'Decomposition'],
  install_requires=['networkx', 'numpy', 'tensorflow'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
