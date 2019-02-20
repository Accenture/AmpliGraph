from setuptools import setup, find_packages
from ampligraph import __version__ as version
from setuptools.command.install import install
from subprocess import call
import sys

setup_params = dict(name='ampligraph',
      version=version,
      description='A Python library for relational learning on knowledge graphs.',
      url='https://innersource.accenture.com/projects/DL/repos/ampligraph/',
      author='Accenture Dublin Labs',
      author_email='luca.costabello@accenture.com',
      license='',
      packages=find_packages(exclude=('tests', 'docs')),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'numpy',
          'pytest',
          'sklearn',
          'deap',
          'joblib',
          'pytest',
          'tqdm',
          'pandas',
          'sphinx',
          'recommonmark',
          'sphinx_rtd_theme',
          'sphinxcontrib-bibtex'
      ],
      extras_require={
          'cpu': ['tensorflow'],
          'gpu': ['tensorflow-gpu'],
      }
    )
if __name__ == '__main__':
    setup(**setup_params)
