# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from setuptools import setup, find_packages
from ampligraph import __version__ as version

setup_params = dict(name='ampligraph',
                    version=version,
                    description='A Python library for relational learning on knowledge graphs.',
                    url='https://github.com/Accenture/AmpliGraph/',
                    author='Accenture Dublin Labs',
                    author_email='about@ampligraph.org',
                    license='Apache 2.0',
                    packages=find_packages(exclude=('tests', 'docs')),
                    include_package_data=True,
                    zip_safe=False,
                    install_requires=[
                        'numpy>=1.14.3',
                        'pytest>=3.5.1',
                        'scikit-learn>=0.19.1',
                        'deap>=1.2.2',
                        'joblib>=0.11',
                        'tqdm>=4.23.4',
                        'pandas>=0.23.1',
                        'sphinx>=2.2',
                        'recommonmark>=0.4.0',
                        'sphinx_rtd_theme>=0.4.0',
                        'sphinxcontrib-bibtex>=0.4.0',
                        'beautifultable>=0.7.0',
                        'pyyaml>=3.13',
                        'rdflib>=4.2.2',
                        'scipy>=1.3.0',
                        'networkx>=2.3',
                        'flake8>=3.7.7',
                        'setuptools>=36'
                    ])
if __name__ == '__main__':
    setup(**setup_params)
