#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='boremeter',
    version='0.1',
    description='An app for tracking auditory boredom on video',
    author='Ilya Soloviev, Artem Shafarostov, Daria Walter, Peter Romov',
    url='https://github.com/walterdd/boremeter',
    packages=['boremeter'],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "Jinja2",
        "tqdm",
    ],
    package_data={'boremeter': ['cv_haar_cascades/*.xml', 'templates/*.html']},
    include_package_data=True,
    entry_points={
          'console_scripts': [
              'boremeter = boremeter.gen_report:main'
          ]
      },
)
