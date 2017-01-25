#!/usr/bin/env python

from setuptools import setup

setup(
    name='boremeter',
    version='0.1',
    description='An app for tracking auditory boredom on video',
    author='Ilya Soloviev, Artem Shaforostov, Daria Walter, Peter Romov',
    url='https://github.com/walterdd/boremeter',
    packages=['boremeter'],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "Jinja2",
    ],
    entry_points={
          'console_scripts': [
              'boremeter = boremeter.gen_report:main'
          ]
      },
)
