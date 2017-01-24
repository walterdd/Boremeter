#!/usr/bin/env python

from setuptools import setup

setup(
    name='boremeter',
    version='0.1',
    description='An app for tracking auditory boredom on video',
    author='Ilya Soloviev, Artem Shaforostov, Daria Walter',
    url='https://github.com/walterdd/Auditory_tracking',
    packages=['boremeter'],
    entry_points={
          'console_scripts': [
              'boremeter = boremeter.gen_report:main'
          ]
      },
)