from setuptools import setup, find_packages

setup(name='find_delay',
      version='2.16',
      packages=find_packages(exclude=["demos"]),
      include_package_data=True)
