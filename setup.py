from setuptools import setup, find_packages

setup(name='find_delay',
      version='2.18',
      packages=find_packages(exclude=["demos"]),
      include_package_data=True)
