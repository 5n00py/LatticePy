from setuptools import setup, find_packages

setup(
    name="LatticePy",
    version="0.2",
    packages=find_packages(),
    author="David Schmid",
    author_email="david.schmid@mailbox.org",
    description="A Python library for working with lattices",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="http://github.com/5n00py/LatticePy",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
