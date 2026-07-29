"""Package and install NebulaPy with its console commands.

This setuptools configuration reads the project version and requirements,
includes the Silo installer, and registers NebulaPy's command-line entry
points.
"""

from setuptools import setup, find_packages
import os
import re

# Read the README.md file
with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

# Read the version from scr.version
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "NebulaPy", "version.py")
    with open(version_file) as f:
        content = f.read()
    match = re.search(r"__version_info__\s*=\s*\(([^)]+)\)", content)
    if match:
        version_tuple = match.group(1).replace("'", "").replace('"', "").split(",")
        return ".".join(v.strip() for v in version_tuple)
    raise RuntimeError("Unable to find version string.")

# Read dependencies from requirements.txt
def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='NebulaPy',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=get_version(),
    author='Arun Mathew',
    author_email='arun@cp.dias.ie',
    url='',
    packages=find_packages(),
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        'NebulaPy': ['scripts/install_silo.sh'],
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        "License :: OSI Approved :: MIT License",
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    entry_points={
        'console_scripts': [
            'install-silo=NebulaPy.scripts.install_silo:main',
        ],
    },
)
