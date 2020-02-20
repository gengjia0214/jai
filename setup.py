"""Setup for the jai."""

import setuptools


with open('README.md') as f:
    README = f.read()

requirements = []
with open('requirements.txt', mode='r') as req:
    reader = req.readlines()
    for pkg in reader:
        pkg = pkg.replace('=', '>=', 1)
        pkg = pkg[:pkg.rfind('=')]
        requirements.append(pkg)

setuptools.setup(
    author="Jia Geng",
    author_email="gengjia0214@hotmail.com",
    name='jai',
    license="BSD 3-Clause License",
    description='jai provide handy toolkit for building ai projects',
    version='v0.0.1',
    long_description=README,
    url='https://github.com/gengjia0214/jai.git',
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research'
    ],
)