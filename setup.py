import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version_contents = {}
with open(os.path.join('deployml', 'version.py')) as f:
    exec(f.read(), version_contents)

setuptools.setup(
    name="deployml",
    version=version_contents['VERSION'],
    author="Maxwell Flitton, Romain Belia",
    author_email="maxwellflitton@gmail.com, belia.bourgeois.romain@gmail.com",
    description="Easy training and deployment of machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deploy-ml/deploy-ml",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Build Tools"
    ),
    install_requires=[
        'pandas',
        'imblearn',
        'sklearn',
        'numpy',
        'matplotlib',
        'keras',
        'tensorflow'
    ],
    zip_safe=False
)
