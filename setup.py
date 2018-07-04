import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deployml",
    version="0.0.1",
    author="Maxwell Flitton, Romain Belia",
    author_email="maxwellflitton@gmail.com",
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
        'matplotlib'
    ],
    zip_safe=False
)
