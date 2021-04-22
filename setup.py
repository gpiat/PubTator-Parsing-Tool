import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pubtatortool",
    version="0.4",
    author="Guilhem Piat",
    author_email="guilhem.piat@protonmail.com",
    description="A package for loading and manipulating PubTator files "
                "as Python objects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpiat/PubTator-Parsing-Tool",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'nltk',
        'diff_match_patch',
        'transformers',
    ],
)
