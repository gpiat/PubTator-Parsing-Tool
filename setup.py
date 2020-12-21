import nltk
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PubTator-gpiat",  # Replace with your own username
    version="0.0.1",
    author="Guilhem Piat",
    author_email="guilhem.piat@protonmail.com",
    description="A package for loading and manipulating PubTator files "
                "as Python objects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpiat/PubTator-Parser",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Unlicense",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

nltk.download('punkt')