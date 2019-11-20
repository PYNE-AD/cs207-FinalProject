import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADPYNE-207",
    version="0.0.3",
    author="Group 12 - CS 207",
    author_email="ptoroisaza@g.harvard.edu, nvanderklaauw@g.harvard.edu, emmali@college.harvard.edu, yaoweili@g.harvard.edu",
    description="Automatic Differentiation Package for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PYNE-AD/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)