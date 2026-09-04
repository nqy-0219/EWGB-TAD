from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="LMTAD",
    version="0.0.10",
    description="Trajectory anomaly detection with LM",
    package_dir={"": "code"},
    packages=find_packages(where="code"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="-",
    author_email="-",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9.9",
        "Operating System :: OS Independent",
    ],    
    python_requires=">=3.9",
)
