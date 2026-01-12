"""EdgeFlow - DSL for deploying AI models on edge devices."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="edgeflow",
    version="0.1.0",
    author="EdgeFlow Contributors",
    description="DSL for deploying AI models on edge devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pointblank-club/edgeFlow",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "PyYAML>=6.0",
        "antlr4-python3-runtime>=4.13.2",
        "lark>=1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
            "mypy>=1.0",
            "genbadge[all]>=1.1.1",
            "numpy>=1.24",
            "tensorflow>=2.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "edgeflow=edgeflow.compiler.edgeflowc:main",
        ],
    },
)
