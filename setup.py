"""
Setup script for llama3np package.
"""

from setuptools import setup, find_packages

setup(
    name="llama3np",
    version="0.1.0",
    description="NumPy implementation of Llama3 with performance optimizations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Swap357",
    author_email="noreply@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "llama3np=run_llama:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)