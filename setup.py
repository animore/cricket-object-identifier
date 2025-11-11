"""Setup configuration for cricket-object-identifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="cricket-object-identifier",
    version="0.1.0",
    author="animore",
    description="Identifying objects in cricket like bat, bowl, stumps from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/animore/cricket-object-identifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
