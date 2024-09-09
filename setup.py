from setuptools import setup, find_packages

setup(
    name="mynnlib",
    version="0.1.1",
    description="A library for building neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gustavo Oliveira Longuinho",
    author_email="gugui2820@gmail.com",
    url="https://github.com/Gustavo2820/My_NN_Library",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
        ],
    },
    package_data={
        "": ["*.txt", "*.md"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License"
    ],
)
