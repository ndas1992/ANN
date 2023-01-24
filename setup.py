from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="ndas1992",
    description="A small package for ANN Implementation",
    Long_description=long_description,
    Long_description_content_type="text/markdown",
    url="https://github.com/ndas1992/ANN.git",
    author_email="dasnayan9678@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas", 
        "PyYAML"
    ]
)