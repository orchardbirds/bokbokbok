from setuptools import setup, find_packages

# with open("README.md", "r", encoding="UTF-8") as fh:
#     long_description = fh.read()

base_packages = [
    "pandas>=1.1.2",
    "scikit-learn>=0.23.2",
]



setup(
    name="bokbokbok",
    version="0.1",
    description="Custom Losses and Metrics for XGBoost, LightGBM, CatBoost",
    #long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel Timbrell",
    author_email="dantimbrell@gmail.com",
    license="Open Source",
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    include_package_data=True,
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
    },
    url="https://github.com/orchardbirds/bokbokbok",
    packages=find_packages(".", exclude=["tests", "notebooks", "docs"]),
)
