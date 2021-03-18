# Contributing guide

`bokbokbok` aims to integrate custom loss functions to LightGBM, XGBoost and CatBoost. 
To add a loss function / eval metric / to contibute in general please follow these steps:

- Discuss the feature you want to add on Github before you write a PR for it. On disagreements, maintainer(s) will have the final word.
- If you’re going to add a loss function, please contribute the derivations of gradients and Hessians.
- When issues or pull requests are not going to be resolved or merged, they should be closed as soon as possible.
 This is kinder than deciding this after a long period. Our issue tracker should reflect work to be done.

That said, there are many ways to contribute to bokbokbok, including:

- Contribution to code
- Improving the documentation
- Reviewing merge requests
- Investigating bugs
- Reporting issues

Starting out with open source? See the guide [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/) and have a look at [our issues labelled *good first issue*](https://github.com/ing-bank/probatus/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Setup

Development install:

```shell
pip install -e '.[all]'
```

Run unit tests with

```shell
pytest
```

## Standards

- Python 3.7+
- Follow [PEP8](http://pep8.org/) as closely as possible (except line length)
- [google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)
- Git: Include a short description of *what* and *why* was done, *how* can be seen in the code. Use present tense, imperative mood
- Git: limit the length of the first line to 72 chars. You can use multiple messages to specify a second (longer) line: `git commit -m "Patch load function" -m "This is a much longer explanation of what was done"`


### Derivations

We use [Code cogs](https://www.codecogs.com/latex/eqneditor.php) to generate equations that are compatible with Git and markdown.
To use an equation, choose svg format and HTML embedding and copy the link at the bottom of the page.
        

### Documentation

Documentation is a very crucial part of the project, because it ensures usability of the package. We develop the docs in the following way:

* We use [mkdocs](https://www.mkdocs.org/) with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme. The `docs/` folder contains all the relevant documentation.
* We use `mkdocs serve` to view the documentation locally. Use it to test the documentation every time you make any changes.
* Maintainers can deploy the docs using `mkdocs gh-deploy`. The documentation is deployed to `https://orchardbirds.github.io/`.