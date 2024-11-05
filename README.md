# Questionnaire-Tools

A python library for exploratory analyses and automated scoring of questionnaires with open and closed response formats.

This code is highly experimental and "researchy" stuff - nothing to be deployed yet!

## Setup
- Install [pre-commit](https://pre-commit.com/) hooks via `pre-commit init` when planning to contribute.
- Setup the Python environment, e.g., using conda via the provided environment-`.yaml`s.
- If you wish: Build the documentation with [MkDocs](https://www.mkdocs.org/).


## ToDos
1. Refactoring the code such that it an analyses is set up as a project with fixed and automatically findable configuration and datafiles such that the load-and-save logic of the different objects can be simplified dramatically.

2. Refactoring the task-wise finetuning capabilities currently integrated in the [`QuFinetuneScorer`][qutools.scoring.scorer_finetune]-class out of this class into a wrapper for [`QuScorer`][qutools.scoring.scorer_base] objects, s. t. the code can be simplified and separated.

3. Setting up tests for the different modules using toy-data (there is already some toy-data in the `_toydata`-directory).

4. ...
