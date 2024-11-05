# Welcome to QuTools

<p align="center">
    <img src="./assets/qt-orange.svg" alt="QuTools Logo" width="30%">
</p>

This package provides methods to analyze questionnaire-style data using
clustering methods and to automize the scoring process of its open-ended tasks.

I wrote this code during my PhD studies for the Machine-Learning-based analysis
of a questionnaire containing closed-format- (multiple-choice) and open-format
tasks and items. I generalized the code, such that it is applicable for other
questionnaires as well by setting up a suitable configuration file.

However, this is definitely a "researchy" codebase with much to be done to
call it a proper package. There are various open ToDos (see a list below) which
could and should be tackeled in the future, but doing this all alone in my
freetime, while my own PhD-analyses are basically completed, is not much fun. I
therefore invite all interested folks, that could make use of the current (and
future) functionalities provided to contribute to the project. Open ToDos include:

1. Refactoring the code such that it an analyses is set up as a project with
fixed and automatically findable configuration and datafiles such that the
load-and-save logic of the different objects can be simplified dramatically.

2. Refactoring the task-wise finetuning capabilities currently integrated in the
[`QuFinetuneScorer`][qutools.scoring.scorer_finetune]-class out of this class into
a wrapper for [`QuScorer`][qutools.scoring.scorer_base] objects, s. t. the
code can be simplified and separated.

3. Setting up tests for the different modules using toy-data (there is already
some toy-data in the `_toydata`-directory).

4. ...
