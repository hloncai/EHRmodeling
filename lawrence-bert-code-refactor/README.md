# BERT Model Code Refactor

This folder contains code that refactor of Haotian Gong's Bert Model. 

1. params.py contains all the constant variables and values that used in the other functions.

2. modeling.py contains code refactoring from Haotian's notebook, which is used to generate predicted results by bert model.

3. preprocessing.py contains code refactoring from Haotian's notebook, which is used to generate processed (transformed) raw data. It is then feed into Bert Model. 

4. plot.py is the initial python file used to generate ROC plots based on the outputs.

5. figure.py is used to generate all the ROC subplots from model's outputs. 

6. eval_metrics.py contains functions that is used to evaluate model's performance by comparing predicted output to true label.

7. run.py contains main function to run all the logic of model training and result visualization.

8. script_* pythons file contains command lines that use the functions from other python file to generate processed data and plot the corresponding model's outputs. 

