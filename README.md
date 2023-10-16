# DataLabAssignement1

# Note for Reviewers:

The code is organized into three seperate directories:
- `jm/SolveALS.py` contains the core SGD and ALS implementations, along with the grid search and hyper parameter tuning.
- `ek/ot_solver.ipynb` contains the notebook with the Inverse Optimal Transport implementation and analysis
- `mi/github_stff.ipynb` contains the notebook with the Regularized Inverse Optimal Transport analysis, the implementation is in the `Matcher.py` script.

## generate.py
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use and it saves the complete matrix as *output.npy*.
DO NOT CHANGE THE LINES TO LOAD AND SAVE THE TABLE. Between those to you are free to use any method for matrix completion. 
Example:
  > python3 generate.py --name ratings_train.npy

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt
