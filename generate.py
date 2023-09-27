
import numpy as np
import os
from tqdm import tqdm, trange
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    m = table.shape[0]
    n = table.shape[1]
    k = 5
    mu = .1
    lam = .1
    step_I = .00007
    step_U = .00007
    I = np.ones((m, k))
    U = np.ones((n, k))
    R = np.nan_to_num(table, copy=True)

    iters = 100

    for i in range(iters):
        grad_U = -2*R.T@I + 2*U@I.T@I + 2*mu*U
        grad_I = -2*R@U + 2*I@U.T@U + 2*lam*I

        U -= step_U*grad_U
        I -= step_I*grad_I


    table = I@U.T
    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
