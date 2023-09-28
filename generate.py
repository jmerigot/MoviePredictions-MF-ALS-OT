
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
from mi.exp import Solve

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


    solver = Solve(k = 5,l=0.0001,mu = 0.0001,alpha = 0.00007,beta = 0.00007,train_data=table, n_steps=1000)
    pred = solver.train()
    table = solver.predict()
    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
