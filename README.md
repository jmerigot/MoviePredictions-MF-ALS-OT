# DataLabAssignement1

# Note for Reviewers:

The code is organized into three seperate directories:
- `jm/SolveALS.py` contains the core SGD and ALS implementations, along with the grid search and hyper parameter tuning.
- `ek/ot_solver.ipynb` contains the notebook with the Inverse Optimal Transport implementation and analysis
- `mi/RIOT.ipynb` contains the notebook with the Regularized Inverse Optimal Transport analysis, the implementation is in the `Matcher.py` and `ot.py` scripts, both derived from the project experiments of https://jmlr.csail.mit.edu/papers/volume20/18-700/18-700.pdf .

## General Information
If you want to cop and recreate this project, or test it for yourself, some important information to know.

### generate.py
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use and it saves the complete matrix as *output.npy*.
Example:
  > python3 generate.py --name ratings_train.npy

### requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
  > pip install -r requirements.txt

## Acknowledgements
This project was made possible with the guidance and support of the following :
 
- **Prof. Benjamin Negrevergne**
  - Associate professor at the Lamsade laboratory from PSL – Paris Dauphine university, in Paris (France)
  - Active member of the *MILES* project
  - Co-director of the IASD Master Program (AI Systems and Data Science) with Olivier Cappé.

- **Alexandre Verine**
  - PhD candidate at LAMSADE, a joint research lab of Université Paris-Dauphine and Université PSL, specializing in Machine Learning.

This project was a group project and was accomplished through team work involving the following students :

- **Matteo Sammut**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Alexandre Ngau**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Jules Merigot** (myself)
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.

