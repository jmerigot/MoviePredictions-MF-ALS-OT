# Collaborative Filtering and Optimal Transport for Recommender Systems

This group project was completed for the Data Science Lab course as part of the IASD Master Program (AI Systems and Data Science) at PSL Research University.
The project accomplished the following:
- Investigated optimal transport plan interpretation and its application in computing optimal matchings, with a focus on the cost matrix interpretation and RMSE results.
- Explored the challenge of interpreting outputs as movie ratings, revealing non-linear distributions and the impact of extreme sparsity in the input ratings matrix on the results.
- Identified the limitations of the developed linear bin-based method in capturing deep representations of movie ratings and the potential for improvement through more intelligent rating mapping.
- Concluded that the extreme sparsity of the input ratings matrix posed a significant challenge for optimal transport, leading to skewed costs and transport plans, and highlighted the need for further investigation into this problem.

The full report associated with this project can be found in the report.pdf file of this repository, which details our approach and methods used to complete this project, as well as the analyzed results.

## Note for Reviewers:

The code is organized into three seperate directories:
- `jm/SolveALS.py` contains the core SGD and ALS implementations, along with the grid search and hyper parameter tuning.
- `ek/ot_solver.ipynb` contains the notebook with the Inverse Optimal Transport implementation and analysis
- `mi/RIOT.ipynb` contains the notebook with the Regularized Inverse Optimal Transport analysis, the implementation is in the `Matcher.py` and `ot.py` scripts, both derived from the project experiments of https://jmlr.csail.mit.edu/papers/volume20/18-700/18-700.pdf .

## General Information
If you want to copy and recreate this project, or test it for yourself, some important information to know.

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

- **Ellington Kirby**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Mehdi Inane**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Jules Merigot** (myself)
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.

