import numpy as np
import time
class Solve:

    def __init__(self,k,mu,alpha,beta,train_data,descent_method = 'SGD',n_steps = 100,seed = 10):
        self.k = k
        self.mu = mu
        self.alpha =alpha
        self.beta = beta
        self.data = np.copy(train_data)
        self.non_nan = np.argwhere(~np.isnan(train_data))
        self.descent = descent_method
        self.I = np.random.rand(len(self.data),self.k) #Generating random matrices, maybe a better initialization can be initialized
        self.U = np.random.rand(len(self.data[0]),self.k).T

        self.n_steps = n_steps
    
    def compute_sgd(self):
        d_I, d_U = 0, 0
        for (i, j) in self.non_nan:
            eij = data[i][j] - np.dot(self.I[i,:],self.U[:,j])
            for k in range(self.k):
                d_I += self.I[i][k] + self.alpha * (2 * eij * self.U[k][j] - self.mu * self.I[i][k])
                d_U += self.U[k][j] + self.beta * (2 * eij * self.I[i][k] - self.mu * self.U[k][j])

        return d_I,d_U
    
    def train(self, output_loss=False):
        loss = []
        for _ in range(self.n_steps):
            if output_loss:
                e = 0
                for (i,j) in self.non_nan:
                    e = e + pow(self.data[i][j] - np.dot(self.I[i,:],self.U[:,j]), 2)
                    for k in range(self.k):
                        e = e + (self.mu/2) * (pow(self.I[i][k],2) + pow(self.U[k][j],2))

                loss.append(e)

            for (i, j) in self.non_nan:
                eij = self.data[i][j] - np.dot(self.I[i,:],self.U[:,j])
                for k in range(self.k):
                    self.I[i, k] = self.I[i, k] + self.alpha * (2 * eij * self.U[k, j] - self.mu * self.I[i, k])
                    self.U[k, j] = self.U[k, j] + self.beta * (2 * eij * self.I[i, k] - self.mu * self.U[k, j])
        
        self.I = np.around(self.I*2, 0)/2
        self.U = np.around(self.U*2, 0)/2
        return loss



    def train_masked(self):
        for _ in range(self.n_steps):
            masked = np.ma.array(self.data, mask=np.isnan(self.data))
            masked_T = np.ma.transpose(masked)
            d_U = np.ma.add(np.ma.add(-2*np.ma.dot(masked_T,self.I), 2*self.U.T@self.I.T@self.I) , 2*self.mu*self.U.T)
            #d_U = np.ma.add(np.ma.add(-2*masked_T@self.I, 2*self.U@self.I.T@self.I),2*self.mu*self.U)
            d_I = np.ma.add(np.ma.add(-2*np.ma.dot(masked,self.U.T) ,2*self.I@self.U@self.U.T), 2*self.mu*self.I)
            self.I -= self.alpha*d_I
            self.U -= self.beta*d_U.T

    def rmse(self, test_matrix):
        # diffs = 0
        # predictions = self.predict()
        # T = len(np.argwhere(~np.isnan(test_matrix)))
        # for (i, j) in np.argwhere(~np.isnan(test_matrix)):
        #     diff = (test_matrix[i, j] - predictions[i, j])**2
        #     diffs += diff
        # return np.sqrt(diffs/T)
        masked = np.ma.array(test_matrix, mask=np.isnan(test_matrix))
        predictions = self.I@self.U
        diff = np.ma.subtract(predictions, masked)
        squared = np.ma.power(diff, 2)
        return np.ma.sqrt(np.ma.mean(squared))



    def predict(self):
        return self.I@self.U

if __name__ == '__main__':
    data_path = '../datasets/'
    data = np.load(data_path + 'ratings_train.npy')
    test_data = np.load(data_path + 'ratings_test.npy')
    np.random.seed(42)
    t_1 = time.time()
    solver = Solve(k=1,mu = 0.5,alpha = 0.005,beta = 0.005,train_data=data, n_steps=100)
    pred = solver.train()
    t_2 = time.time()
    print(f'elapsed time solver without mask: {t_2 - t_1}')
    rmse = solver.rmse(test_data)
    train_rmse = solver.rmse(data)
    print("Solver no mask")
    print(f"RMSE against TRAIN: {train_rmse}")
    print(f"RMSE against TEST: {rmse}")
