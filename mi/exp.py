import numpy as np



data_path = '../datasets/'
data = np.nan_to_num(np.load(data_path + 'ratings_train.npy'))
class Solve:
    

    def __init__(self,k,l,mu,alpha,beta,train_data,descent_method = 'SGD',n_steps = 100):
        self.k = k
        self.l = l
        self.mu = mu
        self.alpha =alpha
        self.beta = beta
        self.data = train_data
        self.descent = descent_method
        self.I = np.random.rand(len(self.data),self.k) #Generating random matrices, maybe a better initialization can be initialized
        self.U = np.random.rand(len(self.data[0]),self.k)

        self.n_steps = n_steps
    
    def compute_sgd(self):
        d_U = -2*(self.data.T@self.I) + 2*(self.U@self.I.T@self.I) + 2*self.mu*self.U
        d_I = -2*(self.data @ self.U) + 2*(self.I@self.U.T@self.U) + 2*self.l*self.I
        return d_I,d_U
    
    def train(self):
        for i in range(self.n_steps):
            if self.descent == 'SGD':
                I,U = self.compute_sgd()
                self.I -= self.alpha*I
                self.U -= self.beta * U
        
        
        return np.sqrt(np.mean(self.I@self.U.T - self.data)**2)



solver = Solve(k = 5,l=0.0001,mu = 0.0001,alpha = 0.00007,beta = 0.00007,train_data=data)
pred = solver.train()
print(pred)