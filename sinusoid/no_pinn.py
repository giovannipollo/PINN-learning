import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import torch
from torch.autograd import grad
import torch.nn as nn
from numpy import genfromtxt
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Neural_net(torch.nn.Module):
    def __init__(self, n_in = 1, n_out =1):
        super(Neural_net, self).__init__()

        self.tanh = torch.nn.Tanh()

        self.layer1 = torch.nn.Linear(n_in,20)
        self.layer2 = torch.nn.Linear(20,20)
        self.layer3 = torch.nn.Linear(20,20)
        self.layer_out = torch.nn.Linear(20,n_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        x = self.tanh(x)
        x = self.layer3(x)
        x = self.tanh(x)
        x = self.layer_out(x)
        return x
    
class sinusoid: 

    def __init__(self, epochs, data):
        self.epochs = epochs
        torch.manual_seed(0)
        self.model = Neural_net(n_out=1)

        self.lbfgs_optimizer = torch.optim.LBFGS(params = self.model.parameters(), lr = 0.001,max_iter = 500)
        self.adam_optimizer = torch.optim.Adam(params = self.model.parameters(), lr = 0.0001)

        self.t_dat = torch.tensor(data[0], dtype=torch.float).reshape(-1,1)
        self.x_dat = torch.tensor(data[1], dtype=torch.float)
        
    def data_loss(self):
        x = torch.unbind(self.model(self.t_dat), dim = 1)[0]
        z1 = torch.mean((x - self.x_dat)**2)
        return z1

    def data_loss_more_points(self, data):
        
        t_dat = torch.tensor(data[0], dtype=torch.float).reshape(-1,1)
        x_dat = torch.tensor(data[1],dtype=torch.float)
        
        # Infer the model
        x = torch.unbind(self.model(t_dat), dim = 1)[0]
        
        z1 = torch.mean((x - x_dat)**2)
    
        plt.plot(t_dat.detach(), x.detach(), label = 'x pred')

        plt.scatter(t_dat.detach(), x_dat, label = 'x data')
        plt.legend()
        plt.savefig("sinusoid/no_pinn_pred.png")
        plt.close()
        return z1 

    def combined_loss(self):
        return self.data_loss()
    
    
    def plot_preds(self):
        x = torch.unbind(self.model(self.t_dat), dim = 1)[0]
        plt.plot(self.t_dat.detach(), x.detach(), label = 'x pred')
        plt.scatter(self.t_dat.detach(), self.x_dat, label = 'x data')
        plt.legend()
        plt.savefig("sinusoid/no_pinn.png")
        plt.close()
       


    def lbfgs_train(self):
                self.model.train()    
                for epoch in range(self.epochs):
                    def closure():
                        self.lbfgs_optimizer.zero_grad()
                        loss = self.combined_loss()
                        loss.backward()
                        return loss
                    self.lbfgs_optimizer.step(closure=closure)
                    print(f'Epoch {epoch}, loss: {self.combined_loss()}')
                self.plot_preds()
            
        

    def adam_train(self):
            steps = 1000
            self.model.train()
            for epoch in range(self.epochs):
                for step in range(steps):
                    def closure():
                        self.adam_optimizer.zero_grad()
                        loss = self.combined_loss()
                        loss.backward()
                        return loss
                    self.adam_optimizer.step(closure=closure)
                print(f'Epoch {epoch}, loss: {self.combined_loss()}')
            self.plot_preds()
      
      
if __name__ == "__main__":      
    t = np.linspace(0,20,100)
    t2 = np.linspace(-10,30,100)
    sol = np.sin(t)
    sol2 = np.sin(t2)

    inp_dat = np.array([t, sol])
    inp_dat2 = np.array([t2, sol2]) 
    
    np.random.seed(1)
    ids = np.random.choice(range(inp_dat.shape[1]), size = 20)
    sample_data = inp_dat[:,ids]
    # test_inst2 = sinusoid(epochs=10, data = sample_data)
    test_inst2 = sinusoid(epochs=10, data = inp_dat)
    for i in range(100):
        test_inst2.adam_train()
        # Compute the mean square error with respect to the reference points
        mse = test_inst2.data_loss_more_points(data=inp_dat2)
        # mse = test_inst2.data_loss()
        print("Mean square error: ", mse)