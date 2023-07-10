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
    
class predprey_pinn: 

    def __init__(self, epochs, data, c0):
        self.epochs = epochs
        torch.manual_seed(0)
        self.model = Neural_net(n_out=2)
        self.domain = torch.linspace(0,int(max(data[0])),100, requires_grad=True).reshape(-1,1)

        self.lbfgs_optimizer = torch.optim.LBFGS(params = self.model.parameters(), lr = 0.001,max_iter = 500)
        self.adam_optimizer = torch.optim.Adam(params = self.model.parameters(), lr = 0.0001)

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4

        self.t_dat = torch.tensor(data[0], dtype=torch.float).reshape(-1,1)
        self.x_dat = torch.tensor(data[1],dtype=torch.float)
        self.y_dat = torch.tensor(data[2], dtype=torch.float)

        self.maxes = {}
        self.mins = {}

        for id,d in enumerate((self.x_dat, self.y_dat)):
            self.maxes[id] = max(d)
            self.mins[id] = min(d)

        self.x_norm = self.normalize(0, self.x_dat)
        self.y_norm = self.normalize(1, self.y_dat)

        x0 = self.normalize(0,c0[0])
        y0 = self.normalize(1,c0[1])
        self.c0 = torch.tensor([x0,y0], dtype = torch.float)


    def normalize(self, id, unnormed):
        return (unnormed - self.mins[id])/(self.maxes[id]- self.mins[id])

    def un_normalize(self, id, normed):
        return normed*(self.maxes[id] -self.mins[id])+ self.mins[id]

        
    def wrap_grad(self, f,x):
        return torch.autograd.grad(f,x,
        grad_outputs=torch.ones_like(x),
        retain_graph=True,
        create_graph=True)[0]

    def de_loss(self):
        pred = self.model(self.domain)
        x,y = (d.reshape(-1,1) for d in torch.unbind(pred, dim =1))
        
        dx = self.wrap_grad(x, self.domain)
        dy = self.wrap_grad(y, self.domain)

        x = self.un_normalize(0,x)
        y = self.un_normalize(1,y)

        ls0 = torch.mean((dx - (self.alpha*x -self.beta*x*y)/(self.maxes[0]-self.mins[0]) )**2)
        ls1 = torch.mean((dy -(self.delta*x*y - y*self.gamma)/(self.maxes[1]-self.mins[1]))**2)
        ic = torch.mean((self.c0-pred[0])**2)
        
        return ls0 + ls1 + ic
    
    def data_loss(self):
        x,y = torch.unbind(self.model(self.t_dat), dim = 1)
        z1 = torch.mean((x - self.x_norm)**2)
        z2 = torch.mean((y- self.y_norm)**2)
        return z1 + z2

    def data_loss_more_points(self, data):
        def normalize(id, unnormed, mins, maxes):
            return (unnormed - mins[id])/(maxes[id]-mins[id])
        
        def un_normalize(id, normed, mins, maxes):
            return normed*(maxes[id] - mins[id])+ mins[id]
        
        t_dat = torch.tensor(data[0], dtype=torch.float).reshape(-1,1)
        x_dat = torch.tensor(data[1],dtype=torch.float)
        y_dat = torch.tensor(data[2], dtype=torch.float)
        
        maxes = {}
        mins = {}
        for id,d in enumerate((x_dat, y_dat)):
            maxes[id] = max(d)
            mins[id] = min(d)
        x_norm = normalize(0, x_dat, mins, maxes)
        y_norm = normalize(1, y_dat, mins, maxes) 
        
        # Infer the model
        x,y = torch.unbind(self.model(t_dat), dim = 1)
        
        z1 = torch.mean((x - x_norm)**2)
        z2 = torch.mean((y- y_norm)**2)
        
        # Plot the predictions
        x = un_normalize(0,x,mins, maxes)
        y = un_normalize(1,y, mins, maxes)
        plt.plot(t_dat.detach(), x.detach(), label = 'x pred')
        plt.plot(t_dat.detach(), y.detach(), label = 'y pred')

        plt.scatter(t_dat.detach(), x_dat, label = 'x data')
        plt.scatter(t_dat.detach(), y_dat, label = 'y data' )
        plt.legend()
        plt.savefig("pred_prey/no_pinn_pred.png")
        plt.close()
        return z1 + z2 

    def combined_loss(self):
        return self.data_loss()
    
    
    def plot_preds(self):
        x,y = torch.unbind(self.model(self.domain), dim = 1)
        x = self.un_normalize(0,x)
        y = self.un_normalize(1,y)
        plt.plot(self.domain.detach(), x.detach(), label = 'x pred')
        plt.plot(self.domain.detach(), y.detach(), label = 'y pred')

        plt.scatter(self.t_dat.detach(), self.x_dat, label = 'x data')
        plt.scatter(self.t_dat.detach(),self.y_dat, label = 'y data' )
        plt.legend()
        plt.savefig("pred_prey/no_pinn.png")
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
                    print(f'Epoch {epoch}, loss: {self.combined_loss()}, beta_param: {self.beta}, gamma_param: {self.gamma}')
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
                print(f'Epoch {epoch}, loss: {self.combined_loss()}, beta_param: {self.beta}, gamma_param: {self.gamma}')
            self.plot_preds()
      
      
if __name__ == "__main__":      
    alpha = 1.1
    beta = 0.4
    delta = 0.1
    gamma = 0.4

    def pp_ode(state, t):
        x,y = state
        dx = alpha*x -beta*x*y
        dy = delta*x*y - y*gamma
        return [dx, dy]

    t = np.linspace(0,50,50)
    t2 = np.linspace(0,100,200)
    sol = odeint(pp_ode,y0 =[10,10], t=t)
    sol2 = odeint(pp_ode, y0=[10,10], t=t2)
    

    inp_dat = np.array([t, sol[:,0], sol[:,1]])
    inp_dat2 = np.array([t2, sol2[:,0], sol2[:,1]]) 
    
    np.random.seed(1)
    ids = np.random.choice(range(inp_dat.shape[1]), size = 20)
    sample_data = inp_dat[:,ids]
    
    t_domain = np.linspace(0,50,100)
    domain_sol = odeint(pp_ode, y0=[10,10], t=t_domain)
    inp_data_domain = np.array([t_domain, domain_sol[:,0], domain_sol[:,1]])
    # Add the domain_sol to the sample_data
    new_sample_data = np.concatenate((sample_data, inp_data_domain), axis = 1)
    print(new_sample_data.shape)
    # test_inst2 = predprey_pinn(epochs=10, data = sample_data, c0 =[10,10])
    # test_inst2 = predprey_pinn(epochs=10, data = inp_dat, c0 =[10,10])
    test_inst2 = predprey_pinn(epochs=10, data = new_sample_data, c0 =[10,10])
    for i in range(100):
        test_inst2.adam_train()
        # Compute the mean square error with respect to the reference points
        # mse = test_inst2.data_loss_more_points(data=inp_dat2)
        mse = test_inst2.data_loss()
        print("Mean square error: ", mse)