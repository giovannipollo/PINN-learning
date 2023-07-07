import torch
import matplotlib.pyplot as plt
import numpy as np

# Set the device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Neural_net(torch.nn.Module):
    def __init__(self):
        super(Neural_net, self).__init__()
        self.layer1 = torch.nn.Linear(1, 32)
        self.tanh = torch.nn.Tanh()
        self.layer3 = torch.nn.Linear(32, 32)
        self.layer2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer3(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Simple_no_pinn():
    def __init__(self, epochs, seed=0):
        torch.manual_seed(seed=seed)
        self.model = Net()  # Simple 1-layer PyTorch neural network model
        self.some_xs = np.linspace(0,2,10)
        self.some_xs_tensor = torch.Tensor(self.some_xs).unsqueeze(1)
        self.some_y = self.true_sol(self.some_xs)
        self.some_y_tensor = torch.Tensor(self.some_y).unsqueeze(1)
        output = self.model(self.some_xs_tensor)
        mean_loss = self.mse_loss(output, self.some_y_tensor)
        # print(self.some_xs)
        # print(self.some_y)
        # print("mean loss initially: ", mean_loss)
        self.some_other_xs = torch.linspace(0,2,100,requires_grad=True).reshape(-1,1) 
        self.x = np.linspace(0, 2, 1000)
        self.y = self.true_sol(self.x)
        self.x_domain = torch.Tensor(self.x).unsqueeze(1)
        self.output = torch.Tensor(self.y).unsqueeze(1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.c0 = 1.0
        self.epochs = epochs

    # Define the mean squared error (MSE) loss function
    def mse_loss(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)
    

    def true_sol(self, x):
        return x ** 2 + np.exp(-x ** 2 / 2) / (1 + x + x ** 3)

    def train(self):
        for epoch in range(self.epochs):
            # print("epoch: ", epoch)
            self.optimizer.zero_grad()
            outputs = self.model(self.x_domain)
            loss = self.mse_loss(outputs, self.output)
            loss.backward()
            self.optimizer.step()
            # print("loss:", loss)


        output = self.model(self.some_xs_tensor)
        mean_loss = self.mse_loss(output, self.some_y_tensor)
        # print("mean loss finally: ", mean_loss)
        # plt.plot(self.some_other_xs.cpu().detach().numpy(), self.model(self.some_other_xs).cpu().detach().numpy(), label='pred_new_data')
        plt.plot(self.x_domain.cpu().detach().numpy(), self.model(self.x_domain).cpu().detach().numpy(), label='pred_train_data')
        plt.scatter(self.some_xs, self.some_y, label='analytic')
        plt.legend()
        plt.grid()
        return mean_loss
        # plt.show()

if __name__ == '__main__':
    instance = Simple_no_pinn(epochs=50)
    instance.train()
