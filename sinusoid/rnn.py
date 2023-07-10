import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Generate training data
x_train = np.linspace(0, 10, num=1000)
y_train = np.sin(x_train)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(x_train).float().unsqueeze(1)
y_train = torch.from_numpy(y_train).float().unsqueeze(1)

# Generate test data
x_test = np.linspace(0, 20, num=2000)
y_test = np.sin(x_test)

# Convert data to PyTorch tensors
x_test = torch.from_numpy(x_test).float().unsqueeze(1)
y_test = torch.from_numpy(y_test).float().unsqueeze(1)

# Define the RNN model
class SinRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SinRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Set the hyperparameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.01
num_epochs = 100

# Create the RNN model
model = SinRNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Test the model
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    test_loss = criterion(y_pred, y_test)
    print('Test Loss:', test_loss.item())

# Plot the ground truth and predictions
import matplotlib.pyplot as plt

plt.plot(x_test.numpy(), y_test.numpy(), label='Ground Truth')
plt.plot(x_test.numpy(), y_pred.numpy(), label='Predictions')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Ground Truth vs Predictions')
plt.legend()
plt.show()
