import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate training data
x_train = np.linspace(0, 10, num=1000)
y_train = np.sin(x_train)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(x_train).float().unsqueeze(1)
y_train = torch.from_numpy(y_train).float().unsqueeze(1)

# Define the neural network model
class SinApproximator(nn.Module):
    def __init__(self):
        super(SinApproximator, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the neural network
model = SinApproximator()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    permutation = torch.randperm(x_train.size(0))
    for i in range(0, x_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Generate test data for prediction
x_pred = np.linspace(10, 20, num=100)
y_pred_groundtruth = np.sin(x_pred)

# Convert prediction data to PyTorch tensors
x_pred = torch.from_numpy(x_pred).float().unsqueeze(1)

# Use the trained model to make predictions
with torch.no_grad():
    model.eval()
    y_pred = model(x_pred).numpy()

# Calculate test loss
y_pred_tensor = torch.from_numpy(y_pred).unsqueeze(1)
y_pred_groundtruth_tensor = torch.from_numpy(y_pred_groundtruth).unsqueeze(1)
loss = criterion(y_pred_tensor, y_pred_groundtruth_tensor)
print('Test loss:', loss.item())

# Plot the ground truth and predictions
plt.plot(x_pred.numpy(), y_pred_groundtruth, label='Ground Truth')
plt.plot(x_pred.numpy(), y_pred, label='Predictions')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Ground Truth vs Predictions')
plt.legend()
plt.show()
