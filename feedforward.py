import torch
import torch.nn as nn
import torch.optim as optim
import os

# Define the neural network model
class CreditWorthinessNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CreditWorthinessNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Define the input size, hidden layer size, and output size
input_size = 4  # Number of input features (income, education, age, house ownership)
hidden_size = 8  # Number of neurons in the hidden layer
output_size = 3  # Number of output classes (excellent, average, poor)

# Create an instance of the neural network
model = CreditWorthinessNN(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample data for training (you should replace this with your dataset)
# Each row represents a person with income, education, age, and house ownership
# The last column represents their creditworthiness level (0 for excellent, 1 for average, 2 for poor)
data = torch.tensor([
    [120000, 5, 35, 1, 0],    # Excellent creditworthiness (high income, high education)
    [30000, 2, 28, 0, 1],    # Poor creditworthiness (low income)
    [80000, 5, 40, 1, 0],    # Excellent creditworthiness (high income, high education)
    [60000, 3, 32, 1, 2],    # Average creditworthiness (average income)
    [20000, 2, 45, 0, 2],    # Poor creditworthiness (low income)
    [150000, 4, 55, 1, 0],  # Excellent creditworthiness (high income, high education)
    [45000, 2, 30, 0, 2],    # Poor creditworthiness (average income)
    [55000, 3, 38, 1, 1],    # Average creditworthiness (average income)
    [18000, 1, 27, 0, 2],    # Poor creditworthiness (low income)
    [100000, 4, 42, 1, 0],  # Excellent creditworthiness (high income, high education)
    [90000, 4, 37, 1, 0],    # Excellent creditworthiness (high income, high education)
    [48000, 3, 29, 0, 2],    # Average creditworthiness (average income)
    [25000, 1, 22, 0, 2],    # Poor creditworthiness (low income)
    [55000, 3, 36, 1, 1],    # Average creditworthiness (average income)
    [135000, 4, 44, 1, 0],  # Excellent creditworthiness (high income, high education)
    [92000, 4, 38, 1, 0],    # Excellent creditworthiness (high income, high education)
    [40000, 3, 31, 0, 2],    # Average creditworthiness (average income)
    [28000, 2, 29, 0, 2],    # Poor creditworthiness (low income)
    [52000, 4, 39, 1, 1],    # Average creditworthiness (average income)
    [140000, 5, 48, 1, 0],  # Excellent creditworthiness (high income, high education)
    [110000, 5, 42, 1, 0],  # Excellent creditworthiness (high income, high education)
], dtype=torch.float32)

# Splitting the data into features (X) and labels (y)
X_train = data[:, :-1]  # Input features (income, education, age, house ownership)
y_train = data[:, -1].long()  # Creditworthiness labels (0 for excellent, 1 for average, 2 for poor)

# Function to train the model
def train_model(model, X_train, y_train, epochs=10000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Function to save the model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

# Function to load the model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

# Check if a pre-trained model exists
model_path = "credit_worthiness_model.pt"
if os.path.exists(model_path):
    print("Found a pre-trained model.")
    choice = input("Do you want to use the pre-trained model? (yes/no): ").lower()
    if choice == "yes":
        model = CreditWorthinessNN(input_size, hidden_size, output_size)
        load_model(model, model_path)
    else:
        print("Continuing training the model.")
        train_model(model, X_train, y_train)
else:
    print("No pre-trained model found. Training a new model.")
    train_model(model, X_train, y_train)
    save_model(model, model_path)

# After training or loading the model, you can use it for predictions
new_data = torch.tensor([[110000, 5, 33, 1]], dtype=torch.float32)  # High income, high education
predicted_class = torch.argmax(model(new_data), dim=1)
print("Predicted Class:", predicted_class.item())
