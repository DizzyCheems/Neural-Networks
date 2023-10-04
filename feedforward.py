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

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
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

# Updated training data
# Each row represents a person with income, education, age, and house ownership
# The last column represents their creditworthiness level (0 for excellent, 1 for average, 2 for poor)
data = torch.tensor([
    [120000, 5, 35, 1, 0],    # Excellent creditworthiness (high income, high education)
    [300, 2, 28, 0, 2],    # Poor creditworthiness (low income)
    [80000, 5, 40, 1, 0],    # Excellent creditworthiness (high income, high education)
    [6000, 3, 32, 1, 1],    # Average creditworthiness (average income)
    [20, 2, 45, 0, 2],    # Poor creditworthiness (low income)
    [150000, 4, 55, 1, 0],  # Excellent creditworthiness (high income, high education)
    [450, 2, 30, 0, 2],    # Poor creditworthiness (average income)
    [55000, 3, 38, 1, 1],    # Average creditworthiness (average income)
    [180, 1, 27, 0, 2],    # Poor creditworthiness (low income)
    [100000, 4, 42, 1, 0],  # Excellent creditworthiness (high income, high education)
    [90000, 4, 37, 1, 0],    # Excellent creditworthiness (high income, high education)
    [4800, 3, 29, 0, 1],    # Average creditworthiness (average income)
    [252, 1, 22, 0, 2],    # Poor creditworthiness (low income)
    [5500, 3, 36, 1, 1],    # Average creditworthiness (average income)
    [135000, 4, 44, 1, 0],  # Excellent creditworthiness (high income, high education)
    [92000, 4, 38, 1, 0],    # Excellent creditworthiness (high income, high education)
    [40000, 3, 31, 0, 1],    # Average creditworthiness (average income)
    [280, 2, 29, 0, 2],    # Poor creditworthiness (low income)
    [5200, 4, 39, 1, 1],    # Average creditworthiness (average income)
    [140000, 5, 48, 1, 0],  # Excellent creditworthiness (high income, high education)
    [110000, 5, 42, 1, 0],  # Excellent creditworthiness (high income, high education)
    [70000, 3, 34, 1, 1],  # Average creditworthiness (average income)
    [2500, 2, 26, 0, 2],    # Poor creditworthiness (low income)
    [30000, 3, 33, 1, 1],    # Average creditworthiness (average income)
    [1800, 1, 23, 0, 2],    # Poor creditworthiness (low income)
    [115000, 4, 47, 1, 0],  # Excellent creditworthiness (high income, high education)
    [65000, 3, 36, 1, 1],  # Average creditworthiness (average income)
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

# Function to input values and make predictions
def predict_creditworthiness(model):
    income = float(input("Enter income: "))
    education = int(input("Enter education level (1 to 5): "))
    age = int(input("Enter age: "))
    property_ownership = int(input("Enter property ownership (0 for no, 1 for yes): "))
    
    new_data = torch.tensor([[income, education, age, property_ownership]], dtype=torch.float32)
    predicted_scores = model(new_data)
    predicted_class = torch.argmax(predicted_scores, dim=1)
    
    return predicted_class.item()

# Input values and make predictions
while True:
    print("\nPredict Creditworthiness")
    predicted_class = predict_creditworthiness(model)
    if predicted_class == 0:
        print("Predicted Class: Excellent")
    elif predicted_class == 1:
        print("Predicted Class: Average")
    elif predicted_class == 2:
        print("Predicted Class: Poor")
    
    cont = input("\nDo you want to make another prediction? (yes/no): ").lower()
    if cont != "yes":
        break
