from CustomTorch import *

# Create a neural network with dropout
model = NN()
model.add(Linear(2, 3))
model.add(Dropout(p=0.3))  # Add dropout with 50% probability
model.add(Linear(3, 1))

# Define loss and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Sample data
x = Tensor(np.array([[1, 2], [3, 4]]))
y_true = Tensor(np.array([[1], [0]]))

# Set model to training mode
model.train()

# Training loop
for epoch in range(100):
    y_pred = model.forward(x)
    loss = criterion.forward(y_pred, y_true)
    grad_output = criterion.backward(y_pred, y_true)
    model.backward(grad_output)
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Switch to evaluation mode for inference
model.eval()
y_pred = model.forward(x)
print("Final predictions:\n", y_pred.data)