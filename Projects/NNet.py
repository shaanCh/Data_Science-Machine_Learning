import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the transformation to apply to the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset and apply the transformation
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create data loaders for batch processing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Create an instance of the model
model = Net()

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.view(-1, 784)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Evaluate the model on test data
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(-1, 784)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



# Evaluate the model
model.eval()
test_loss = 0
correct = 0
misclassified_images = []
misclassified_targets = []
misclassified_predictions = []
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 784)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        misclassified_mask = pred.eq(target.view_as(pred)) == False
        misclassified_images.extend(data[misclassified_mask])
        misclassified_targets.extend(target.view_as(pred)[misclassified_mask])
        misclassified_predictions.extend(pred[misclassified_mask])

test_loss /= len(test_loader.dataset)

print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
    test_loss, 100. * correct / len(test_loader.dataset)))

# Plot misclassified images
num_samples = min(10, len(misclassified_images))
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

for i, ax in enumerate(axes):
    image = misclassified_images[i].view(28, 28)
    target = misclassified_targets[i].item()
    prediction = misclassified_predictions[i].item()

    ax.imshow(image, cmap='gray')
    ax.set_title(f'Target: {target}, Pred: {prediction}')
    ax.axis('off')

plt.show()
