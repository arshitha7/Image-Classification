# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

To classify the given images into it's category

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 3:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

### STEP 4:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 5:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 5:
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name:
### Register Number:
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: ARSHITHA MS')
        print('Register Number: 212223240015')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

## OUTPUT
### Training Loss per Epoch

![image](https://github.com/user-attachments/assets/a3900c38-9909-4111-ac4c-a2f5bb6fb737)


### Confusion Matrix

![image](https://github.com/user-attachments/assets/78897750-7c1f-409b-8516-03df7efa8fc0)


### Classification Report

![image](https://github.com/user-attachments/assets/ae83d5f2-d2bb-436b-800b-faeda9dfec2d)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/7a075fee-8bbd-4e1e-adea-8f7c8e041c2a)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
