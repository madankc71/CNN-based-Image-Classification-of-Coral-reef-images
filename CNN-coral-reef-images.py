import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

import os
# from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Define CNN Class
class convolutionalNetwork(nn.Module):
    def __init__(self):
        super(convolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256*14*14, 5000)
        self.fc2 = nn.Linear(5000, 512)
        self.fc3 = nn.Linear(512,2)
        self.dropout = nn.Dropout(p=0.5)     
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = F.relu(self.conv3(X))
        X = self.pool(X)
        X = F.relu(self.conv4(X))
        X = self.pool(X)
        X = X.view(-1, 256*14*14)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.dropout(F.relu(self.fc2(X)))
        X = self.fc3(X)
        return F.log_softmax(X,dim=1)

# def get_mean_std(train_data):
#     #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     num_samples = 0
#     for data, _ in train_data:
#         batch_samples = data.size(0)
#         data = data.view(batch_samples, data.size(1), -1)
#         mean += data.mean(2).sum(0)
#         std += data.std(2).sum(0)
#         num_samples += batch_samples
#     mean /= num_samples
#     std /= num_samples
#     return mean.tolist(), std.tolist()

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    
    print(f'Using device: {device}')
        
    root = "/home/madan/Coral/"
    batch_size = 32
    val_split = 0.2
    test_split = 0.15


    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Lambda(lambda x: transforms.Normalize(*get_mean_std(train_data))(x))
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    coral_reef = datasets.ImageFolder(os.path.join(root, 'coral-data'))

    val_size = int(val_split * len(coral_reef))
    test_size = int(test_split * len(coral_reef))
    train_size = len(coral_reef) - val_size - test_size

    train_data, val_data, test_data = random_split(coral_reef, [train_size, val_size, test_size])

    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_test_transform
    test_data.dataset.transform = val_test_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=100, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=100, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=100, pin_memory=True, drop_last=True)

    CNNModel = convolutionalNetwork().to(device)
    print(CNNModel)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNNModel.parameters(), lr = 0.001)

    n_epochs = 10
    train_losslist = []
    valid_loss_min = np.Inf # track change in validation loss

    import time
    start_time = time.time()
    for epoch in range(n_epochs):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        CNNModel.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # move data to GPU device
            # forward pass: compute predicted outputs by passing inputs to the model
            output = CNNModel(data)

            # calculate the batch loss
            loss = criterion(output, target)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()*data.size(0)

        # validate the model
        CNNModel.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device) 

            # forward pass: compute predicted outputs by passing inputs to the model
            output = CNNModel(data)

            # calculate the batch loss
            loss = criterion(output, target)

            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(val_loader.dataset)
        train_losslist.append(train_loss)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))


        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(CNNModel.state_dict(), 'model_marvel_heroes.pt')
            valid_loss_min = valid_loss

    #Testing the accuracy on test images
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            
            # calculate outputs by running images through the network
            outputs = CNNModel(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    total_time = time.time() - start_time
    print(f'Total time taken: {total_time/60} minutes')
	    
if __name__ == '__main__':
    main()
