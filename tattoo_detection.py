"""
Author: Dimitrios Spanos
Date: 26/07/2021
"""
from CNN import CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "Data/Datasets/Week2_Dataset"
experiments = 5
learning_rate = 1e-4
epochs = 25
batch_size = 128
training_mode = False


def train(model, train_loader, test_loader):
    """
    :param model: The untrained CNN model
    :param train_loader: The train loader containing the augmented dataset
    :param test_loader: The test loader
    :return: The trained model
    """
    CNNmodel.train()
    criterion = nn.CrossEntropyLoss()
    if device == torch.device("cuda"):
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    accuracies = []
    loss=0
    for epoch in range(epochs):

        num_correct = 0
        num_samples = 0
        for b, (X_train, y_train) in enumerate(train_loader):
            
            b += 1
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            num_correct += (predicted == y_train).sum()
            num_samples += len(predicted)

            # Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"\nEpoch:{epoch} Loss:{loss.item():.5f}Accuracy:{(float(num_correct) / float(num_samples) * 100):.2f}%\n")

    # OPTIONAL
    """
        accuracies.append(test(model, test_loader))
        model.train()
        
    max_accuracy = max(accuracies)
    max_index = accuracies.index(max_accuracy)
    print(f"\nMax accuracy = {max_accuracy:.2f}% when epochs = {max_index + 1}.")
    """

    total = time.time() - start_time
    print(f"\nTraining took {(total / 60):.2f} minutes.\n")
    return model


def test(trained_model, loader):
    """
    :param trained_model: The trained CNN model
    :param loader: The test loader
    """
    trained_model.eval()
    num_correct = 0
    num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for x_test, y_test in loader:

            x_test, y_test = x_test.to(device=device), y_test.to(device=device)
            y_val = trained_model(x_test)

            predicted =  torch.max(y_val.data, 1)[1]
            num_correct += (predicted == y_test).sum()
            num_samples += len(predicted)

    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Testing accuracy: {accuracy:.2f}%")

    total = time.time() - start_time
    print(f"\nTesting took {total:.5f}secs.\n{(total/float(num_samples)*1000):.5f}ms per image on average.")
    return accuracy


def getData():
    """
    :return: The train loader and test loader
    """
    train_tranform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256), # The shorter side of the images is rescaled to 256 pixels
        transforms.CenterCrop(227), # The central patch with a size of 227x227 pixels is cropped
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(root, 'Flickr(10k)'), transform=train_tranform)
    test_data = datasets.ImageFolder(os.path.join(root, 'Flickr(2349)'), transform=test_transform)

    return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(test_data, batch_size=batch_size)


if __name__ == '__main__':

    # Creation of train and test loaders
    training_loader, testing_loader = getData()

    # Creation of the CNN model
    CNNmodel = CNN()
    if device == torch.device("cuda"):
        CNNmodel.cuda()

    if training_mode:
        max_accuracy = 0
        for i in range(experiments):

            # Reset the model
            CNNmodel = CNN()
            if device == torch.device("cuda"):
                CNNmodel.cuda()

            # Training of the model
            CNNmodel = train(CNNmodel, training_loader, testing_loader)

            # Save the model
            if test(CNNmodel, testing_loader) > max_accuracy:
                torch.save(CNNmodel.state_dict(), 'CNN.pt')
                max_accuracy = test(CNNmodel, testing_loader)

    # Load the model
    CNNmodel.load_state_dict(torch.load('CNN.pt'))
    test(CNNmodel, testing_loader)
