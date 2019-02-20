import torchvision.transforms as transforms
import numpy as np
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help='path to the directory to train the network with')

    parser.add_argument('--save_dir', type=str, help='path to save checkpoint of trained neural network.',
                        default='checkpoints')

    parser.add_argument('--gpu', help='enables gpu mode', action='store_true')

    parser.add_argument('--arch', help='sets pre trained model architecture', type=str, default='vgg')

    parser.add_argument('--epochs', help='sets amount of epochs to train the network', type=int, default=5)

    parser.add_argument('--learn_rate', help='sets the learning rate used when training the network', type=float,
                        default=0.01)

    parser.add_argument('--hidden_units', help='sets amount of hidden units in the network', type=int, default=64)

    args = parser.parse_args()
    return args


def set_directory_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
    return data


def set_loaders(data):
    data_transforms_train = transforms.Compose([
        transforms.Resize(250),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transforms_valid = transforms.Compose([
        transforms.Resize(250),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_test = transforms.Compose([
        transforms.Resize(250),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(data['train'], data_transforms_train)
    valid_data = datasets.ImageFolder(data['valid'], data_transforms_valid)
    test_data = datasets.ImageFolder(data['test'], data_transforms_test)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=24)
    validationloader = torch.utils.data.DataLoader(valid_data, batch_size=24)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=24)

    loaders = {'train': trainloader, 'valid': validationloader, 'test': testloader, 'trainset': train_data}

    return loaders


def load_network(arch):
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    select_models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
    if arch in select_models:
        in_arch = arch
    else:
        in_arch = 'vgg'
    model = select_models[in_arch]
    return model, in_arch, select_models


def build_classifier(model, hidden):
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.classifier[0].in_features, hidden)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden, 102)),
        ('out', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def validation(model, validloader, device):
    criterion = nn.CrossEntropyLoss()
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1]).to(device)
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return valid_loss, accuracy


def train(model, device, epochs, trainloader, validationloader, learn_rate):
    print('Training..', flush = True)
    steps = 0
    criterion = nn.CrossEntropyLoss()
    print_every = 30
    model.to(device)
    for i in range(epochs):
        model.train()
        running_loss = 0
        optimizer = optim.ASGD(model.classifier.parameters(), lr=learn_rate)
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validationloader, device)

                print("Epoch: {}/{}... ".format(i + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "validation Loss: {:.3f}.. ".format(valid_loss / len(validationloader)),
                      "validation Accuracy: {:.3f}".format(accuracy / len(validationloader)), flush = True)

                running_loss = 0

                model.train()


def test(model, testloader, device):
    print('Performing validation on test set...', flush = True)
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            model.eval()
            total += len(labels)
            inputs, label = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            predictions = torch.exp(output).max(dim=1)[1]
            for i in range(len(predictions)):
                if predictions[i] == label[i]:
                    correct += 1
        print('{} correct out of {}, {:.2f}% accurate'.format(correct, total, (correct / total) * 100))


def save_model(model, arch, epochs, path, trainset, select_models):
    print('saving trained model to ./{}/checkpoint.pth'.format(path), flush = True)
    model.class_to_idx = trainset.class_to_idx
    checkpoint = {
        'model': select_models[arch],
        'features': model.features,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }
    if not os.path.exists(path):
        print('Creating save directories...', flush = True)
        os.makedirs(path)
    torch.save(checkpoint, path + '/checkpoint.pth')


def main():
    args = get_args()
    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    data = set_directory_data(args.data_dir)

    loaders = set_loaders(data)

    model, in_arch, select_models = load_network(args.arch)

    model = build_classifier(model, args.hidden_units)

    train(model, device, args.epochs, loaders['train'], loaders['valid'], args.learn_rate)

    test(model, loaders['test'], device)

    save_model(model, in_arch, args.epochs, args.save_dir, loaders['trainset'], select_models)

    print('Complete!', flush = True)


if __name__ == "__main__":
    sys.stdout.flush()
    main()

    # run with python train.py flowers --save_dir checkpoints --gpu --arch vgg --epochs 3 --learn_rate 0.01 --hidden_units 3 