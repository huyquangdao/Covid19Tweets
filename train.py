from trainers.mnist_trainer import MNISTTrainer
from metrics.accuracy import AccuracyScore
from metrics.f1score import F1Score
from metrics.precision import PrecisionScore
from metrics.recall import RecallScore
from logs.writer import PandasWriter, TensorboardWriter
from models.mnist_model import MNISTModel
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch


train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor())


train_loader = DataLoader(train_dataset, 32, shuffle=True)
test_loader = DataLoader(test_dataset, 32, shuffle=False)

model = MNISTModel(n_classes=10, image_size=28, hidden_size=256)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

trainer = MNISTTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[
        AccuracyScore(),
        PandasWriter(),
        TensorboardWriter(),
        F1Score(
            average_type='micro'),
        scheduler],
    device=torch.device('cpu'))

trainer.train(epochs=5, train_loader=train_loader, test_loader= test_loader)
