import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main = nn.Sequential(
            nn.Conv1d(29, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
           
            
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)
    

    

def train(model, train_dataloader, test_dataloader, num_epochs=5, lr=0.001, beta1=0.5):
    classifier = model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(beta1, 0.999))

    e_loss_val = []
    e_loss_train = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader, 0):
            iter_loss = 0.0
            classifier.zero_grad()
            real_cpu = data[0].to(device)
            real_labels = data[1].to(device)
            
            real_cpu = real_cpu.view(-1, 29, 1)
            real_labels = real_labels.view(-1, 1)
            
            output = classifier(real_cpu)
            loss = criterion(output, real_labels)
            loss.backward()
            optimizer.step()
            iter_loss+=loss.item()
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\t Train Loss: %.4f'
                    % (epoch, num_epochs, i, len(train_dataloader),
                        loss.item()))
        e_loss_train.append(iter_loss / len(train_dataloader))
            
        
        with torch.no_grad():
            for data in test_dataloader:
                iter_loss = 0.0
                real_cpu = data[0].to(device)
                real_labels = data[1].to(device)
                
                real_cpu = real_cpu.view(-1, 29, 1)
                real_labels = real_labels.view(-1, 1)
                
                outputs = classifier(real_cpu)
                loss = criterion(outputs, real_labels)
                iter_loss+=loss.item()
                
            e_loss_val.append(iter_loss / len(test_dataloader))
            
    return e_loss_train, e_loss_val, classifier

def evaluate_and_plot(model, x_test, y_test, e_loss_train, e_loss_val, name="classifier_loss.png"):
    accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


    plt.figure(figsize=(10,5))
    plt.title("Loss")
    plt.plot(e_loss_train,label="Train")
    plt.plot(e_loss_val,label="Validation")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(name)
    plt.show()
    
    
if __name__ == "__main__":
    manualSeed = 999

    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) 

    workers = 2
    batch_size = 256

    num_epochs = 5
    lr = 0.001
    beta1 = 0.5

    X, y = load_and_preprocess_data('/home/vgl/Projects/ML-Project/creditcard.csv')
    x_train, x_test, y_train, y_test = split_data(X, y, test_size=0.2)

    y_train = y_train.values
    y_test = y_test.values


    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_train.shape)


    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    ## Train
    e_loss_train, e_loss_val, model = train(Classifier, train_dataloader, test_dataloader, num_epochs=num_epochs, lr=lr, beta1=beta1)
    
    ## evaluate
    evaluate_and_plot(model, x_test, y_test, e_loss_train, e_loss_val)

