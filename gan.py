import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
from classifier import *

class Generator(nn.Module):
    def __init__(self, X_minority) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        nn.Linear(in_features=100, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=X_minority.shape[1])
    )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, X_minority) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=X_minority.shape[1], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
    )
    def forward(self, x):
        return self.main(x)


def apply_gan(X, y, num_steps=10000):

    
    X_minority = X[y == 1]
    X_minority_tensor = torch.FloatTensor(X_minority)
    
    # 2. Define the GAN model layers
    generator = Generator(X_minority)
    discriminator = Discriminator(X_minority)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001)

    # 3. Train GAN
    G_loss = []
    D_loss = []
    for step in range(num_steps):
        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(len(X_minority_tensor), 1)
        fake_data = generator(torch.randn(len(X_minority_tensor), 100))
        fake_labels = torch.zeros(len(X_minority_tensor), 1)
        
        logits_real = discriminator(X_minority_tensor)
        logits_fake = discriminator(fake_data.detach())
        
        loss_real = criterion(logits_real, real_labels)
        loss_fake = criterion(logits_fake, fake_labels)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        logits_fake = discriminator(fake_data)
        loss_g = criterion(logits_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()
        
        print(f"Step {step+1}/{num_steps} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")
        G_loss.append(loss_g.item())
        D_loss.append(loss_d.item())

  
    num_synthetic_samples = len(X[y == 0]) - len(X_minority)
    print(num_synthetic_samples)
    with torch.no_grad():
        synthetic_samples = generator(torch.randn(num_synthetic_samples, 100)).numpy()

    print(f"Mean of synthetic samples {np.mean(synthetic_samples)}:.4f" )
    print(f"Std of synthetic samples {np.std(synthetic_samples)}:.4f" )
    
    print(f"Mean of real samples {np.mean(X[y == 1])}:.4f" )
    print(f"Std of real samples {np.std(X[y == 1])}:.4f" )
    
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.hstack([y, [1]*num_synthetic_samples])

    return X_resampled, y_resampled, G_loss, D_loss


if __name__ == "__main__":
    X, y = load_and_preprocess_data("creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_resampled, y_resampled, gloss, dloss = apply_gan(X_train, y_train)
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    model = Classifier().to(device)
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_resampled).float(), torch.from_numpy(y_resampled).float())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    e_loss_train, e_loss_val, classifier = train(Classifier, train_dataloader, test_dataloader, num_epochs=2)
    evaluate_and_plot(classifier, X_test, y_test, e_loss_train, e_loss_val, name="Resampled_Classifier.png")
    
    plt.figure(figsize=(10,5))
    plt.title("Loss")
    plt.plot(gloss,label="Generator")
    plt.plot(dloss,label="Discriminator")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig("gan_loss.png")
    plt.show()
    