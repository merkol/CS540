import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def evaluate_model(model, X_test, y_test):
    x_test = torch.from_numpy(X_test).float().view(-1, 29, 1).to(device)
    y_pred = model(x_test).cpu().detach().numpy()
    y_pred = np.where(y_pred > 0.5, 1, 0)
   
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    features = ['V' + str(i) for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y
    
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y) 