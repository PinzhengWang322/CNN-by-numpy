import torch
from Utils.normalize import *

def evaluate(model, data, args):
    model.eval()
    X, y = data
    X = X[:100]
    y = y[:100]
    if args.normalize_input:
        X = normalize(X)

    X, y = torch.tensor(X), torch.tensor(y)
    if args.model == 'cnn':
        X = X.reshape(-1, 1, 28, 28).float()
    else:
        X = X.float()
    
    out = model(X)
    _, predicted = torch.max(out, 1)

    hit = 0
    for idx, i in enumerate(predicted):
        if i == y[idx][0]:
            hit += 1
            
    model.train()
    return hit / y.shape[0]
