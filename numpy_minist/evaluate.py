import torch
from data_utils.normalize import *

def evaluate(model, data, args):
    
    X, y = data
    # X = X[:200]
    # y = y[:200]

    if args.normalize_x:
        X = normalize(X)

    if args.model == 'cnn':
        X = X.reshape(-1, 1, 28, 28)
    
    out = model.forward(X)
    predicted = np.argmax(out, axis=1)
    

    hit = 0
    for idx, i in enumerate(predicted):
        if i == y[idx][0]:
            hit += 1
            
    return hit / y.shape[0]
