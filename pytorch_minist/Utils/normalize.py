import numpy as np
def normalize(X):
    if (np.max(X) - np.min(X) != 0):
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X = (X - 0.5) / 0.5
    return X

if __name__ == '__main__':
    X = np.array([1.,2,100])
    print(normalize(X))