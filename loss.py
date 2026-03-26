import numpy as np

def mse(y, y_hat):
    # y: gerçek değerler, y_hat: modelimizin tahminleri
    # hataları karele, topla, ortalamasını al
    errors = y - y_hat
    squared = errors ** 2
    loss = np.mean(squared)
    return loss
