import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0.0
        self.b = 0.0

    def sigmoid(self, z):
        # z'yi 0-1 arasına sıkıştırıyoruz — olasılık gibi yorumlanır
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        # Önce linear kombinasyon, sonra sigmoid'den geçir
        z = self.w * x + self.b
        y_hat = self.sigmoid(z)
        return y_hat

    def compute_loss(self, y, y_hat):
        n = len(y)
        # Binary Cross-Entropy: MSE yerine bunu kullanıyoruz çünkü çıktı olasılık
        # y=1 ise log(y_hat) büyük olsun, y=0 ise log(1-y_hat) büyük olsun istiyoruz
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def compute_gradients(self, x, y, y_hat):
        n = len(y)
        # Sigmoid + BCE kombinasyonunda gradients linear'dakiyle aynı forma geliyor
        error = y_hat - y
        dw = (1 / n) * np.sum(x * error)
        db = (1 / n) * np.sum(error)
        return dw, db

    def update_weights(self, dw, db):
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, x):
        # 0.5 üstü → 1, altı → 0
        y_hat = self.forward(x)
        return (y_hat >= 0.5).astype(int)

    def train(self, x, y):
        loss_history = []

        for i in range(self.iterations):
            y_hat = self.forward(x)
            loss = self.compute_loss(y, y_hat)
            loss_history.append(loss)

            dw, db = self.compute_gradients(x, y, y_hat)
            self.update_weights(dw, db)

        return loss_history
