import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        # Hiperparametreler — model eğitilmeden önce biz belirliyoruz
        self.learning_rate = learning_rate
        self.iterations = iterations

        # w ve b'yi sıfırdan başlatıyoruz, model bunları kendisi öğrenecek
        self.w = 0.0
        self.b = 0.0

    def forward(self, x):
        # Tahmin üret: y_hat = w*x + b
        # This is the core equation of linear regression
        y_hat = self.w * x + self.b
        return y_hat

    def compute_gradients(self, x, y, y_hat):
        n = len(y)

        # MSE'nin w'ya göre türevi
        # dL/dw = (-2/n) * sum(x * (y - y_hat))
        dw = (-2 / n) * np.sum(x * (y - y_hat))

        # MSE'nin b'ye göre türevi
        # dL/db = (-2/n) * sum(y - y_hat)
        db = (-2 / n) * np.sum(y - y_hat)

        return dw, db

    def update_weights(self, dw, db):
        # Gradient'in tersine adım at — aşağı in
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def train(self, x, y):
        loss_history = []  # her iterasyondaki loss'u kaydediyoruz, sonra plot edeceğiz

        for i in range(self.iterations):
            y_hat = self.forward(x)

            # loss hesapla — ne kadar yanıldık?
            loss = np.mean((y - y_hat) ** 2)
            loss_history.append(loss)

            # gradients → weight update
            dw, db = self.compute_gradients(x, y, y_hat)
            self.update_weights(dw, db)

        return loss_history
