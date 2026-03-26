import numpy as np

def generate_linear_data():
    np.random.seed(42)  # aynı veriyi üretmek için seed sabitliyoruz

    x = np.linspace(0, 10, 50)  # 0-10 arası 50 nokta

    # gerçek denklem: y = 2x + 1, üstüne noise ekliyoruz
    noise = np.random.randn(50)
    y = 2 * x + 1 + noise

    return x, y

def generate_logistic_data():
    np.random.seed(42)

    # İki grup veri: 0 sınıfı 0-4 civarı, 1 sınıfı 6-10 civarı
    x0 = np.random.normal(loc=2, scale=1, size=50)  # sınıf 0
    x1 = np.random.normal(loc=8, scale=1, size=50)  # sınıf 1

    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(50), np.ones(50)])  # etiketler: 0 ve 1

    return x, y
