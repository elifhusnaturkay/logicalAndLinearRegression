import matplotlib.pyplot as plt
import numpy as np

def plot_linear_results(x, y, model, loss_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sol grafik: veri noktaları + modelimizin öğrendiği doğru
    ax1.scatter(x, y, color="steelblue", label="Gerçek veri")
    y_hat = model.forward(x)
    ax1.plot(x, y_hat, color="red", label=f"Tahmin: y = {model.w:.2f}x + {model.b:.2f}")
    ax1.set_title("Linear Regression — Fit")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    # Sağ grafik: her iterasyonda loss nasıl düştü?
    ax2.plot(loss_history, color="orange")
    ax2.set_title("Loss over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE Loss")

    plt.tight_layout()
    plt.show()

def plot_logistic_results(x, y, model, loss_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sol grafik: veri noktaları + sigmoid eğrisi
    x_sorted = np.linspace(x.min(), x.max(), 200)
    y_curve = model.forward(x_sorted)

    ax1.scatter(x, y, color="steelblue", alpha=0.5, label="Gerçek etiketler (0 / 1)")
    ax1.plot(x_sorted, y_curve, color="red", label="Sigmoid curve")
    ax1.axhline(0.5, color="gray", linestyle="--", label="Karar sınırı (0.5)")
    ax1.set_title("Logistic Regression — Sigmoid Fit")
    ax1.set_xlabel("x")
    ax1.set_ylabel("P(y=1)")
    ax1.legend()

    # Sağ grafik: BCE loss
    ax2.plot(loss_history, color="orange")
    ax2.set_title("Loss over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("BCE Loss")

    plt.tight_layout()
    plt.show()
