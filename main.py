from data import generate_linear_data
from linear_regression import LinearRegression
from plotter import plot_linear_results

# Veriyi üret
x, y = generate_linear_data()

# Modeli oluştur ve eğit
model = LinearRegression(learning_rate=0.01, iterations=1000)
loss_history = model.train(x, y)

# Sonuçları yazdır — w=2, b=1'e yakın olmalı
print(f"Learned w: {model.w:.4f}  (expected ~2.0)")
print(f"Learned b: {model.b:.4f}  (expected ~1.0)")
print(f"Final loss: {loss_history[-1]:.4f}")

# Sonuçları görselleştir
plot_linear_results(x, y, model, loss_history)
