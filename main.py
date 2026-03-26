# main.py
# Entry point — modeli burada çalıştırıyoruz.

from data import generate_linear_data, generate_logistic_data
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from plotter import plot_linear_results, plot_logistic_results

# --- Linear Regression ---
x, y = generate_linear_data()

model = LinearRegression(learning_rate=0.01, iterations=1000)
loss_history = model.train(x, y)

print("--- Linear Regression ---")
print(f"Learned w: {model.w:.4f}  (expected ~2.0)")
print(f"Learned b: {model.b:.4f}  (expected ~1.0)")
print(f"Final loss: {loss_history[-1]:.4f}")

plot_linear_results(x, y, model, loss_history)

# --- Logistic Regression ---
x2, y2 = generate_logistic_data()

model2 = LogisticRegression(learning_rate=0.1, iterations=1000)
loss_history2 = model2.train(x2, y2)

predictions = model2.predict(x2)
accuracy = (predictions == y2).mean() * 100

print("\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Final loss: {loss_history2[-1]:.4f}")

plot_logistic_results(x2, y2, model2, loss_history2)
