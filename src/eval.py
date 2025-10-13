import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

if __name__ == "__main__":
    y_train = pd.read_csv("y_train.csv", index_col=0, parse_dates=True)["Close"]
    y_test  = pd.read_csv("y_test.csv",  index_col=0, parse_dates=True)["Close"]

    # Naïve: today's forecast = yesterday's actual
    y_pred_naive = y_test.shift(1)
    y_pred_naive.iloc[0] = y_train.iloc[-1]

    print(f"Naïve — MAPE: {mape(y_test, y_pred_naive):.2f}% | RMSE: {rmse(y_test, y_pred_naive):.2f}")

    # Plot (añade y_pred_lstm cuando lo tengas)
    plt.figure(figsize=(10,4))
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, y_pred_naive.values, label="Naïve", alpha=0.7)
    plt.title("Nikkei-225 — Test period (2023)")
    plt.legend(); plt.tight_layout()
    plt.savefig("reports/figures/actual_vs_pred.png", dpi=160)
