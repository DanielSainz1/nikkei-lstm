import yfinance as yf
import pandas as pd

def load_nikkei():
    df = yf.download("^N225", start="2020-01-01", progress=False)[["Close"]]
    df = df.dropna().asfreq("B").ffill()  # business days
    df.index.name = "Date"
    return df

def split_series(df, train_end="2022-12-30", test_start="2023-01-02"):
    y = df["Close"].copy()
    y_train = y[:train_end]
    y_test  = y[test_start:]
    return y_train, y_test

if __name__ == "__main__":
    df = load_nikkei()
    y_train, y_test = split_series(df)
    df.to_csv("data_nikkei.csv")
    y_train.to_csv("y_train.csv")
    y_test.to_csv("y_test.csv")
