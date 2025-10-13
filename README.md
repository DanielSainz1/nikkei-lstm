\# Nikkei-225 Forecasting with LSTM



\[!\[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielSainz1/nikkei-lstm/blob/main/notebooks/01\_lstm\_nikkei.ipynb)



\*\*Individual project – Neural Networks \& Deep Learning (grade: 9/10)\*\*  

\*\*Stack:\*\* Python, TensorFlow/Keras, scikit-learn, pandas, yfinance, Jupyter, Matplotlib



---



\## 1) Objective

Forecast the \*\*daily close\*\* of the Nikkei-225 using a sequence model and evaluate \*\*out-of-sample\*\* performance with business-relevant metrics.



\## 2) Data \& Split

\- Historical daily close from Yahoo Finance via `yfinance` (`^N225`).

\- \*\*Sliding window:\*\* 60 \*\*trading\*\* days → next-day close (\*sequence-to-one\*).

\- \*\*Train/Val/Test:\*\* 2021–2022 for training/validation; \*\*2023\*\* held-out for test.

\- \*\*Scaling:\*\* `MinMaxScaler` fitted on \*\*train only\*\*, then applied to val/test (to avoid leakage).



\## 3) Model \& Training

\- \*\*Architecture:\*\* 3× LSTM (64 units) + dropout + dense (\*\*~85k params\*\*).

\- \*\*Loss/Opt:\*\* MSE with Adam; learning curves tracked.

\- \*\*Regularization:\*\* dropout + \*\*EarlyStopping\*\* on validation loss.



\## 4) Evaluation

\- \*\*Primary metric:\*\* \*\*MAPE ≈ 8.3%\*\* on the \*\*2023\*\* test set.

\- \*\*Additional signals:\*\* validation-loss minimum around \*\*epoch ~54\*\*; plots of \*\*actual vs. predicted\*\* and \*\*residuals\*\* to assess drift and fit stability.



\## 5) Baselines (uplift)

\- \*\*Naïve last-value\*\* and \*\*Moving Average (k = 5, 10, 20)\*\* using only past data.

\- \*\*Result:\*\* LSTM improves MAPE vs. Naïve by \*\*+X%\*\* and vs. MA-10 by \*\*+Y%\*\* on 2023. \*(Replace X/Y once computed.)\*



\## 6) What mattered technically

\- Correct handling of \*\*time-series leakage\*\* (scaler trained on train only).

\- \*\*Reproducible\*\* pipeline (pandas/Jupyter) and clear plots.

\- Sensible \*\*sequence length\*\* (60) balancing memory and short-term signal.

\- Detection of \*\*overfitting\*\* after prolonged training → EarlyStopping.



\## 7) Results \& Next steps

\- Achieved \*\*MAPE ~8.3% OOS (2023)\*\*; model tracks trend and short-term dynamics.

\- Next: multivariate features (\*\*OHLCV\*\*, returns, MAs, RSI), \*\*ReduceLROnPlateau\*\*, and walk-forward CV.



---



\## Reproduce



```bash

pip install -r requirements.txt



\# 1) Download data \& create splits

python -m src.data



\# 2) (Optional) Train model from scripts

python -m src.model



\# 3) Evaluate baselines / produce plots

python -m src.eval



