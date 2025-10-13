\# Nikkei-225 Forecasting with LSTM



\*\*Individual project – Neural Networks \& Deep Learning (grade: 9/10)\*\*  

Stack: Python, TensorFlow/Keras, scikit-learn, pandas, yfinance, Jupyter, Matplotlib



\## Objective

Forecast the \*\*daily close\*\* of the Nikkei-225 using a sequence model and evaluate \*\*out-of-sample\*\* performance with business-relevant metrics.



\## Data \& Split

\- Source: Yahoo Finance (`^N225`) via `yfinance`.

\- Sliding window: \*\*60 trading days → next-day close\*\* (\*sequence-to-one\*).

\- Train/Val/Test: \*\*2021–2022\*\* for training/validation; \*\*2023\*\* held out for test.

\- Scaling: `MinMaxScaler` \*\*fitted on train only\*\*, then applied to val/test (avoid leakage).



\## Model \& Training

\- Architecture: \*\*3× LSTM (64 units)\*\* + dropout + dense (~\*\*85k params\*\*).

\- Loss/Opt: \*\*MSE\*\* with \*\*Adam\*\*; learning curves tracked.

\- Regularization: dropout + \*\*EarlyStopping\*\* on validation loss.



\## Evaluation

\- \*\*MAPE ≈ 8.3%\*\* on the \*\*2023\*\* test set.

\- Additional signals: validation-loss minimum around \*\*epoch ~54\*\*; plots of \*\*actual vs. predicted\*\* and \*\*residuals\*\*.



\## Baselines (uplift)

\- \*\*Naïve last-value\*\* and \*\*Moving Average (k=5, 10, 20)\*\* using only past data.

\- Result: LSTM improves MAPE vs. Naïve by \*\*+X%\*\* and vs. MA-10 by \*\*+Y%\*\* on 2023. \*(actualiza X/Y cuando lo calcules).\*



\## Reproduce

\- Notebook: `notebooks/01\_lstm\_nikkei.ipynb`  

\- Full report (PDF): `reports/ProyectoFinalRedes.pdf`



