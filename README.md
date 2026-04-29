# Stock Price Prediction & Telegram Alert System

This project is a stock monitoring and forecasting system built for the Vietnamese stock market. It combines machine learning, technical indicators, real-time data tracking, and Telegram alerts to support faster and more data-driven investment decisions.

## Project Overview

The system tracks 20 selected stocks across 5 key sectors in Vietnam. It uses historical price data from FiinQuant and applies an LSTM model together with rule-based technical signals such as SMA, MACD, RSI, and OBV.

The main goal of this project is not only to predict price trends, but also to help investors react quickly when important trading signals appear.

## Key Features

- Stock price forecasting using LSTM
- Technical signal detection using SMA, MACD, RSI, and OBV
- Real-time Telegram alerts for buy/sell signals
- Detection of price peaks, bottoms, and unusual volume movements
- Streamlit dashboard for visualizing forecasts and trading signals
- Profit/loss tracking with thresholds such as:
  - Take-profit: +30%
  - Stop-loss: -10% / -20%

## Tech Stack

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Streamlit
- Telegram Bot API
- FiinQuant data
- Technical indicators: SMA, MACD, RSI, OBV

## How It Works

1. Collect historical and real-time stock data from FiinQuant.
2. Clean and process price, volume, and technical indicator data.
3. Train an LSTM model to forecast stock price trends.
4. Combine model output with rule-based technical signals.
5. Send automatic Telegram alerts when trading signals are detected.
6. Display results through an interactive Streamlit dashboard.

## Project Purpose

This project was developed to improve decision-making in stock analysis by combining long-term price forecasting with short-term market alerts. Instead of manually checking charts, users can receive timely signals and monitor selected stocks through a simple dashboard.

## Future Improvements

- Add backtesting for trading strategies
- Improve model accuracy with more market features
- Optimize signals by sector
- Add portfolio performance tracking
- Deploy the dashboard online
