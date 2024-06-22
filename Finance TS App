import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Function to fetch historical data from yfinance
def fetch_data(ticker):
    data = yf.download(ticker, period='5y')
    data.reset_index(inplace=True)
    return data

# Function to plot the time series data
def plot_time_series(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

# Function to perform time series forecasting
def forecast(data, periods):
    data_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(data_prophet)
    future = model.make_future_dataframe(periods=periods)  # Forecast for the selected period
    forecast = model.predict(future)
    return forecast, model

# Streamlit app
st.title("Stock Price Time Series Forecasting")

# Input for stock ticker
stock_ticker = st.text_input("Enter Stock Ticker:", "AAPL")

# Dropdown menu for selecting forecast period
forecast_period = st.selectbox("Select Forecast Period:", ("2 years", "5 years", "10 years", "20 years"))

# Mapping forecast periods to number of days
forecast_periods = {
    "2 years": 365 * 2,
    "5 years": 365 * 5,
    "10 years": 365 * 10,
    "20 years": 365 * 20
}

if stock_ticker and forecast_period:
    # Fetch data based on the stock ticker
    data = fetch_data(stock_ticker)

    # Display raw data
    st.write("### Raw Data")
    st.write(data.head())

    # Plot the time series data
    st.write("### Time Series Data")
    plot_time_series(data, f"Historical Stock Price for {stock_ticker}")

    # Forecast the data
    forecast_data, model = forecast(data, forecast_periods[forecast_period])

    # Display the forecasted data in a data table
    st.write("### Forecasted Data Table")
    st.write(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))  # Display the last 10 predictions

    # Plot the forecast
    st.write("### Forecasted Time Series")
    fig = model.plot(forecast_data)
    plt.title(f'Forecasted Time Series ({forecast_period})')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    st.pyplot(fig)

    # Plot the components
    st.write("### Forecast Components")
    fig2 = model.plot_components(forecast_data)
    st.pyplot(fig2)
