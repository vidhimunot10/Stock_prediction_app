import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set page title and configuration
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")

# Application title and description
st.title("Stock Price Prediction App")
st.markdown("This application allows you to view historical stock prices and predict future trends.")

# Define a list of popular stocks
stock_list = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Facebook (Meta)": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA"
}

# Create sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Let user select a stock
selected_stock_name = st.sidebar.selectbox("Select a stock", list(stock_list.keys()))
selected_stock_symbol = stock_list[selected_stock_name]

# User input for number of days to display
days_to_display = st.sidebar.number_input("Enter number of days to display:", min_value=1, max_value=365, value=30)

# User input for number of days to predict
days_to_predict = st.sidebar.number_input("Enter number of days to predict:", min_value=1, max_value=30, value=7)

# Moving average periods
st.sidebar.subheader("Moving Averages")
ma_short = st.sidebar.number_input("Short MA Period:", min_value=5, max_value=50, value=7)
ma_medium = st.sidebar.number_input("Medium MA Period:", min_value=10, max_value=100, value=20)
ma_long = st.sidebar.number_input("Long MA Period:", min_value=20, max_value=200, value=50)

# Display selected stock information
st.header(f"Showing data for: {selected_stock_name} ({selected_stock_symbol})")

# Current date and calculation of start date - get more data for moving averages
end_date = datetime.datetime.now()  # Use full datetime object, not just date
# Get more historical data to compute moving averages properly
extended_start_date = end_date - datetime.timedelta(days=days_to_display + ma_long)

# Function for simple linear regression prediction
def predict_prices(data, days_to_predict):
    # Extract the closing prices
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Create a sequence of days
    days = np.arange(len(close_prices)).reshape(-1, 1)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(days, close_prices)
    
    # Generate days for prediction
    future_days = np.arange(len(close_prices), len(close_prices) + days_to_predict).reshape(-1, 1)
    
    # Predict prices
    future_prices = model.predict(future_days)
    
    # Convert to list of floats to avoid array conversion issues
    future_prices_list = [float(price[0]) for price in future_prices]
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days_to_predict)]
    
    # Create a DataFrame for predictions
    prediction_df = pd.DataFrame({
        'Close': future_prices_list
    }, index=future_dates)
    
    return prediction_df

# Button to submit and show data
if st.sidebar.button("Show Stock Data"):
    # Load stock data
    try:
        # Get stock data with extended period for moving averages
        extended_stock_data = yf.download(selected_stock_symbol, start=extended_start_date, end=end_date)
        
        if extended_stock_data.empty:
            st.error("No data available for the selected stock and time period.")
        else:
            # Calculate moving averages on the extended data
            extended_stock_data[f'MA_{ma_short}'] = extended_stock_data['Close'].rolling(window=ma_short).mean()
            extended_stock_data[f'MA_{ma_medium}'] = extended_stock_data['Close'].rolling(window=ma_medium).mean()
            extended_stock_data[f'MA_{ma_long}'] = extended_stock_data['Close'].rolling(window=ma_long).mean()
            
            # Filter data to the requested display period
            display_start_date = end_date - datetime.timedelta(days=days_to_display)
            # Use pandas datetime comparison instead of Python datetime
            stock_data = extended_stock_data[extended_stock_data.index >= pd.Timestamp(display_start_date)]
            
            # Display stock data
            st.subheader("Stock Price Data")
            st.dataframe(stock_data)
            
            # Display statistics
            st.subheader("Summary Statistics")
            st.write(stock_data.describe())
            
            # Make predictions
            prediction_data = predict_prices(stock_data, days_to_predict)
            
            # Create and display charts
            st.subheader("Stock Price Charts")
            
            try:
                # Create multiple figures for different analyses
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                
                # Convert to simple lists to avoid numpy array issues
                historical_dates = list(stock_data.index)
                historical_prices = [float(price) for price in stock_data['Close'].values]
                prediction_dates = list(prediction_data.index)
                prediction_prices = list(prediction_data['Close'].values)
                
                # Historical closing price and prediction
                ax1.plot(historical_dates, historical_prices, 'b-', label='Historical Close Price')
                ax1.plot(prediction_dates, prediction_prices, 'r--', label='Predicted Close Price')
                
                # Plot moving averages
                ax1.plot(historical_dates, [float(ma) if not np.isnan(ma) else None for ma in stock_data[f'MA_{ma_short}'].values], 
                         'g-', label=f'{ma_short}-Day MA')
                ax1.plot(historical_dates, [float(ma) if not np.isnan(ma) else None for ma in stock_data[f'MA_{ma_medium}'].values], 
                         'orange', label=f'{ma_medium}-Day MA')
                ax1.plot(historical_dates, [float(ma) if not np.isnan(ma) else None for ma in stock_data[f'MA_{ma_long}'].values], 
                         'purple', label=f'{ma_long}-Day MA')
                
                ax1.set_title(f'{selected_stock_name} Close Price with Moving Averages')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price (USD)')
                ax1.legend()
                ax1.grid(True)
                
                st.pyplot(fig1)
                
                # Volume and price chart
                fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Price chart
                ax2.plot(historical_dates, historical_prices, 'b-', label='Close Price')
                ax2.set_title(f'{selected_stock_name} Close Price')
                ax2.set_ylabel('Price (USD)')
                ax2.legend()
                ax2.grid(True)
                
                # Trading volume - convert to list of standard Python types
                volume_data = [float(vol) for vol in stock_data['Volume'].values]
                
                # Trading volume over time
                ax3.bar(historical_dates, volume_data, color='g', alpha=0.7, label='Volume')
                ax3.set_title(f'{selected_stock_name} Trading Volume')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Volume')
                ax3.legend()
                ax3.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Moving Average Convergence Divergence (MACD) chart
                fig3, ax4 = plt.subplots(figsize=(12, 6))
                
                # Calculate MACD
                stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
                stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
                stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
                stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
                stock_data['Histogram'] = stock_data['MACD'] - stock_data['Signal']
                
                # Plot MACD
                macd_values = [float(macd) for macd in stock_data['MACD'].values]
                signal_values = [float(signal) for signal in stock_data['Signal'].values]
                histogram_values = [float(hist) for hist in stock_data['Histogram'].values]
                
                ax4.plot(historical_dates, macd_values, 'b-', label='MACD')
                ax4.plot(historical_dates, signal_values, 'r-', label='Signal')
                
                # Plot histogram
                for i, date in enumerate(historical_dates):
                    if i < len(histogram_values):
                        if histogram_values[i] >= 0:
                            ax4.bar(date, histogram_values[i], color='g', width=0.5, alpha=0.4)
                        else:
                            ax4.bar(date, histogram_values[i], color='r', width=0.5, alpha=0.4)
                
                ax4.set_title(f'{selected_stock_name} MACD Indicator')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Value')
                ax4.legend()
                ax4.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig3)
                
                # RSI Chart
                fig4, ax5 = plt.subplots(figsize=(12, 6))
                
                # Calculate RSI (14-day)
                delta = stock_data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                stock_data['RSI'] = 100 - (100 / (1 + rs))
                
                # Plot RSI
                rsi_values = [float(rsi) if not np.isnan(rsi) else None for rsi in stock_data['RSI'].values]
                
                ax5.plot(historical_dates, rsi_values, 'b-', label='RSI')
                
                # Add overbought/oversold lines
                ax5.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax5.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                
                ax5.fill_between(historical_dates, 70, [100] * len(historical_dates), color='r', alpha=0.1)
                ax5.fill_between(historical_dates, 0, [30] * len(historical_dates), color='g', alpha=0.1)
                
                ax5.set_title(f'{selected_stock_name} Relative Strength Index (14-day)')
                ax5.set_xlabel('Date')
                ax5.set_ylabel('RSI Value')
                ax5.set_ylim(0, 100)
                ax5.legend()
                ax5.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig4)
                
            except Exception as chart_error:
                st.error(f"Error generating charts: {chart_error}")
                st.error(f"Chart error details: {chart_error._class.name_}")
            
            # Display prediction table
            st.subheader(f"Price Predictions for Next {days_to_predict} Days")
            st.dataframe(prediction_data)
            
            # Price change calculation for historical data
            first_price = float(stock_data['Close'].iloc[0])
            last_price = float(stock_data['Close'].iloc[-1])
            price_change = ((last_price - first_price) / first_price) * 100
            
            # Display historical price change information
            st.subheader("Historical Price Change Analysis")
            st.write(f"Starting price: ${first_price:.2f}")
            st.write(f"Ending price: ${last_price:.2f}")
            
            if price_change > 0:
                st.success(f"Price increased by {price_change:.2f}% over the selected period.")
            elif price_change < 0:
                st.error(f"Price decreased by {abs(price_change):.2f}% over the selected period.")
            else:
                st.info("Price remained unchanged over the selected period.")
            
            # Price change calculation for prediction - with explicit type conversion
            predicted_first_price = float(prediction_data['Close'].iloc[0])
            predicted_last_price = float(prediction_data['Close'].iloc[-1])
            predicted_change = ((predicted_last_price - predicted_first_price) / predicted_first_price) * 100
            
            # Display predicted price change information
            st.subheader("Predicted Price Change Analysis")
            st.write(f"Predicted starting price: ${predicted_first_price:.2f}")
            st.write(f"Predicted ending price: ${predicted_last_price:.2f}")
            
            if predicted_change > 0:
                st.success(f"Price predicted to increase by {predicted_change:.2f}% over the next {days_to_predict} days.")
            elif predicted_change < 0:
                st.error(f"Price predicted to decrease by {abs(predicted_change):.2f}% over the next {days_to_predict} days.")
            else:
                st.info(f"Price predicted to remain unchanged over the next {days_to_predict} days.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Debugging information: Please check your input parameters and try again.")

# Additional information
st.markdown("---")
st.markdown("### How to use this app")
st.markdown("""
1. Select a stock from the dropdown menu in the sidebar
2. Enter the number of days of historical data you want to see
3. Enter the number of days you want to predict into the future
4. Adjust the Moving Average periods (short, medium, long) as needed
5. Click the "Show Stock Data" button to display the results
6. Analyze the various charts and technical indicators to make informed decisions

*Technical Indicators Included:*
- Multiple Moving Averages (customizable periods)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)

*Note:* The prediction is based on a simple linear regression model and should be used for educational purposes only. Real stock markets are influenced by many complex factors.
""")