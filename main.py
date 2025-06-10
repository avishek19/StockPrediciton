

import sys
import yfinance as yf
import argparse
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore, Style, Back
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
import streamlit as st

USERS = {
    "user1": "pass123",
    "admin": "admin",
    "professor Mr. Frank": "admin"
}


# --- Authentication Page ---
def show_login_page():
    st.header("User Authentication Page")
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    if (username=="admin"):
        username = "professor Mr. Frank"
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun() # Rerun to switch to main app
        else:
            st.sidebar.error("Invalid Username or Password")

# --- Main Application Page ---
def show_main_app():
    st.title(f"Welcome, {st.session_state.username}!")
    st.header("Stock Forecast System")

    st.write("Enter stock symbol to gather LLM prediction:",'\n')
    ticker_input = st.text_input("Stock Symbol")    

    if st.button("Forecast short-term prediction"):
        if ticker_input:
            ticker = ticker_input.upper()
            print_header(ticker)
            print(Fore.YELLOW + f"Fetching data for {ticker}..." + Style.RESET_ALL)
            data, info = get_stock_data(ticker)
            print('Feeding data to LLM after readin yahoo: ','\n',data['daily'].iloc[-1])
            print('*** Feeding stock info to LLM \n',info['longBusinessSummary'],'\n\n','***')
            llm_response = get_llm_prediction(ticker, data['daily'].iloc[-1], info)
            print('**** LLM Forecast *****','\n',llm_response)
            
            
            if data and data.get('daily') is not None and not data['daily'].empty and info:
                # print('Feeding data to LLM after readin yahoo: ','\n',data['daily'].iloc[-1]) # Console
                # print('*** Feeding stock info to LLM \n',info['longBusinessSummary'],'\n\n','***') # Console
                with st.spinner("Generating LLM prediction..."):
                    llm_response = get_llm_prediction(ticker, data['daily'].iloc[-1], info)
                
                # print('**** LLM Forecast *****','\n',llm_response) # Console
                if llm_response:
                    st.subheader("LLM Forecast:")
                    st.markdown(llm_response)
                else:
                    st.error("Could not retrieve LLM prediction. This might be due to an API issue or missing data.")
            else:
                st.error(f"Failed to fetch sufficient data for {ticker} to make a prediction.")
        else:
            st.warning("Please enter a stock symbol.")
            exit(0)
        # --- Logout Button ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun() # Rerun to go back to login page    
    return 



def get_llm_prediction(ticker: str, latest_daily_data: Optional[pd.Series], company_info: Optional[Dict]) -> Optional[str]:
    """
    Generates a stock prediction using an LLM.
    Note: LLM predictions are speculative and not financial advice.
    """
    if latest_daily_data is None or latest_daily_data.empty:
        return "Not enough data for LLM prediction."

    load_dotenv()  # load variables from .env

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(Fore.RED + "OPENAI_API_KEY environment variable not set. Skipping LLM prediction." + Style.RESET_ALL)
        return None
    
    client=OpenAI(api_key=api_key)


    def format_value(value, precision=2):
        if pd.isna(value):
            return "N/A"
        return f"{value:.{precision}f}"

    prompt_data = f"Stock Ticker: {ticker}\n"
    prompt_data += f"Latest Close: {format_value(latest_daily_data.get('Close'))}\n"
    prompt_data += f"RSI: {format_value(latest_daily_data.get('RSI'))}\n"
    prompt_data += f"SMA_20: {format_value(latest_daily_data.get('SMA_20'))}\n"
    prompt_data += f"SMA_50: {format_value(latest_daily_data.get('SMA_50'))}\n"
    prompt_data += f"VWAP: {format_value(latest_daily_data.get('VWAP'))}\n"

    if company_info:
        prompt_data += f"Company: {company_info.get('shortName', ticker)}\n"
        prompt_data += f"Sector: {company_info.get('sector', 'N/A')}\n"
        prompt_data += f"Summary (first 200 chars): {company_info.get('longBusinessSummary', 'N/A')[:200]}...\n"

    system_prompt = "You are a financial analyst AI. Based on the provided technical indicators and company information, offer a speculative short-term outlook (e.g., next 1-5 trading days) for the stock. Briefly state your prediction (e.g., bullish, bearish, neutral) and provide a concise reasoning (1-2 sentences). This is not financial advice."
    user_prompt = f"Analyze the following stock data and provide a short-term prediction:\n{prompt_data}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # You can try other models like "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(Fore.RED + f"Error getting LLM prediction: {str(e)}" + Style.RESET_ALL)
        return None

def get_stock_data(ticker: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Fetch comprehensive stock data"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data for different timeframes
        data = {
            'daily': stock.history(period='1mo', interval='1d'),
            'weekly': stock.history(period='1y', interval='1wk'), 
            'monthly': stock.history(period='5y', interval='1mo')
        }
        
        # Get company info
        info = stock.info
        
        # Calculate technical indicators
        daily_data = data['daily']
        if not daily_data.empty:
            # RSI calculation
            delta = daily_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            daily_data['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving averages
            daily_data['SMA_20'] = daily_data['Close'].rolling(window=20).mean()
            daily_data['SMA_50'] = daily_data['Close'].rolling(window=50).mean()
            
            # Calculate VWAP
            daily_data['VWAP'] = (daily_data['Close'] * daily_data['Volume']).cumsum() / daily_data['Volume'].cumsum()
        
        return data, info
    except Exception as e:
        print(Fore.RED + f"Error fetching data for {ticker}: {str(e)}" + Style.RESET_ALL)
        return None, None

def print_header(ticker: str) -> None:
    """Print formatted header"""
    print(Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "STOCK ANALYSIS: " + Fore.YELLOW + ticker + Style.RESET_ALL)
    print(Fore.CYAN + "=" * 80 + Style.RESET_ALL + "\n")


# --- Main App Logic (Session State Management) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if st.session_state.logged_in:
     show_main_app()
else:
    show_login_page()