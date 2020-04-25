"""
Title: Stock Price Prediction
Developed  by: Syeda Atiya Husain and Pankaj Lal
Language: Python
Requirement:
            Python version: Python 3 or later
            Libraries:1) django
                      2) pandas
                      3) yahoo_fin 
                      4) sklearn
                      5) numpy
                      6) plotly
"""

from django.shortcuts import render                         #importing necessary libraries.
from django.http import HttpResponse
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.offline as opy
import plotly.graph_objs as go


def predict(quotes_df):
    
    # Make a copy of the dataframe so that the original data remains unchanged.
    df = quotes_df.copy()
     
    # Add the percent change of the daily closing price.
    df['ClosingPctChange'] = df['close'].pct_change()
    
    # Get today's record (the last record) so that we can predict it later. 
    # Do this before we add the 'NextDayPrice' column so that we don't have to drop it later.
    df_today = df.iloc[-1:, :].copy()
   
        
    # Create a column of the next day's closing prices so as we can train on it.
    # and then eventually predict the value.
    df['NextClose'] = df['close'].shift(-1)
    
    #linear Regression
    linreg = LinearRegression()
    
    # Decide which features to use for our regression. 
    features_to_fit = ['open', 'high', 'low', 'close', 'volume']
    
    # Create our target and labels
    X = df[features_to_fit]
    y = df['adjclose']
    
    # Create training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2)

    # Fit the regressor with the full dataset to be used with predictions
    linreg.fit(X_train, y_train)

    # compute our average accuracy
    ac=linreg.score(X_test, y_test)

    # Predict today's closing price
    X_new = df_today[features_to_fit]
    next_price_prediction = linreg.predict(X_new)
    
    #graphs
    fig = go.Figure(data=[go.Bar(name='Actual', x=df_today.index, y=df_today['close']),go.Bar(name='Predictied', x=df_today.index, y=[next_price_prediction[0]])])
    fig.update_layout(barmode='group')
    div = opy.plot(fig, auto_open=False, output_type='div')
    
    # Return the predicted closing price
    return next_price_prediction,ac,div

def index(request):
    if request.method == 'POST': 
        comp=request.POST.get('Company')
        quotes_df=si.get_data(comp.upper())
        C=comp.upper()
        pred,ac,fig=predict(quotes_df)
        current_price=si.get_live_price(comp.upper())
        params={'comp':C,'Prediction':pred[-1:],'Accuracy':ac,'Live':current_price,'graph':fig}
        return render(request,'StockMarketPrediction/prediction.htm',params)
    else:
        return render(request,'StockMarketPrediction/index.htm')

def features(request):
    return render(request,'StockMarketPrediction/features.htm')

def contact(request):
    return render(request,'StockMarketPrediction/contact.htm')
