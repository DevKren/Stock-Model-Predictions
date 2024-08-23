# Stock-Model-Predictions
Written by: Anthony Krenek
Predictions made by: Anthony Krenek


** IMPORTANT DO NOT USE AS FINANCIAL ADVICE THESE ARE JUST PREDICTIONS BASED OFF OF DATA AND NOTHING ELSE **


Model: Long Short Term Model.
The Long Short Term Memory is a deep learning neural network model that is best used to predict future numbers. I used this to predict the stock price of companies. I taught myself how to make this model and it took awhile to learn and even harder to debug.
I found this model by looking up the models that are most commonly used for data forecasting. 
Libraries for LSTM: 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
Numpy was also used to make calculations: 

Libraries for Finding Data: 
yfinance as yf 
pandas as pd 

I have always wanted to make some sort of model that will predict stock prices for companies. Able to use any S&P 500 Company
yfinance allows you to scrape the data from within the python file which I thought was super cool



Goals: 
Be able to find somewhat accurate predictions for companies by succesfully using Linear Regression and my LSTM model. 
Find helpful trends in the stocks data 
Learn more about Machine Learning
Get out of my comfort zone in terms of coding 

Data Intake: 
Using Yfinance to pull values and write to csv values of actual data
Data depends on how long out you are looking for and how accurate
Most of the time I used about 25 months of past stock info grabbed from Yfinance 
Still need to play around with different data sizes and epochs 

Training Model: 
X_noc = noc_data[['Lag_1']] (Training on the last days close) 
y_noc = noc_data['Close'] (Finding Average Close Price)

LSTM Model Compiling: 
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_lstm, y_noc, epochs=100, batch_size=1, verbose=2)


loss = mean_squared_error which specifes the model to train on the MSE of predictied and actual values.
optimizer = Adam which is the neural network and will update the parameters during training 

X_lstm: This is the input data for the model. It is expected to be a tensor or array of shape (samples, time steps, features).
y_noc: This is the target data for the model. It is expected to be a tensor or array of shape (samples, output dimension).

epochs=100: This specifies the number of times the complete training dataset will be passed through the network.
batch_size=1: This specifies the number of samples to process per gradient update. Setting it to 1 means that the model will be trained using Stochastic Gradient Descent (SGD).
verbose=2: This specifies the verbosity level for training. Setting it to 2 will print progress bar logs to monitor the training process.

Data Storage: 
Wrote the "stock ticker"_future_predictions stores the values as a list then we iterate by how many days we want to forecast for our stocks
Then wrote all the future predictions to a csv file for comparision to the actual values vcan be used on (High,Low,Close,Volume,Open).
Example: 
For the NOC Model I was trying to forecast for 15 months in advanced because I gave it about just over 24 months of data that it was trained on. 

Using Google Collab made this a lot easier by mounting my drive and storing the future and actual data as csv's. 

Setbacks: 
This took me about a week to get succesful through this I learned many different strategies to program.
I learned that by mounting my google drive and reading in the files individually worked a lot better than trying to do two in the same program. I was trying to do two stocks at once and one would pass. Then the other would do some weird data manipulation and it was stroing the future data as an ndarray of float type 32. I eventually broke the two stocks into different files and realized the problem. The problem was I was losing my data for one of the stocks in the nd array. This took me over 15+ hours to debug, but I am very glad that I did. 

Takeaways: 
Being organized with code will save you a ton of time. I am very glad to know that now because I was unorgainzed causing me to spend countless hours on debugging. 
It is awesome to learn different libraries. Outside of Numpy and Pandas these are libraires that I have never used before especially the sklearn and keras libraries. I taught these to myself and it was super intresting to learn. 
Lastly, not being afraid to write hard code and things that I don't know about. I like stocks and this seemed really intresting to me. 

Final Thoughts:
I am very glad that I did this project, but I am not done yet. I will keep updating as time goes on to see how my predictions compare to the actual stock market prices. Stay tuned!
