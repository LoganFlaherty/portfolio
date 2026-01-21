# Orb Stock Predictor
Orb is a stock market ML tool that fetches up to date data on a specified stock and trains a Gradient Boost Regression ML model to predict the stock's price at the next specified time interval. 
Built using Python 3.11 with libraries sklearn, pandas, matplot, and tastytrade API. TastyTrade is my stock broker that I actually use, so I have access to their API. 
You will not be able to run this tool without having a funded account with TastyTrade. However, I have provided examples of the CSV and graphed predictions of Orb in the results folder.

With access to the tastytrade API, you can run this tool with the "orb.py" script. Even supporting CLI options handling to automatically adjust variables such as "ticker", "steps", and "plot_data". 
Alternatively, these variables can be changed in the code file.

Orb works by first, handling your options:

-t {symbol} : ticker (stock symbol)

-s {num} : steps (how many 5 minute steps it will predict next. For example 3 steps would be a next 15 minute prediciton)

-p : bool if it will plot the predictions or not

Second, Orb will fetch the most recent 90 days worth of market data for the ticker given and save it to a CSV. Third, "model.py" file will execute. This cleans the CSV to be a ML friendly dataframe first, 
and applies indicators to the dataset such as rate of change, EMA, Bollinger Bands, and more as features. These indicators were chosen delicately after a long series of config testing. 
Then, splits the data into training and test data 90-10. Next, it will train on the training data and make predictions based on the test data. 
Every prediction is recorded and plotted at the end into a graph (if applied the option). 
Then it prints some analtical data on the predictions such as directional accuracy (if it moved right), mean absolute error (average the prediction was off by), 
and root mean squared error (like the previous but punishes outlier harder). Finally, it makes its last prediction for the next interval based on the steps integer passed.

As fun as Orb was to build, unfortunately the accuracy of the model ranges from 50-55%, which as a tool is not statically significant. However, the experience of fetching real data, formating the data,
configurating an ML model, and training it has been extremely value to my personal growth as a computer scientist.
