import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def new(ticker, interval, period, steps, plot_data: bool):
    print(f"Downloading {interval} data for {ticker} (Last {period})...")
    df = get_csv(ticker)
    
    print("Preping Data...")
    df = add_technical_indicators(df)
    X, y, features, X_predict = prepare_ml_data(df, steps)
    
    print("Training Model...")
    model, X_test, y_test = train_model(X, y)

    print("Analyzing Performance...")
    predictions = evaluate(model, X_test, y_test)
    predict_next_step(model, X_predict, steps, features, ticker)

    if plot_data: plot(y_test, predictions, ticker)

def get_csv(ticker):
    ticker = ticker.lower()
    df = pd.read_csv(f"results/{ticker}.csv")
    df = df.iloc[:-1] # Drop last row
    df.columns = df.columns.str.replace(f"{ticker.upper()}: ", "")

    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    except:
        pass
    df.set_index('Timestamp', inplace=True)

    df.sort_values(by='Timestamp', ascending=True, inplace=True)
    df['Volume'] = df['Volume'].apply(convert_volumn).astype(int)
    df.to_csv(f"results/{ticker}.csv")
    return df

def convert_volumn(vol):
    if isinstance(vol, str):
        if 'K' in vol:
            return float(vol.replace('K', '')) * 1_000
        elif 'M' in vol:
            return float(vol.replace('M', '')) * 1_000_000
        elif 'B' in vol:
            return float(vol.replace('B', '')) * 1_000_000_000
        else:
            return float(vol)
    return vol

def add_technical_indicators(df):
    ## Rate of Change (ROC) indicator
    df['ROC'] = ((df['Close'] - df['Close'].shift(4)) / df['Close'].shift(4)) * 100

    ## Bollinger Band Width (Volatility/Expansion Feature)
    # Measures how tight the bands are, indicating low volatility that often precedes a move.
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Std'] * 2) / df['BB_Mid'] # Standardized bandwidth

    ## Stochastic Oscillator (%K and %D)
    low_min = df['Low'].rolling(window=8).min()
    high_max = df['High'].rolling(window=8).max()
    
    # Calculate %K (Fast Stochastic)
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    
    # Calculate %D (Slow Stochastic - SMA of %K)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=5).mean()

    ## EMA
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    return df

def prepare_ml_data(df, steps):
    df = df.copy()
    X_predict = df.iloc[[-1]]
    df['Target'] = df['Close'].shift(-steps)
    df = df.dropna()

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'ROC', 'BB_Width', 'Stoch_K', 'Stoch_D',
                    'EMA_9', 'EMA_20']
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y, feature_cols, X_predict

def train_model(X, y):
    # Split data (90% train, 10% test)
    split = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train
    model = GradientBoostingRegressor(n_estimators=75, learning_rate=0.2, min_samples_split=150, random_state=25) # 75, 0.2, 150, 25 seem optimal
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    # Directional accuracy
    current_prices = X_test['Close']
    actual_direction = y_test.values > current_prices.values
    predicted_direction = predictions > current_prices.values
    correct_moves = actual_direction == predicted_direction
    direction_accuracy = correct_moves.mean() * 100

    print(f"\n--- Model Testing Performance: ---")
    print(f"Mean Absolute Error:     ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"Directional Accuracy:    {direction_accuracy:.2f}%")
    print(f"----------------------------------")

    return predictions

def plot(y_test, predictions, ticker):
    # Plot only the last 6 days (approx 468 periods of 5m)
    subset_len = 468
    if len(y_test) < subset_len: subset_len = len(y_test)
    
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#1f1f1f'
    plt.rcParams['figure.facecolor'] = '#1f1f1f'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#444444'

    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='#04d9ff', linewidth=1)
    plt.plot(y_test.index, predictions, label='Prediction', color='#f72119', linestyle='--', linewidth=1)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=24))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gcf().autofmt_xdate()

    plt.title(f"{ticker} 5-Minute Predictions (6 Trading Days)")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/{ticker.lower()}_prediction_graph")
    plt.show()

def predict_next_step(model, X_predict, steps, feature_cols, ticker):
    last_time = X_predict.index[-1]
    X_predict['Next_5m_PtC'] = 0.0
    current_price = round(X_predict['Close'].iloc[-1].item(), 2)
    next_price = round(model.predict(X_predict.iloc[[-1]][feature_cols])[0], 2)
    
    print(f"\n--------- {ticker} Prediction Report ---------")
    print(f"Latest Data Point: {last_time}")
    print(f"Current Price:     ${current_price:.2f}")
    print(f"Next {5 * steps}m Close:     ${next_price:.2f}")
    
    diff = next_price - current_price
    pct = (diff / current_price) * 100
    if diff >= 0: print(f"Signal: UP (+${diff:.2f} | +{pct:.2f}%)")
    else: print(f"Signal: DOWN (${diff:.2f} | {pct:.2f}%)")
    print("-" * 42)