import sys
import tasty
import model
from datetime import datetime, timedelta
import asyncio

if __name__ == '__main__':
    ticker = "NVDA"
    interval = "5m"
    period = 90 # In days. Max allowed is 90 days
    steps = 3 # Number of prediction steps - based on the interval
    plot_data = True

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for i in range(0, len(args)):
            match args[i]:
                case "-t": # Ticker option
                    ticker = args[i + 1].upper()
                    i += 1
                case "-s": # Steps option
                    steps = int(args[i + 1])
                    i += 1
                case "-p": # To plot option
                    plot_data = True
                    i += 1

    start_time = datetime.now() - timedelta(days=period)
    asyncio.run(tasty.fetch_historical_csv(ticker, interval, start_time))
    model.new(ticker, interval, period, steps, plot_data)