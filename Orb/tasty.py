import os
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Candle

async def fetch_historical_csv(ticker, interval, start_date):
    load_dotenv("auth.env")
    secret = os.getenv("SECRET")
    token = os.getenv("REFRESH_TOKEN")
    session = Session(secret, token)
    filename = f"results/{ticker}.csv"
    
    # Connect to the streamer
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe_candle([ticker], interval=interval, start_time=start_date, extended_trading_hours=True)
        candles = []
        
        try:
            while True:
                # Wait for the next candle
                candle = await asyncio.wait_for(streamer.get_event(Candle), timeout=3.0)
                candles.append({
                    "Timestamp": datetime.fromtimestamp(candle.time / 1000), # Convert ms to datetime
                    "Open": candle.open,
                    "High": candle.high,
                    "Low": candle.low,
                    "Close": candle.close,
                    "Volume": candle.volume
                })

                if candle.time / 1000 >= (datetime.now().timestamp() - 60):
                    break
        except asyncio.TimeoutError:
            print("All data collected...")
        
        # Process and save to CSV
        if len(candles) != 0:
            df = pd.DataFrame(candles)
            df.to_csv(filename, index=False)
            print(f"Successfully saved {filename}")
        else:
            print("No candles found.")