'''
coinranking.py
Created on Sep 20, 2021

Get hourly bitcoin data from last 30 days using https://api.coinranking.com/v1/public/coin/1/history/30d

Reading that API, output data in the following schemas:
 
Schema 1:
{
    "date": "{date}",
    "price": "{value}",
    "direction": "{up/down/same}",
    "change": "{amount}",
    "dayOfWeek": "{name}",
    "highSinceStart": "{true/false}",
    "lowSinceStart": "{true/false}"
}
 
- one entry per day at "00:00:00" hours
- results ordered by oldest date first.
- "date" in format "2019-03-17T00:00:00"
- "price" in 2 decimal dollars
- "direction" is direction of price change since previous day in the list, first day can be "na" ({up/down/same})
- "change" is the price difference between current and previous day. "na" for first
- "dayOfWeek" is name of the day (Saturday, Tuesday, etc)
- "highSinceStart" true/false if highest since list start. "na" for first
- "lowSinceStart" true/false if lowest since list start. "na" for first
 
Schema 2:
{
    "date": "{date}",
    "price": "{value}",
    "dailyAverage": "{value}",
    "dailyVariance": "{value}",
    "volatilityAlert:": "{true/false}"
}
- one entry per day
- results ordered by oldest date first.
- "date" in format "2019-03-17T00:00:00"
- "price" in 2 decimal dollars
- "dailyAverage" average of all entries for that day
- "dailyVariance" variance of all entries for that day
- "volatilityAlert:" true/false if any price that day is outside 2 standard deviations.
@author: bowmj
'''
import urllib.request
import json
import pandas
import datetime
import numpy as np


def load_price_history(url: str) -> pandas.DataFrame:
    """Read price history into pandas DataFrame. Format price and timestamp."""
    # Load API results into JSON object
    with urllib.request.urlopen(url) as f:
        try:
            response = json.loads(f.read())
        except:
            raise Exception('Unable to open URL {}'.format(url))

    # Create pandas DataFrame
    try:
        df = pandas.DataFrame(data = response['data']['history'])
    except:
        raise Exception('Unable to parse JSON: {}'.format(response))
    
    # Check columns
    if list(df.columns) != ['price', 'timestamp']:
        raise Exception("Expected columns: ['price', 'timestamp']")

    # Parse timestamps using datetime objects
    df['datetime'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)/1000))
    
    # Add formatted timestamp string
    df['datetime_str'] = df['datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%dT%H:%M:%S'))
    
    # Add formatted date string
    df['date_str'] = df['datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d') + 'T00:00:00')
    
    # Round price to 2 decimal places
    df['price_decimal'] = df['price'].apply(lambda x: round(float(x), 2))
    
    return df


def compute_schema_1(df: pandas.DataFrame) -> pandas.DataFrame:
    """Compute schema 1 output."""
    # Filter to one entry per day at "00:00:00" hours
    criterion = df['datetime'].map(lambda x: x.hour == 0 and x.minute == 0 and x.second == 0)
    df_filtered = df[criterion]
    
    # Build output dataframe
    output = []
    # Add first row
    row = df_filtered.iloc[0]
    output += [{"date": row.date_str,
               "price": str(row.price_decimal),
               "direction": "na",
               "change": "na",
               "dayOfWeek": datetime.datetime.strftime(row.datetime, '%A'),
               "lowSinceStart": "na",
               "highSinceStart": "na"}]
    high = row.price_decimal
    low = row.price_decimal
    last = row.price_decimal
    # Add remaining rows
    for i in range(1, len(df_filtered)):
        row = df_filtered.iloc[i]
        change = round(row.price_decimal - last, 2)
        if change > 0:
            direction = "up"
        elif change < 0:
            direction = "down"
        else:
            direction = "same"
        if row.price_decimal < low:
            low_since_start = "true"
            low = row.price_decimal
        else:
            low_since_start = "false"
        if row.price_decimal > high:
            high_since_start = "true"
            high = row.price_decimal
        else:
            high_since_start = "false"
        output += [{"date": row.date_str,
                    "price": str(row.price_decimal),
                    "direction": direction,
                    "change": str(change),
                    "dayOfWeek": datetime.datetime.strftime(row.datetime, '%A'),
                    "lowSinceStart": low_since_start,
                    "highSinceStart": high_since_start}]
        last = row.price_decimal
        
    return pandas.DataFrame(output)


def compute_schema_2(df: pandas.DataFrame) -> pandas.DataFrame:
    """Compute schema 2 output."""
    # Compute aggregations
    ct = df[['date_str', 'datetime_str']].groupby('date_str').count()
    dt = df[['date_str', 'datetime_str']].groupby('date_str').first()
    price = df[['date_str', 'price_decimal']].groupby('date_str').first()
    mean = df[['date_str', 'price_decimal']].groupby('date_str').mean()
    var = df[['date_str', 'price_decimal']].groupby('date_str').var()
    maxval = df[['date_str', 'price_decimal']].groupby('date_str').max()
    minval = df[['date_str', 'price_decimal']].groupby('date_str').min()
    # Iterate over aggregations and write schema 2.
    # Skip date if there is only one data point.
    # Skip date if the first timestamp is not at 00 hours
    output = []
    for i in range(len(ct)):
        if ct.iloc[i].values[0] > 1:
            avg = mean.iloc[i].values[0]
            std = np.sqrt(var.iloc[i].values[0])
            dailymax = maxval.iloc[i].values[0]
            dailymin = minval.iloc[i].values[0]
            if (dailymax - avg > std * 2) or (dailymin - avg < std * -2):
                alert = "true"
            else:
                alert = "false"
            output += [{"date": dt.iloc[i].values[0],
                        "price": str(price.iloc[i].values[0]),
                        "dailyAverage": str(round(avg, 2)),
                        "dailyVariance": str(round(var.iloc[i].values[0], 2)),
                        "volatilityAlert": alert}]
    return pandas.DataFrame(output)


def write_dataframe(df: pandas.DataFrame, filename: str):
    """Write dataframe to file in JSON format."""
    with open(filename, 'w') as fileout:
        fileout.write('[')
        for i in range(len(df)):
            row = df.iloc[i]
            fileout.write(row.to_json())
            if i < len(df) - 1:
                fileout.write(',')
        fileout.write(']') 


if __name__ == "__main__":
    df = load_price_history("https://api.coinranking.com/v1/public/coin/1/history/30d")
    schema_1 = compute_schema_1(df)
    schema_2 = compute_schema_2(df)
    write_dataframe(schema_1, 'schema_1.json')
    write_dataframe(schema_2, 'schema_2.json')
    schema_1.to_csv('schema_1.csv')
    schema_2.to_csv('schema_2.csv')
