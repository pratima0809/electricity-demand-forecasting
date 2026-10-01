import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    return df
