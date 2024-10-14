import pandas as pd

def on_balance_volume(price, volume):
    obv = volume.where(price.diff() > 0, -volume).cumsum()
    return obv
