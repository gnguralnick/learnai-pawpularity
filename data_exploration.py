"""Exploring the data to see if there's any parts of it we should filter out"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')

print(df)
print(df.info())

print(df.cumsum().drop('Id', axis=1).drop('Pawpularity', axis=1))

def plot_attribute(attribute: str) -> None:
    sorted_df = df.groupby('Pawpularity')[attribute].apply(list)
    yvalues = []
    xvalues = range(0, 100)
    for lst in sorted_df:
        yvalues.append(sum(lst) / len(lst))
    plt.scatter(xvalues, yvalues)
    plt.xlabel("Pawpularity")
    plt.ylabel(attribute)


plot_attribute("Human")
