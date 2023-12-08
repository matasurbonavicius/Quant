import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

def plot_(df):
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df = df.dropna(subset=['Size'])

    plt.hist(df['Size'], bins=10, density=True, alpha=0.6, color='g', label='Histogram')
    mu, std = norm.fit(df['Size'])

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fit results: $\mu$ = %.2f, $\sigma$ = %.2f' % (mu, std))
    plt.title('Histogram and Fitted Distribution for "Size"')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


df = pd.read_csv('fakes.csv')
df.set_index('ID', inplace=True)

df_above_average = df[df['Above Average '] == 1]
df_below_average = df[df['Above Average '] == 0]

plot_(df)
plot_(df_above_average)
plot_(df_below_average)
