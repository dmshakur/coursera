import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_plot(df):
    sns.set(style = "white")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype = np.bool))
    f, ax = plt.subplots(figsize = (11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    sns.heatmap(corr, 
            mask = mask, 
            cmap = cmap, 
            vmax = .3, 
            center = 0, 
            square = True, 
            linewidths = .5, 
            cbar_kws = {"shrink": .5})
