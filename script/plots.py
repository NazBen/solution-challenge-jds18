import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_prediction(y_pred, y_true=None, dates=None, ax=None, figsize=(8, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if dates is None:
        dates = np.linspace(y_pred)
        ax.set_xlabel('Values')
    else:
        ax.set_xlabel('Dates')
    ax.plot(dates, y_pred, label='pred')
    if y_true is not None:
        ax.plot(dates, y_true, label='true')

    ax.legend(loc=0)
    ax.set_ylabel('y')


def plot_evolution_test(plot_features, df1, df2):
    n_feat = len(plot_features)
    fig, axes = plt.subplots(
        n_feat, 1, figsize=(10, 2 + 2*n_feat), sharex=True)
    for i, feat in enumerate(plot_features):
        ax = axes if n_feat == 1 else axes[i]
        ax.plot(df1['date'], df1[feat], '.-', label='train')
        ax.plot(df2['date'], df2[feat], '.-', label='test')
        ax.set_ylabel(feat)
    ax.legend(loc=0)
    ax.set_xlabel('Date')
    fig.tight_layout()


def plot_importance(importance, features):
    features = np.asarray(features)
    idx = importance.argsort()
    pd.DataFrame(importance[idx], index=features[idx]).plot(
        kind='barh', figsize=(4, int(len(features)/3)))


def plot_correlation(df, method='pearson', ratio=(0.7, 0.5)):
    n_features = df.shape[1]
    fig, ax = plt.subplots(figsize=(ratio[0]*n_features, ratio[1]*n_features))
    corr = df.corr(method=method)
    sns.heatmap(corr, ax=ax, vmin=-1., vmax=1., square=True,
                robust=True, annot=True, fmt='.2f')
    ax.set_title('Correlation map')
    fig.tight_layout()
