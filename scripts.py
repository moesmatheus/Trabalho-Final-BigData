import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tools import add_constant


def print_results(func):
    
    def wraper(*args, **kwargs):

        r = func(*args, **kwargs)
        
        print('Results')
        
        for i in r:
            print(f'{i}: {r[i]:.3f}')
            
        return r
        
    return wraper

def plot_results(func):
    
    def wrapper(*args, **kwargs):
        
        r = func(*args, **kwargs)
        
        ax = sns.scatterplot(x = kwargs['prediction'], y = kwargs['actual'])
        
        mn = min(kwargs['prediction'].min(), kwargs['actual'].min())
        mx = max(kwargs['prediction'].max(), kwargs['actual'].max())
        points = np.linspace(mn, mx, 100)
        
        ax.plot(points, points, color='grey', marker=None,
            linestyle='--', linewidth=1.0)
        
        ax.set_xlabel('Previsões')
        
        ax.set_ylabel('Valor Real')
        
        plt.show()
        
        return r
        
    return wrapper


@plot_results
@print_results
def get_metrics(prediction, actual):
    metrics = {}

    metrics['MSE'] = mean_squared_error(prediction, actual)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(prediction, actual)
    metrics['R2'] = r2_score(y_true = actual, y_pred = prediction)

    return metrics

def out_of_fold_pred(x, y, model, folds = 10):
    
    pred = np.zeros(x.shape[0])

    kfold = KFold(n_splits = 10)
    
    for train, test in kfold.split(x):
        
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        
        model.fit(x_train, y_train)
        
        pred[test] = model.predict(x_test)
        
    return pred


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def corr(x_y, figsize = (11,9)):
    # Compute the correlation matrix
    corr = x_y.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize = figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)
    
def get_aic(x, y, labels):
    
    aic = []
    
    for i in labels:

        lm = sm.OLS(endog = y, exog = add_constant(x.drop(i, axis = 1)))

        aic.append(lm.fit().aic)
        
    return aic

def iterate_aic(x, y):
    
    x = x.copy()
    
    remove = []
    
    aic_base = sm.OLS(endog = y, exog = add_constant(x)).fit().aic
    
    aic = get_aic(x, y, x.columns)
    
    while max(aic) > aic_base:
        
#         print(max(aic), aic_base)
        
        r = x.columns[np.argmax(aic)]

        remove.append(r)

        x = x.drop(r, axis = 1)
        
        aic_base = sm.OLS(endog = y, exog = add_constant(x)).fit().aic
        
        aic = get_aic(x, y, x.columns)
        
        
    return remove, aic, x.columns