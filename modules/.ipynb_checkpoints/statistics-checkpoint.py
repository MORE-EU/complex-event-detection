import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics

def mape1(y_true, y_pred):

    """
    Computes the Mean Absolute Percentage Error between the 2 given time series

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.
    
    Return: 
        Mean Absolute Percentage Error value.
            
        
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    mape1 = (np.mean(np.abs(y_true-y_pred)/np.mean(y_true)))
    
    return mape1


def mpe1(y_true, y_pred):

    """
    Computes the Mean Percentage Error between the 2 given time series.

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.
    
    Return: 
        Mean Absolute Error value.
            
        
    """
    mpe1 = (np.mean(y_true-y_pred)/np.mean(y_true))
    return mpe1


def score(y_true, y_pred):

    """
    Computes a set of values that measure how well a predicted time series matches the actual time series.

    Args:
        y_true: A numpy array that contains the actual values of the time series.
        y_pred: A numpy array that contains the predicted values of the time series.
    
    Return: 
        Returns a value for each of the following measures: 
        r-squared, mean absolute error, mean error, mean absolute percentage error, mean percentage error
        
            
        
    """
    r_sq = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    me = np.mean(y_true-y_pred)
    mape = mape1(y_true, y_pred)
    mpe = mpe1(y_true, y_pred)
    return r_sq, mae, me, mape, mpe


def get_top_deviations(scores, metric='mpe', n=5):

    """
    Given a matrix that each row contains scores of how well a segment fits a model, find the indices of the top most deviant segments.
    
    Args:
        scores: A 2-D numpy array (NxM) that contains M scores for each one of the N segments.
        metric: A string that specifies which score to consider.
    
    Return: 
        The indices of the segments.
            
        
    """
    metrics = {'mpe': 4, 'me': 2, 'mpe_slope':5}
    score_column = metrics[metric]
    indices = np.argsort(scores[:, score_column])[:n]
    return indices


def multi_corr(df, dep_column):

    """
    Computation of the coefficient of multiple correlation.
    The input consists of a dataframe and the column corresponding to the dependent variable.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        dep_column: The corresponding the column to the dependent variable.
    
    Return: 
        The coefficient of multiple correlation between the dependant column and the rest.
            
        
    """
    df_str_corr = df.corr(method='pearson')
    df_str_corr_ind_temp = df_str_corr.drop(index=dep_column)
    df_str_corr_ind = df_str_corr_ind_temp.drop(columns=dep_column)
    df_str_corr_ind_inv = inv(df_str_corr_ind.values)
    df_str_corr_dep = df_str_corr_ind_temp.loc[:, dep_column]
    return np.matmul(np.matmul(np.transpose(df_str_corr_dep.values), df_str_corr_ind_inv), df_str_corr_dep.values)
