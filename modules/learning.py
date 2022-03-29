import numpy as np
import pandas as pd
import modules.statistics as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from modules.preprocessing import enumerate2
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import TheilSenRegressor


def predict(df_test, model, feats, target):
    """
    Applies a regression model to predict values of a dependent variable for a given dataframe and 
    given features.

    Args:
        df_test: The input dataframe.
        model: The regression model. Instance of Pipeline.
        feats: List of strings: each string is the name of a column of df_test.
        target: The name of the column of df corresponding to the dependent variable.
    Returns:
        y_pred: Array of predicted values. 
    """

    df_x = df_test[feats]
    df_y = df_test[target] #is this needed?
    X = df_x.values
    y_true = df_y.values #is this needed?
    y_pred = model.predict(X)
    return y_pred


def fit_linear_model(df, feats, target, a=1e-4, deg=3, method='ridge', fit_intercept=True, include_bias=True):
    """
    Fits a regression model on a given dataframe, and returns the model, the predicted values and the associated 
    scores. Applies Ridge Regression with polynomial features. 

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        a: A positive float. Regularization strength parameter for the linear least squares function 
        (the loss function) where regularization is given by the l2-norm. 
        deg: The degree of the regression polynomial.

    Returns:    
        pipeline: The regression model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """

    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values
    polynomial_features = PolynomialFeatures(degree=deg, include_bias=include_bias)
    if method == 'ridge':
        model = Ridge(alpha=a, fit_intercept=fit_intercept)
        
    elif method == 'ols':
        model = LinearRegression(fit_intercept=fit_intercept)
    elif method == 'rf':
        model = RFRegressor(n_jobs = -1)
    else:
        print('Unsupported method')
    

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("regression", model)])
    

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe


def get_line_and_slope(values):
    """
    Fits a line on the 2-dimensional graph of a regular time series, defined by a sequence of real values. 

    Args:
        values: A list of real values.

    Returns: 
        line: The list of values as predicted by the linear model.
        slope: Slope of the line.
        intercept: Intercept of the line.   
    """

    ols = LinearRegression()
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    ols.fit(X, y)
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept


def train_on_reference_points(df, w_train, ref_points, feats, target, random_state=0):
    """
    Trains a regression model on a training set defined by segments of a dataframe. 
    These segments are defined by a set of starting points and a parameter indicating their duration. 
    In each segment, one subset of points is randomly chosen as the training set and the remaining points 
    define the validation set.
    
    Args:
        df: Input dataframe. 
        w_train: The duration, given as a number of days, of the segments where the model is trained.
        ref_points: A list containing the starting date of each segment where the model is trained.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        random_state: Seed for a random number generator, which is used in randomly selecting the validation 
        set among the points in a fixed segment.

    Returns:
        model: The regression model. This is an instance of Pipeline.
        training_scores: An array containing scores for the training set. It contains the coefficient 
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
        validation_scores: An array containing scores for the validation set. It contains the coefficient 
        of determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error.
    """

    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    for idx in range(ref_points.size):
        d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
        df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
        df_tmp2 = df_tmp.sample(frac=1, random_state=random_state) # added random state for reproducibility during experiments
        size_train = int(len(df_tmp2) * 0.80)
        df_train = df_train.append(df_tmp2[:size_train])
        df_val = df_val.append(df_tmp2[size_train:])

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train = fit_linear_model(df_train, feats, target)
    y_pred_val = predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val = st.score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores


def predict_on_sliding_windows(df, win_size, step, model, feats, target):
    """
    Given a regression model, predicts values on a sliding window in a dataframe 
    and outputs scores, a list of predictions and a list of windows. 

    Args: 
        df: The input dataframe.
        win_size: The size of the sliding window, as a number of days.
        step: The sliding step.
        model: The regression model. 
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.

    Returns:
        scores: An array of arrays of scores: one array for each window containing the coefficient of 
        determination “R squared”, the mean absolute error, the mean error, the mean absolute percentage error, 
        the mean percentage error.
        preds_test: a list of predictions: one list of predicted values for each window.
        windows: A list of starting/ending dates: one for each window.
    """

    windows = []
    preds_test = []
    scores_list = []
    for i, time in enumerate2(min(df.index), max(df.index), step=step):
        window = pd.to_datetime(time) + pd.Timedelta(days=win_size)
        df_test = df.loc[time:window]
        if df_test.shape[0]>0:
            y_pred = predict(df_test, model, feats, target)
            r_sq, mae, me, mape, mpe = st.score(df_test[target].values, y_pred)
            scores_list.append([r_sq, mae, me, mape, mpe])
            preds_test.append(y_pred)
            windows.append((time, window))
    scores = np.array(scores_list)
    return scores, preds_test, windows


def changepoint_scores(df, feats, target, d1, d2, w_train, w_val, w_test):
    """
    Given as input a dataframe and a reference interval where a changepoint may lie, trains a regression model in
    a window before the reference interval, validates the model in a window before the reference interval and tests 
    the model in a window after the reference interval. 

    Args:
        df: The input dataframe.
        feats: A list of names of columns of df indicating the feature variables.
        target: The name of a column of df indicating the dependent variable.
        d1: The first date in the reference interval.
        d2: The last date in the reference interval.
        w_train: The number of days defining the training set.
        w_val: The number of days defining the validation set.
        w_test: The number of days defining the test set.
    Returns:
        y_pred_train: The array of predicted values in the training set.
        score_train: An array containing scores for the training set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
        y_pred_val: The array of predicted values in the validation set.
        score_val: An array containing scores for the validation set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
        y_pred_test: The array of predicted values in the test set.
        score_test: An array containing scores for the test set: 
        the coefficient of determination “R squared”, the mean absolute error, the mean error, 
        the mean absolute percentage error, the mean percentage error.
    """

    d_train_start = pd.to_datetime(d1) - pd.Timedelta(days=w_train) - pd.Timedelta(days=w_val)
    d_train_stop = pd.to_datetime(d1) - pd.Timedelta(days=w_val)
    d_test_stop = pd.to_datetime(d2) + pd.Timedelta(days=w_test)
    df_train = df.loc[str(d_train_start):str(d_train_stop)]
    df_val = df.loc[str(d_train_stop):str(d1)]
    df_test = df.loc[str(d2):str(d_test_stop)]
    if len(df_train) > 0 and len(df_test) > 0:
        model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train = fit_linear_model(df_train, feats, target)
        y_pred_val = predict(df_val, model, feats, target)
        y_pred_test = predict(df_test, model, feats, target)
        
        r_sq_val, mae_val, me_val, mape_val, mpe_val = st.score(df_val[target].values, y_pred_val)
        r_sq_test, mae_test, me_test, mape_test, mpe_test = st.score(df_test[target].values, y_pred_test)
        score_train = np.array([-r_sq_train, mae_train, me_train, mape_train, mpe_train])
        score_val = np.array([-r_sq_val, mae_val, me_val, mape_val, mpe_val])
        score_test = np.array([-r_sq_test, mae_test, me_test, mape_test, mpe_test])
        return y_pred_train, score_train, y_pred_val, score_val, y_pred_test, score_test
    else:
        raise Exception("Either the training set is empty or the test set is empty")


def fit_pipeline(df, feats, target, pipeline, params):
    """
    Fits a regression pipeline on a given dataframe, and returns the fitted pipline,
    the predicted values and the associated scores.

    Args:
        df: The input dataframe.
        feats: List of names of columns of df. These are the feature variables.
        target: The name of a column of df corresponding to the dependent variable.
        pipeline: A pipeline instance, a scikit-learn object that sequentially applies a list of given 
                  preprocessing steps and fits a selected machine learning model.
        params: A dictionary that contains all the parameters that will be used by `pipeline`.
                The dictionary keys must follow the scikit-learn naming conventions.

    Returns:    
        pipeline: The fitted model. This is an instance of Pipeline.
        y_pred: An array with the predicted values.
        r_sq: The coefficient of determination “R squared”.
        mae: The mean absolute error.
        me: The mean error.
        mape: The mean absolute percentage error.
        mpe: The mean percentage error.
    """
    
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values

    pipeline.set_params(**params)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    r_sq, mae, me, mape, mpe = st.score(y, y_pred)
    return pipeline, y_pred, r_sq, mae, me, mape, mpe


def lasso_selection(df, features, target, alphas=None):
    """
    Utilizes Lasso regression, which penalizes the l1 norm of the weights and indtroduces sparsity in the solution, 
    to find the most `relevant` features, i.e. the ones that have non-zero weights

    Args:
        df: DataFrame that contains the dataset.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        alphas: A list of regularization coefficients to be used, if left as None the alphas are set automatically.

    Returns: 
        selected_features: A list with the name of the columns that were selected by the Lasso method.
    """
    
    X = df[features]
    y = df[target]
    lasso = LassoCV(cv=5, random_state=42, alphas=alphas).fit(X, y)
    model = SelectFromModel(estimator=lasso, prefit=True)
    sup = model.get_support()
    selected_features = list(np.array(features)[sup])
    return selected_features


def perform_grid_search(df, feats, target, scorer, model, params, randomized=False):
    
    """
    Performs a grid-search to find the parameters in the provided search space that yield the best results.
    Used for model tuning.

    Args:
        df: DataFrame that contains the dataset.
        feats: A list of names of columns of df corresponding to the feature variables.
        target: A name of a column of df corresponding to the dependent variable.
        scorer: A performance metric or loss function used in the grid search.
        model: The model that will be tuned.
        params: A dictionary containing all the parameters to be tested.
        randomized: If set to True, a random sample of all the parameter combinations will be tested.

    Returns: 
        selected_params: Dictionary that contains the combination of parameters that resulted in the best score during grid search.
    """
    
    df_x = df[feats]
    df_y = df[target]
    X = df_x.values
    y = df_y.values
    
    model = model
    
    if randomized == True:
        grid_pipeline = RandomizedSearchCV(model, params, n_iter=100, verbose=0, n_jobs=-1, pre_dispatch=64, scoring=scorer, cv=3, random_state=42).fit(X, y)
    else:
        grid_pipeline = GridSearchCV(model, params, verbose=0, n_jobs=-1, pre_dispatch=64, scoring=scorer, cv=3).fit(X, y)
   
    selected_params = {}
    for key in  params:
        selected_params[key] = grid_pipeline.best_params_[key]
    
    return selected_params


def get_ts_line_and_slope(values):
    """
    Fits a line on the 2-dimensional graph of a regular time series, defined by a sequence of real values. 

    Args:
        values: A list of real values.

    Returns: 
        line: The list of values as predicted by the linear model.
        slope: Slope of the line.
        intercept: Intercept of the line.   
    """

    ols = TheilSenRegressor(random_state=0)
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    
    ols.fit(X, y.ravel())
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept  


def calc_changepoints_one_model(df, dates_rain_start, dates_rain_stop, model, target, feats, w1, w2):
    """
    Returns errors associated with changepoint detection in the input segments. Applies the method using one 
    model in all segments (Method 2).
    Args:
        df: Input pandas dataframe
        dates_rain_start: Array of starting points of segments under investigation 
        dates_rain_stop: Array of ending points of segments under investigation 
        model: Regression model 
        target: Name of dependant variable in model
        feats: List of feature variables in model
        w1: Number of days defining the period before each segment, which will be used for calculating the associated score 
        w2: Number of days defining the period after each segment, which will be used for calculating the associated score 
    Returns:
        errors_br: Array containing prediction errors before each segment  
        errors_ar: Array containing prediction errors after each segment 
    """
    errors_br = np.empty((dates_rain_start.size, 6))
    errors_ar = np.empty((dates_rain_start.size, 6))
    for i in range(dates_rain_start.size):
        d1 = dates_rain_start.iloc[i]
        d0 = d1 - pd.Timedelta(days=w1)
        d2 = dates_rain_stop.iloc[i]
        d3 = d2 + pd.Timedelta(days=w2)
        df_ar = df[d2:d3]
        df_br = df[d0:d1]
        try:
            y_pred_ar = predict(df_ar, model, feats, target)
            y_pred_br = predict(df_br, model, feats, target)
            errors_ar[i,:] = st.score(df_ar[target].array, y_pred_ar)
            errors_br[i,:] = st.score(df_br[target].array, y_pred_br)
        except:
            errors_ar[i,:] = [np.nan]*6
            errors_br[i,:] = [np.nan]*6
    return errors_br, errors_ar        


def calc_changepoints_many_models(df, dates_rain_start, dates_rain_stop, target, feats, w1, w2, w3):
    """
    Returns errors associated with changepoint detection in the input segments. Applies the method using one 
    model for each segment (Method 1).
    Args:
        df: Input pandas dataframe
        dates_rain_start: Array of starting points of segments under investigation 
        dates_rain_stop: Array of ending points of segments under investigation 
        target: Name of dependant variable in model
        feats: List of feature variables in model
        w1: Number of days defining the period before each segment, which will be used for training the model
        w2: Number of days defining the period before each segment, which will be used for calculating the associated score 
        w3: Number of days defining the period after each segment, which will be used for calculating the associated score 
    Returns:
        errors_br: Array containing prediction errors before each segment  
        errors_ar: Array containing prediction errors after each segment 
    """
    errors_br = np.empty((dates_rain_start.size, 6))
    errors_ar = np.empty((dates_rain_start.size, 6))
    for i in range(dates_rain_start.size):
        d1 = dates_rain_start.iloc[i]
        d2 = dates_rain_stop.iloc[i]
        try:
            y_pred_train, score_train, y_pred_val, errors_br[i,:], y_pred_test, errors_ar[i,:] = changepoint_scores(df, feats, target, d1, d2, w1, w2, w3)
        except:
            errors_ar[i,:] = [np.nan]*6
            errors_br[i,:] = [np.nan]*6
    return errors_br, errors_ar 
