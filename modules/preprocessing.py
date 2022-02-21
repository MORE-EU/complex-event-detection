import pandas as pd
import numpy as np
from matrixprofile import core
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize as norm

def enumerate2(start, end, step=1):
    """ 
    Args:
        start: starting point
        end: ending point    .
        step: step of the process                   
        
        
    Return: 
        The interpolated DataFrame/TimeSeries
     """
    i=0
    while start < pd.to_datetime(end):
        yield (i, start)
        start = pd.to_datetime(start) + pd.Timedelta(days=step)
        i += 1

def change_granularity(df,granularity='30s',size=10**7,chunk=True): 
    """ 
    Changing the offset of a TimeSeries. 
    We do this procedure by using chunk_interpolate. 
    We divide our TimeSeries into pieces in order to interpolate them.
        
    Args:
        df: Date/Time DataFrame. 
        size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.       .
        granularity: The offset user wants to resample the Time Series                  
        chunk: If set True, It applies the chunk_interpolation
    
    Return: 
        The interpolated DataFrame/TimeSeries
     """

    df = df.resample(granularity).mean()
    print('Resample Complete')
    if chunk==True: #Getting rid of NaN occurances.
        df=chunk_interpolate(df,size=size,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1)
        print('Interpolate Complete')
    return df
  
def filter_col(df, col, less_than=None, bigger_than=None): 
    """ 
    Remove rows of the dataframe that they are under, over/both from a specific/two different input price/prices.
        
    Args:
        df: Date/Time DataFrame. 
        col: The desired column to work on our DataFrame. 
        less_than: Filtering the column dropping values below that price.
        bigger_than: Filtering the column dropping values above that price.
    
    Return: 
        The Filtrated TimeSeries/DataFrame
    """
    if(less_than is not None):
        df=df.drop(df[df.iloc[:,col] < less_than].index)
    if(bigger_than is not None):
        df=df.drop(df[df.iloc[:,col] > bigger_than].index)
    print('Filter Complete')
    return df


def filter_dates(df, start, end):
    """ 
    Remove rows of the dataframe that are not in the [start, end] interval.
    
    Args:
        df:DataFrame that has a datetime index.
        start: Date that signifies the start of the interval.
        end: Date that signifies the end of the interval.
   
   Returns:
        The Filtrared TimeSeries/DataFrame
    """
    date_range = (df.index >= start) & (df.index <= end)
    df = df[date_range]
    return df

def normalize(df):
    """ 
    This function transforms an input dataframe by rescaling values to the range [0,1]. 
    
    Args:
        df: Date/Time DataFrame or any DataFrame given with a specific column to Normalize. 
   
    Return:
        Normalized Array
    """
    values=[]
        # prepare data for normalization
    values = df.values
    values = values.reshape((len(values), 1))
        # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    return normalized


def add_noise_to_series(series, noise_max=0.00009):
    
    """ 
    Add uniform noise to series.
    
    Args:
        series: The time series to be added noise.
        noise_max: The upper limit of the amount of noise that can be added to a time series point
    
    Return: 
        DataFrame with noise
    """
    
    if not core.is_array_like(series):
        raise ValueError('series is not array like!')

    temp = np.copy(core.to_np_array(series))
    noise = np.random.uniform(0, noise_max, size=len(temp))
    temp = temp + noise

    return temp


def add_noise_to_series_md(df, noise_max=0.00009):
    
    """ 
    Add uniform noise to a multidimensional time series that is given as a pandas DataFrame.
    
    Args:
        df: The DataFrame that contains the multidimensional time series.
        noise_max: The upper limit of the amount of noise that can be added to a time series point.
   
    Return:
        The DataFrame with noise to all the columns
    """
    
    for col in df.columns:
        df[col] = add_noise_to_series(df[col].values, noise_max)
    return df


def filter_df(df, filter_dict):
    """ 
    Creates a filtered DataFrame with multiple columns.
        
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        filter_dict: A dictionary of columns user wants to filter
    
    Return: 
        Filtered DataFrame
    """

    mask = np.ones(df.shape[0]).astype(bool)
    for name, item in filter_dict.items():
        val, perc = item
        if val is not None:
            mask = mask & (np.abs(df[name].values - val) < val * perc)
            
    df.loc[~mask, df.columns != df.index] = np.NaN
    f_df = df
    print(f_df.shape)
    return f_df

def chunker(seq, size):
    """
    Dividing a file/DataFrame etc into pieces for better hadling of RAM. 
    
    Args:
        seq: Sequence, Folder, Date/Time DataFrame or any Given DataFrame.
        size: The size/chunks we want to divide our Seq/Folder/DataFrame.
    
    Return:
        The divided groups
        
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def chunk_interpolate(df,size=10**6,interpolate=True, method="linear", axis=0,limit_direction="both", limit=1):

    """
    After Chunker makes the pieces according to index, we Interpolate them with *args* of pandas.interpolate() and then we Merge them back together.
    This step is crucial for the complete data interpolation without RAM problems especially in large DataSets.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        size: The size/chunks we want to divide our /DataFrame according to the global index of the set. The Default price is 10 million.
    
    Return:
        The Interpolated DataFrame
    """
    
    group=[]
    for g in chunker(df,size):
        group.append(g)
    print('Groupping Complete')
    for i in range(len(group)):
            group[i].interpolate(method=method,axis=axis,limit_direction = limit_direction, limit = limit, inplace=True)
            df_int=pd.concat(group[i] for i in range(len(group)))
            df_int=pd.concat(group[i] for i in range(len(group)))
    print('Chunk Interpolate Done')
    return df_int


def is_stable(*args, epsilon):
    """
    Args:
        epsilon: A small value in order to avoid dividing with Zero.
    
    Return: 
        A boolean vector from the division of variance with mean of a column.
    """
    #implemented using the index of dispersion (or Fano factor)
    dis = np.var(np.array(args),axis = 1)/np.mean(np.array(args),axis = 1)
    return np.all(np.logical_or((dis < epsilon),np.isnan(dis)))


def filter_dispersed(df, window, eps):
    """
    We are looking at windows of consecutive row and calculate the mean and variance. For each window if the index of disperse or given column is in the given threshhold
    then the last row will remain in the data frame.
    
    Args:
        df: Date/Time DataFrame or any Given DataFrame.
        window: A small value in order to avoid dividing with Zero.
        eps: A small value in order to avoid dividing with Zero (See is_stable)
    
    Return: The Filtered DataFrame
    """
    df_tmp = df[rolling_apply(is_stable, window, *df.transpose().values, epsilon= eps)]
    return df_tmp[window:]
  
def scale_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    min_max_scaler = MinMaxScaler()
    df[df.columns] = min_max_scaler.fit_transform(df)
    return df

def standardize_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    standard_scaler = StandardScaler()
    df[df.columns] = standard_scaler.fit_transform(df)
    return df

def unit_norm_df(df):
    """ 
    Scale each column of a dataframe to the [0, 1] range performing the min max scaling
    
    Args:
        df: The DataFrame to be scaled.
    
    Return: Scaled DataFrame
    """
    df[df.columns] = norm(df, axis=0)
    return df