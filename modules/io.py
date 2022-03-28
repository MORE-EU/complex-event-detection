import pandas as pd
import numpy as np
import h5py
import os
import pathlib
import warnings
import gc
import matplotlib as plt


def load_df(path): 
    """ 
    Loading a parquet file to a pandas DataFrame. Return this pandas DataFrame.

    Args:
        path: Path of the under loading DataFrame.
    Return: 
        pandas DataFrame.
    """

    df = pd.DataFrame()
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df.set_index(df.index, inplace=True)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df.index)
        df.set_index(df.index, inplace=True)
    return df
  
    
def load_mp(path):
    """
    Load the Univariate/Multivariate Matrix profile which was saved from Create_mp in a .npz file.

    Args:
      path: Path of the directory where the file is saved.

    Return:
        mp: Matrixprofile Distances
        mpi: Matrixprofile Indices
    """
    mp={}
    mpi={}
    loaded = np.load(path + ".npz", allow_pickle=True)
    mp = loaded['mp']
    mpi = loaded['mpi']
    return mp, mpi

  
def save_mdmp_as_h5(dir_path, name, mps, idx, k=0):

    """
    Save a multidimensional matrix profile as a pair of hdf5 files. Input is based on the output of (https://stumpy.readthedocs.io/en/latest/api.html#mstump).

    Args:
       dir_path: Path of the directory where the file will be saved.
       name: Name that will be appended to the file after a default prefix. (i.e. mp_multivariate_<name>.h5)
       mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                   (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
       idx: The multi-dimensional matrix profile index where each row of the array corresponds to each matrix profile index for a given dimension.
       k: If mps and idx are one-dimensional k can be used to specify the given dimension of the matrix profile. The default value specifies the 1-D matrix profile.
                 If mps and idx are multi-dimensional, k is ignored.

    Return:

    """
    if mps.ndim != idx.ndim:
        err = 'Dimensions of mps and idx should match'
        raise ValueError(f"{err}")
    if mps.ndim == 1:
        mps = mps[None, :]
        idx = idx[None, :]
        h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','w')
        h5f.create_dataset(f'mp{k}', data=mps[0])
        h5f.close()

        h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','w')
        h5f.create_dataset(f'idx{k}', data=idx[0])
        h5f.close()
        return

    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','w')
    for i in range(mps.shape[0]):
        h5f.create_dataset(f'mp{i}', data=mps[i])
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','w')
    for i in range(mps.shape[0]):
        h5f.create_dataset(f'idx{i}', data=idx[i])
    h5f.close()
    return

  
def load_mdmp_from_h5(dir_path, name, k):
   
    """
    Load a multidimensional matrix profile that has been saved as a pair of hdf5 files.
    
    Args:
      dir_path: Path of the directory where the file is located.
     name: Name that follows the default prefix. (i.e. mp_multivariate_<name>.h5)
      k: Specifies which K-dimensional matrix profile to load. 
                 (i.e. k=2 loads the 2-D matrix profile
    
    Return:
        mp: matrixprofile/stumpy distances
        index: matrixprofile/stumpy indexes
            
          
        
    """
    # Load MP from disk
    
    h5f = h5py.File(dir_path + 'mp_multivariate_' + name + '.h5','r')
    mp= h5f[f'mp{k}'][:]
    h5f.close()

    h5f = h5py.File(dir_path + 'index_multivariate_' + name + '.h5','r')
    index = h5f[f'idx{k}'][:]
    h5f.close()
    return mp, index


def save_results(results_dir, sub_dir_name, p, df_stats, m, radius, ez, k, max_neighbors):
    """ 
    Save the results of a specific run in the directory specified by the results_dir and sub_dir_name.
    The results contain some figures that are created with an adaptation of the matrix profile foundation visualize() function.
    The adaptation works for multi dimensional timeseries and can be found at 
    (https://github.com/MORE-EU/matrixprofile/blob/master/matrixprofile/visualize.py) as visualize_md()

    Args:
        results: Path of the directory where the results will be saved.
        sub_directory: Path of the sub directory where the results will be saved.
        p: A profile object as it is defined in the matrixprofile foundation python library.
        df_stats: DataFrame with the desired statistics that need to be saved.
        m: The subsequence window size.
        ez: The exclusion zone to use.
        radius: The radius to use.
        k: The number of the top motifs that were calculated.
        max_neighbors: The maximum amount of neighbors to find for each of the top k motifs.

    Return:
        None

    """




    path = os.path.join(results_dir, sub_dir_name)

    print(path)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        figs = visualize_md(p)

        for i,f in enumerate(figs):
            f.savefig(path + f'/fig{i}.png' , facecolor='white', transparent=False, bbox_inches="tight")
            f.clf()

    # remove figures from memory
    plt.close('all')
    gc.collect() 

    df_stats.to_csv(path + '/stats.csv')

    lines = [f'Window size (m): {m}',
             f'Radius: {radius} (radius * min_dist)',
             f'Exclusion zone: {ez} * window_size',
             f'Top k motifs: {k}',
             f'Max neighbors: {max_neighbors}']

    with open(path+'/info.txt', 'w') as f:
        for ln in lines:
            f.write(ln + '\n')
            
            
def output_changepoints(scores, indices, dates_rain_start, dates_rain_stop, errors_br, errors_ar, error_name_br, error_name_ar, precip):
    """
    Given the output of one of the functions calc_changepoints_one_model, calc_changepoints_many_models, 
    returns a dataframe containing all relevant information about changepoints.  
    Args:
        scores: An array of scores associated to each input segmnt 
        indices: Indices of the input segments (pointing to the input list)
        dates_rain_start: Array of starting points of segments under investigation 
        dates_rain_stop: Array of ending points of segments under investigation 
        errors_br: Array of errors corresponding to periods preceding the segments under investigation
        errors_ar: Array of errors corresponding to periods following the segments under investigation
        error_name_br: Name of the error used for the period preceding segments
        error_name_ar: Name of the error used for the period following segments
        precip: Array of precipitation values
    Returns:
        A pandas dataframe containing relevant information about the detected changepoints/segments. This includes scores and
        precipitation values for the input segments. 
    """
    start_dates = []
    end_dates = []
    all_prec = []
    all_errors_br = []
    all_errors_ar = []
    all_scores = []
    prec1 = []
    prec2 = []
    types = []
    ids = []
    for i in indices:
        d1 = dates_rain_start.iloc[i]
        d2 = dates_rain_stop.iloc[i]
        all_errors_br.append(errors_br[i])
        all_errors_ar.append(errors_ar[i])
        precip2 = precip.loc[d1:d2][1:-1].values
        start_dates.append(d1)
        end_dates.append(d2)
        if len(precip2)>0:
            prec1.append(precip2.max())
            prec2.append(precip2.mean())
        else:
            prec1.append(0)
            prec2.append(0)    
        ids.append(i)    
        all_scores.append(scores[i])
    return pd.DataFrame.from_dict({"Score": all_scores, "id": ids, "Starting date": start_dates, "Ending date": end_dates, "Max precipitation": np.round(prec1,2),  "Mean precipitation": np.round(prec2,2), error_name_br+" before rain (true-pred)": all_errors_br,  error_name_ar+" after rain (true-pred)": all_errors_ar})    