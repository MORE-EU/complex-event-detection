import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from dtw import dtw

def plot_knee(mps, save_plot=False, filename='knee.png'):
    
    """ 
    Plot the minimum value of the matrix profile for each dimension. This plot is used to visually look for a 'knee' or 'elbow' that
    can be used to find the optimal number of dimensions to use.
    
    Args:
        mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
        save_plot: If save_plot is True then the figure will be saved. Otherwise it will just be shown.
        filename: Used if save_plot=True, the name of the file to be saved.
    
    Return:
        Figure
    """
    
    motifs_idx = np.argsort(mps, axis=1)[:, :2]
    mp_len = mps.shape[0]
    plt.figure(figsize=(15, 5), dpi=80)
    plt.xlabel('k (Number of dimensions, zero-indexed)', fontsize='20')
    plt.ylabel('Matrix Profile Min Value', fontsize='20')
    plt.xticks(range(mp_len))
    plt.plot(mps[range(mp_len), motifs_idx[:mp_len, 0]], c='red', linewidth='4');
    if save_plot:
        plt.savefig(filename)
    else:
        plt.show()
    return

  
def predefined_labels(axs, df, fixed_dates):
    """
    Given a df and a list of dates we find the index of those dates in a dataset
    
    Args:
        axs: axes
        df: DateTime DataFrame
        fixed_dates: a list of dates (sorted)
    Return:
        ids: indexes of the dates 
    
    """

    ids = []
    for d in fixed_dates:
        axs.axvline(x=df.index.get_loc(d), ls="dashed", color='darkgreen')
        ids.append(df.index.get_loc(d))
    return ids


def convert_labels_to_dt(labels, df): 
    """
    
    Args:
    
    Returns:
    """
    l = []
    for label in labels:
        if(len(df) >= round(label)):
            l.append(df.index[round(label) ].strftime("%Y-%m-%d"))
        else:
            l.append(None)
            #l = [ df.index[round(label)].strftime("%Y-%m-%d") for label in labels]
    return l


def get_fixed_dates(df,fixed_dates):
    """
    For sorted data we get the array of indexes of those dates
    
    Args:
        df: DateTime dataframe
        fixed_dates: a list of dates (sorted)
         
    Return:
        array of indexes of the dates 
    
    """
    dates = []
    for date_point in fixed_dates:
        date_point = datetime.strptime(date_point, '%Y-%m-%d %H:%M:%S')
        start_date_d = df.index[0]
        end_date_d = df.index[-1]
        if (date_point >= start_date_d) and (date_point <= end_date_d):
            dates.append(df.index.get_loc(date_point))
    return np.array(dates)

  
def plot_profile(df, mp, col_name): 
    """
    Plot the 'Column_name' graph and the corresponding MatrixProfile. We denote with the black arrow to top motif in our set
    
    Args:
        df : A pandas DataFrame/Time Series
        mp : matrix profile distances
        col_name: name of a column
    
    Returns:
        list : figures
        A list of matplotlib figures.
    """
    motifs_idx = np.argsort(mp)[:2]
    mp_len = mp.shape[0]
    with sns.axes_style('ticks'):
        fig1, axs1 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))
        df.iloc['col_name'].plot(ax=axs1, figsize=(20, 6))

        fig2, axs2 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))


        axs2.set_ylabel(f'MP-', fontsize='20')
        axs2.plot(mp, c='orange')
        axs2.set_xlabel('Time', fontsize ='20')

        axs2.plot(motifs_idx, mp[motifs_idx] + 1, marker="v", markersize=10, color='black')
        axs2.plot(motifs_idx, mp[motifs_idx] + 1, marker="v", markersize=10, color='black')
        
        
def plot_profile_md(df,column_name,mp): 
    """
    Plot the 'Column_name' graph and the corresponding Profile. We denote with the black arrow to top motif in our set.
    
    Args:
        df : A pandas DataFrame/Time Series
        mp : matrix profile distances
        column_name: name of the column
        
    Returns:
        list : figures
        A list of matplotlib figures.
    """
    
    motifs_idx = np.argsort(mp, axis=1)[:, :2]
    with sns.axes_style('ticks'):
        fig1, axs1 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 12))
        df['col_name'].plot(ax=axs1, figsize=(20, 2))

        fig2, axs2 = plt.subplots(mp.shape[0], sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 12))
        for k, dim_name in enumerate(cols):
            if np.all(np.isinf(mp[k])): continue

            axs2[k].set_ylabel(f'MP-{dim_name}', fontsize='10')
            axs2[k].plot(mp[k], c='orange')
            axs2[k].set_xlabel('Time', fontsize ='20')

        axs2[k].plot(motifs_idx[k, 0], mp[k, motifs_idx[k, 0]] + 1, marker="v", markersize=10, color='black')
        axs2[k].plot(motifs_idx[k, 1], mp[k, motifs_idx[k, 1]] + 1, marker="v", markersize=10, color='black')
        
        
def plot_segmentation(df, path, output, fixed_dates, file_name, top_seg=3):
    """
    Plotting the ChangePoints/ Regimes that we precomputed from change_points. 
    The result would be multiple graphs up to the number of L ( subsquence list).  
    Later we use the dtw in order to find a generalized distance between a list of dates and the
    precomputed change points/regimes.
    In the end we save the ones that have the minimum distance up to a number the user wishes.
    
    Args:
        df: DateTime DataFrame
        path: path to be saved the figures
        output: output from change_points
        top_seg: number of the best plots we want to save
        fixed_dates: list of sorted dates
    
    Return:
        List of figures, we are saving up to top_seg
    """

    
    best = []
    
    diff1=[]
    dloc=[]
    regi=[]
    model=[]
    lam=[]
    with sns.axes_style('ticks'): 
        figs = []
        
        for l, v in output.items():
            lam.append(l)
            dloc.append(convert_labels_to_dt(v[0][1],df))
            
            sz = len(df)
            threshold = sz / 1000
            ids = get_fixed_dates(df,fixed_dates)
            fig, axs = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0.1}, figsize=(20, 3))
            diff = 0
        
            for idx, (cac, regime_locations) in enumerate(v):
                axs.plot(np.arange(0, cac.shape[0]), cac, color='C1')
                filtered_regimes = regime_locations[(regime_locations > ids[0] - threshold) & (regime_locations < ids[- 1] + threshold)]
                if len(filtered_regimes) == 0:
                    filtered_regimes = regime_locations
                manhattan_distance = lambda x, y: np.abs(x - y)
                diff = dtw( regime_locations, ids, dist=manhattan_distance)[0]
                axs.set_ylabel(f'L:{str(l)}', fontsize=18)
                axs.set_title(f'{file_name}-{diff}')
                for regime in regime_locations:
                    
                    axs.axvline(x=regime, linestyle=":")
                plt.minorticks_on()
                diff1.append(diff)
                regi.append(regime_locations)
              
            ids = predefined_labels(axs, df,fixed_dates)
            labels = [item for item in axs.get_xticks()]
            visible = convert_labels_to_dt(labels[-1:1], df)
            locs, _ = plt.xticks()
            plt.xticks(ticks=locs[1:-1], rotation=30)
            name = f'{path}segmentation-{str(l)}-'
            config = {"L": l, "regime": regime, 'diff': diff}
            figs.append([fig, diff, name, config])
        sorted_figs = sorted(figs, key=lambda tup: tup[1])
       
        if(len(sorted_figs) < top_seg):
            top_seg = len(sorted_figs)
        for i in range(0, top_seg, 1):
            fig, diff, name, config = sorted_figs[i]
            best.append(config)
            fig.savefig(name + "_" + str(i))
    model = pd.DataFrame.from_dict({"L": lam,'Changepoints indexes': regi, "Changepoints Dates": dloc, "Normalized Distance": (diff1-np.min(diff1))/(np.max(diff1) - np.min(diff1))})
    return best, model
