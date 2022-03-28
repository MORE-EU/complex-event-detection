import numpy as np
import pandas as pd
import matrixprofile
import matrixprofile as mpf
from stumpy import stump, fluss, gpu_stump, mstumped, mstump, subspace
from stumpy import stump, motifs, mass, match, core


def pick_subspace_columns(df, mps, idx, k, m, include):
    
    """ 
    Given a multi-dimensional time series as a pandas Dataframe, keep only the columns that have been used for the creation of the k-dimensional matrix profile.
    
    Args:
        df: The DataFrame that contains the multidimensional time series.
        mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
        idx: The multi-dimensional matrix profile index where each row of the array corresponds to each matrix profile index for a given dimension.
        k: If mps and idx are one-dimensional k can be used to specify the given dimension of the matrix profile. The default value specifies the 1-D matrix profile.
              If mps and idx are multi-dimensional, k is ignored.
        m: The subsequence window size. Should be the same as the one used to create the multidimensional matrix profile that is the input.
        include: A list of the column names that must be included in the constrained multidimensional motif search.
    
    Return:
        The list of subspace columns
    """
    
    motifs_idx = np.argsort(mps, axis=1)[:, :2]
    col_indexes = []
    for n in include:
        col_indexes.append(df.columns.get_loc(n))
    
    print(f'Include dimensions: {include}, indexes in df = {col_indexes}')
    S = subspace(df, m, motifs_idx[k][0], idx[k][motifs_idx[k][0]], k, include=col_indexes)
    print(f"For k = {k}, the {k + 1}-dimensional subspace includes subsequences from {df.columns[S].values}")
    subspace_cols = list(df.columns[S].values)
    return subspace_cols

  
def to_mpf(mp, index, window, ts):
    """ 
    Using a matrix profile, a matrix profile index, the window size and the timeseries used to calculate the previous, create a matrix profile object that
    is compatible with the matrix profile foundation library (https://github.com/matrix-profile-foundation/matrixprofile). This is useful for cases where another               library was used to generate the matrix profile.
    
    Args:
        mp: A matrix profile.
        index: The matrix profile index that accompanies the matrix profile.
        window: The subsequence window size.
        ts: The timeseries that was used to calculate the matrix profile.
    
    Return: 
        The Matrixprofile structure
    """
    
    
    mp_mpf = matrixprofile.utils.empty_mp()
    mp_mpf['mp'] = np.array(mp)
    mp_mpf['pi'] = np.array(index)
    mp_mpf['metric'] = 'euclidean'
    mp_mpf['w'] = window
    mp_mpf['ez'] = 0
    mp_mpf['join'] = False
    mp_mpf['sample_pct'] = 1
    mp_mpf['data']['ts'] = np.array(ts).astype('d')
    mp_mpf['algorithm']='mpx'
    
    return mp_mpf

  
def compute_mp_av(mp, index, m, df, k):
    
    """ 
    Given a matrix profile, a matrix profile index, the window size and the DataFrame that contains the timeseries.
    Create a matrix profile object and add the corrected matrix profile after applying the complexity av.
    Uses an extended version of the apply_av function from matrixprofile foundation that is compatible with multi-dimensional timeseries.
    The implementation can be found here (https://github.com/MORE-EU/matrixprofile/blob/master/matrixprofile/transform.py)
    
    Args:
        mp: A matrix profile.
        index: The matrix profile index that accompanies the matrix profile.
        window: The subsequence window size.
        ts: The timeseries that was used to calculate the matrix profile.
    
    Return:
        Updated profile with an annotation vector
    """
    
    # Apply the annotation vector
    m  = m # window size
    mp = np.nan_to_num(mp, np.nanmax(mp)) # remove nan values
    profile = to_mpf(mp, index, m, df)
    av_type = 'complexity'
    profile = mpf.transform.apply_av(profile, av_type)
    
    return profile


def pattern_loc(start, end, mask, segment_labels):
    
    """ 
    Considering that a time series is characterized by regions belonging to two different labels.
    
    
    Args:
        start: The starting index of the pattern.
        end: The ending index of the pattern. 
        mask: Binary mask used to annotate the time series.
        segment_labels: List of the two labels that characterize the time series.
    
    Return: The label name of the region that the pattern is contained in.
    """
    
    if len(segment_labels) != 2:
        raise ValueError('segment_labels must contain exactly 2 labels')
    
    start = start
    end = end
    
    # the first label in the list will be assigned to for the True regions in the mask
    true_label = segment_labels[0]
    
    # the second label in the list will be assigned to for the False regions in the mask
    false_label = segment_labels[1]
    
    if mask[start] == mask[end]:
        if mask[start] == True:
            loc = true_label
        else:
            loc = false_label
    else:
        # if a pattern spans both regions return the label 'both'
        loc = 'both'
        
    return loc

  
def calc_cost(cl1_len, cl2_len, num_cl1, num_cl2):
    
    """ 
    Assign a cost to a pattern based on if the majority of its occurances are observed
    in regions of a time series that are annotated with the same binary label.
    The cost calculation takes into account a possible difference in the total lengths of the segments.
    
    
    Args:
        cl1_len: Total length of the time series that belong to the class 1.
        cl2_len: Total length of the time series that belong to the class 2.
        num_cl1: Number of occurances of the pattern in regions that belong to cl1.
        num_cl2: Number of occurances of the pattern in regions that belong to cl2.
    
    Return: The label name of the region that the pattern is contained in, as well as the normalized number of occurences.
    """
    
    if (num_cl1 + num_cl2 <= 2):
        return 1.0, None, None
    if (cl1_len == 0 or cl2_len == 0):
        return 1.0, None, None
    f = cl1_len / cl2_len
    norm_cl1 = num_cl1 / f
    norm_cl2 = num_cl2
    cost = 1 - (abs(norm_cl1 - norm_cl2 ) / (norm_cl1 + norm_cl2))
    return cost, norm_cl1, norm_cl2

  
def calculate_motif_stats(p, mask, k, m, ez, radius, segment_labels):
    
    """ 
    Calculate some useful statistics for the motifs found.
    
    Args:
        p: A profile object as it is defined in the matrixprofile foundation python library.
        mask: Binary mask used to annotate the time series.
        m: The window size (length of the motif).
        ez: The exclusion zone used.
        radius: The radius that has been used in the experiment.
        segment_labels: List of the two labels that characterize the time series.
    
    Return: List of the statistics
    """
    
    if len(segment_labels) != 2:
        raise ValueError('segment_labels must contain exactly 2 labels')
    
    
    # the first label in the list will be assigned to for the True regions in the mask
    true_label = segment_labels[0]
    
    # the second label in the list will be assigned to for the False regions in the mask
    false_label = segment_labels[1]
    
    output_list = []
    
    cls1_len = np.count_nonzero(mask)
    cls2_len = abs(mask.shape[0] - cls1_len)
    
    for i in range(0, len(p['motifs'])):
        idx, nn1 = p['motifs'][i]['motifs']
        neighbors = p['motifs'][i]['neighbors']
        motif_pair = p['motifs'][i]['motifs']
        start = idx
        end = idx + m - 1
        nn_idx_start = []
        nn_idx_end = []
        for neighbor in neighbors:
            nn_idx_start.append(neighbor)
            nn_idx_end.append(neighbor + m - 1)
        cls1_count = 0
        cls2_count  = 0
        spanning_both = 0
        for nn_start, nn_end in zip(nn_idx_start, nn_idx_end):
            location_in_ts = pattern_loc(nn_start, nn_end, mask, segment_labels)
            if location_in_ts == true_label:
                cls1_count += 1
            elif location_in_ts == false_label:
                cls2_count += 1
            else:
                spanning_both += 1
                
        motif_location = pattern_loc(start, end, mask, segment_labels)
        if motif_location == true_label:
            cls1_count += 1
        elif motif_location == false_label:
            cls2_count += 1
            
        nearest_neighbor_location = pattern_loc(nn1, nn1+m-1, mask, segment_labels)
        if motif_location == true_label:
            cls1_count += 1
        elif motif_location == false_label:
            cls2_count += 1
            
        cost, norm_cls1, norm_cls2 = calc_cost(cls1_len, cls2_len, cls1_count, cls2_count)
        
        maj = ''
        if norm_cls1 == norm_cls2:
            maj = 'None'
        elif norm_cls1 is None and norm_cls2 is None:
            maj = 'None'
        elif norm_cls1 > norm_cls2:
            maj = true_label
        elif norm_cls1 < norm_cls2:
            maj = false_label
            
        output_list.append([i+1, motif_location, nearest_neighbor_location, cls1_count, cls2_count, cost, m, ez, radius, motif_pair, maj])
        
    return output_list
  
def calculate_nn_stats(nn, mask, m, ez, segment_labels, maj_other):

    """ 
        Calculate some useful statistics for a pattern based on its nearest neighbors. That pattern is supposed to be found 
        in another time series and is examined based on its neighbors on the current time series.

        Args:
            nn: The indices of the nearest neighbors in the time series at hand.
            mask: Binary mask used to annotate the time series at hand.
            m: The window size (length of the motif).
            ez: The exclusion zone used.
            segment_labels: List of the two labels that characterize the time series.
            maj_other: The labels of the majority of neighbors the pattern had in the initial time series it was extracted from.

        Return: 

    """


    if len(segment_labels) != 2:
        raise ValueError('segment_labels must contain exactly 2 labels')


# the first label in the list will be assigned to for the True regions in the mask
    true_label = segment_labels[0]

# the second label in the list will be assigned to for the False regions in the mask
    false_label = segment_labels[1]

    cls1_len = np.count_nonzero(mask)
    cls2_len = abs(mask.shape[0] - cls1_len)

    neighbors = nn
    nn_idx_start = []
    nn_idx_end = []
    for neighbor in neighbors:
        nn_idx_start.append(neighbor)
        nn_idx_end.append(neighbor + m - 1)

    cls1_count = 0
    cls2_count  = 0
    spanning_both = 0
    for nn_start, nn_end in zip(nn_idx_start, nn_idx_end):
        location_in_ts = pattern_loc(nn_start, nn_end, mask, segment_labels)
        if location_in_ts == true_label:
            cls1_count += 1
        elif location_in_ts == false_label:
            cls2_count += 1
        else:
            spanning_both += 1

    cost, norm_cls1, norm_cls2 = calc_cost(cls1_len, cls2_len, cls1_count, cls2_count)

    maj = ''
    if norm_cls1 == norm_cls2:
        maj = 'None'
    elif norm_cls1 is None and norm_cls2 is None:
        maj = 'None'
    elif norm_cls1 > norm_cls2:
        maj = true_label
    elif norm_cls1 < norm_cls2:
        maj = false_label


    matching_maj = (maj_other == maj)
    return [nn, cls1_count, cls2_count, ez, cost, matching_maj]


def create_mp(df, motif_len, column, path, include, dask=True):
    """
       Create and Save a univariate/multidimensional matrix profile as a pair of npz files. Input is based on the
       output of (https://stumpy.readthedocs.io/en/latest/api.html#mstump)

       Args:
          df: The DataFrame that contains the multidimensional time series. 
          motif_len: The subsequence window size. 
          columns: A list of the column indexes that are included in the comptutation univariate/multidimensional
          profile.
          path: Path of the directory where the file will be saved.
          dask: A Dask Distributed client that is connected to a Dask scheduler and Dask workers
          include: A list of the column names that must be included in the constrained multidimensional motif search

       Return: 
           Matrix profile distances, matrix profile indexes
    """

    column1=str(column)
    if len(column1)<2:
        if dask==True:
            from dask.distributed import Client, LocalCluster
            with Client(scheduler_port=8782, dashboard_address=None, processes=False,
                        n_workers=4, threads_per_worker=2, memory_limit='50GB') as dask_client:
                mps=stumped(dask_client, df.iloc[:,column], motif_len, include)# Note that a dask client is needed
                if(path):
                    np.savez_compressed(path,mp=mps[:,0],mpi=mps[:,1] )
                print('Univariate with Dask')
                return mps[:,0],mps[:,1]

        mps = stump(df.iloc[:,column], motif_len, include)
        if(path):
            np.savez_compressed(path, mp=mps[:,0],mpi=mps[:,1])
        print('Uvivariate without Dask')
        return mps[:,0],mps[:,1]

    else:
        if dask==True:
            from dask.distributed import Client, LocalCluster
            with Client(scheduler_port=8782, dashboard_address=None,
                        processes=False, n_workers=4, threads_per_worker=2, memory_limit='50GB') as dask_client:
                mps,indices = mstumped(dask_client, df.iloc[:,column], motif_len, include)  # Note that a dask client needed
                if(path):
                    np.savez_compressed(path, mp=mps, mpi=indices)
            print('Multivariate with Dask')
            return mps, indices

        mps,indices = mstump(df.iloc[:,column], motif_len, include) 
        if(path):
            np.savez_compressed(path, mp=mps, mpi=indices)
        print('Multivariate without Dask')
        return mps, indices


     
def segment_ts(mpi, k_optimal, path, L=None, regions=4, excl_factor=5):
        """ 
        Calculation of total change points(segments) we want to divide our region with respect to a
        computed Univariate Matrix Profile. This procedure is illustated through the Fluss Algorithm
        (https://stumpy.readthedocs.io/en/latest/_modules/stumpy/floss.html#fluss). We input a L which is a list of
        integers. The L is a factor which excludes change point detection. It replaces the Arc Curve with 1 depending
        of the size of L multiplied with an exclusion Factor (excl_factor). This algorithm can work for a
        multidimensional DataFrames. User need just to specify the column in mpi. eg mpi[3] so we look for change_points
        in  the 3rd column. In return we provide the locations(indexes) of change_points and the arc-curve which
        are contained in a specific L.

        Args:
           mpi: The one-dimensional matrix profile index where the array corresponds to the matrix profile
           index for a given dimension.
           L: The subsequence length that is set roughly to be one period length.
           This is likely to be the same value as the motif_len, used to compute the matrix profile
           and matrix profile index.
           excl_factor: The multiplying factor for the regime exclusion zone.       
           regions: Number of segments that our space is going to be divided.
           path: Path of the directory where the file will be saved.

        Return: The locations(indexes) of change_points and the arc-curve which are contained in a specific L.
        """
    
        lam=[]
        regimes = [regions]
        output = dict()

        for l in tqdm(L):
            lam.append(l)
            output[l] = [fluss(mpi[k_optimal - 1], L=int(l), n_regimes=int(r), excl_factor=excl_factor) for r in regimes]
            
        if(path):
            np.save(path, output)
        return output, lam

    
def summary_motifs(mot,wholedict,Ranking,show_best=True):
    
    dflist=[]
    for m in mot:
        for typem in range(0,len(wholedict[m])):
            for stats in wholedict[m][typem]:
                dflist.append([m,typem+1,Ranking[m][typem],wholedict[m][typem][0][0],wholedict[m][typem][0][1],wholedict[m][typem][0][2],wholedict[m][typem][0][11],wholedict[m][typem][0][12],wholedict[m][typem][0][13],wholedict[m][typem][0][6],wholedict[m][typem][0][7],wholedict[m][typem][0][8],wholedict[m][typem][0][9],wholedict[m][typem][0][10]])
    df_output = pd.DataFrame(dflist, columns=['Pattern Lenght in Days', 'Motif type','Motif Instances passing from Wash/Rain','Motif Neighbors','Soil Derate Before ','Soil Derate After','Slope at motif %','Slope Before Motif %','Slope ratio','Mean Power Before Motif','Gain of Power %','Mean Power of Neighbors','Mean Precititation of Neighbors','Precipitation Percentage %'])
    if show_best==False:
        return df_output
    else:
        return df_output[df_output['Slope ratio']<0]

            
def get_corrected_matrix_profile(matrix_profile, annotation_vector):
    corrected_matrix_profile = matrix_profile.copy()    
    corrected_matrix_profile[:, 0] = matrix_profile[:, 0] + ((1 - annotation_vector) * np.max(matrix_profile[:, 0]))
    return corrected_matrix_profile


def Ranker(df,mot,miclean,mdclean,df_rains_output,df_wash_output):
    Ranker={}
    wholedict={}
    dates = pd.DataFrame(index=range(len(df)))
    dates = dates.set_index(df.index)
    dates['power']=df['power']
    dates['soil']=df.soiling_derate
    dates['preci']=df.precipitation
    dates['irradiance']=df.poa
    
    for m in mot:
        mscores=[]
        for motif in miclean[m]:
            scores={}
            for index in motif:
                d=dates.index[index]
                scores[index]=0
                for idx,row in df_rains_output.iterrows():
                    if d<row.RainStop and d>row.RainStart:
                        if index in scores.keys():
                            scores[index]+=1         
                for idx,row in df_wash_output.iterrows():
                    if d<row.WashStop and d>row.WashStart:
                        if index in scores.keys():
                             scores[index]+=1
                for idx,row in df_rains_output.iterrows():
                    if d<row.RainStop and d>row.RainStart:
                        if index+m in scores.keys():
                            scores[index]+=1       
                for idx,row in df_wash_output.iterrows():
                    if d<row.WashStop and d>row.WashStart:
                        if index+m in scores.keys():
                            scores[index]+=1      
            score = 0
            for val in scores.values():
                score = score + val
            mscores.append(score)             
        Ranker[m]=mscores

    for m in mot:
        fortype={}
        for mtype in range(0,len(miclean[m])):

            lista=[]
            metritis=0
            soilb=0
            metritisprin=0
            metritismd=0
            metritisprmd=0
            metritisstd=0
            metritisprstd=0
            stoind=0
            soila=0
            meanper=0
            meanperpr=0
            metrslope=0
            metrslopeb=0
            len(miclean[m][mtype])
            for index in miclean[m][mtype][::]:
                
                temp=df.reset_index(drop=True)
                if temp.index[0:index].size == 0:
                    slopep=np.nan  
                else:
                    slope, intercept = np.polyfit(temp.index[index:index+m],temp.power[index:index+m],1)
                    metrslope=metrslope+slope
                    slopep, intercept = np.polyfit(temp.index[0:index],temp.power[0:index],1)
                    metrslopeb= metrslopeb + slopep                
                #mean       
                metritis=metritis + df.power[index+m:index+2*m].mean()  
                metritisprin=metritisprin+ df.power[index-2*m:index-m].mean()
                stoind=stoind+df.power[index:index+m]
                meanper=meanper+df.precipitation[index+m:index+2*m].mean()
                meanperpr=meanperpr+df.precipitation[index-2*m:index-m].mean()
                soilb=soilb+df.soiling_derate[index-2*m:index-m].mean()
                soila=soila+df.soiling_derate[index+m:index+2*m].mean()
            
            
            if len(miclean[m][mtype])==0:
                break;
            metritis=metritis/len(miclean[m][mtype])
            metrslope=metrslope/len(miclean[m][mtype])*100
            metrslopep=metrslopeb/len(miclean[m][mtype])*100
            soilb=soilb/len(miclean[m][mtype])
            soila=soila/len(miclean[m][mtype])
            meanper=meanper/len(miclean[m][mtype])
            meanperpr=meanperpr/len(miclean[m][mtype])
            metritisprin=metritisprin/len(miclean[m][mtype])
            stoind=stoind/len(miclean[m][mtype])
    
            percent_diff = ((metritis - metritisprin)/metritis)*100
            prec_diff=((meanper-meanperpr)/meanper)*100
            slope_ratio=(metrslopep/metrslope)
            lista.append([len(miclean[m][mtype]),soilb,soila,np.min(mdclean[m][mtype]),np.average(mdclean[m][mtype]),np.max(mdclean[m][mtype]),
                          metritisprin,percent_diff,metritis,meanper,prec_diff,metrslope,metrslopep,slope_ratio])
            fortype[mtype]=lista
        wholedict[m]=fortype
    return Ranker ,wholedict


def clean_motifs(corrected,mot,dailypower,max_motifs=25,min_nei=1,max_di=None,cut=None,max_matc=200):
    md={}
    mi={}
    mdtest={}
    mitest={}
    for m in mot:
        md[m],mi[m]= motifs(dailypower,corrected[m][:,0], min_neighbors=min_nei, max_distance=max_di,
                            cutoff=cut, max_matches=max_matc, max_motifs=max_motifs, normalize=True)
    miclean=dict()
    for m in mot:
        outp=[]
        for j in range(0,len(mi[m])):
            outp.append(np.delete(mi[m][j], np.where(mi[m][j] == -1)))
        miclean[m]=outp

    mdclean=dict()
    for m in mot:
        outp=[]
        for j in range(0,len(md[m])):
            outp.append(md[m][j][~np.isnan(md[m][j])])
        mdclean[m]=outp
    return miclean ,mdclean