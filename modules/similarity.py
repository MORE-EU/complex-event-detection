import numpy as np
from matrixprofile import core
from matrixprofile.algorithms import top_k_motifs
from matrixprofile.algorithms.mass2 import mass2
from modules.patterns import to_mpf


def get_top_k_motifs(df, mp, index, m, ez, radius, k, max_neighbors=50):

    """ Given a matrix profile, a matrix profile index, the window size and the DataFrame that contains a multi-dimensional timeseries,
        Find the top k motifs in the timeseries, as well as neighbors that are within the range <radius * min_mp_value> of each of the top k motifs.
        Uses an extended version of the top_k_motifs function from matrixprofile foundation library that is compatible with multi-dimensional                 timeseries.
        The implementation can be found here (https://github.com/MORE-EU/matrixprofile/blob/master/matrixprofile/algorithms/top_k_motifs.py)
    :param df: DataFrame that contains the multi-dimensional timeseries that was used to calculate the matrix profile.
    :param mp: A multi-dimensional matrix profile.
    :param index: The matrix profile index that accompanies the matrix profile.
    :param m: The subsequence window size.
    :param ez: The exclusion zone to use.
    :param radius: The radius to use.
    :param k: The number of the top motifs that were found.
    :param max_neighbors: The maximum amount of neighbors to find for each of the top k motifs.
    """

    np_df = df.to_numpy()
    mp = np.nan_to_num(mp, nan=np.nanmax(mp))  # remove nan values
    profile = to_mpf(mp, index, m, np_df)
    exclusion_zone = int(np.floor(m * ez))
    p = top_k_motifs.top_k_motifs(profile, k=k, radius=radius, exclusion_zone=exclusion_zone,  max_neighbors=max_neighbors)
    return p


def find_neighbors(query, ts, w, min_dist, exclusion_zone=None, max_neighbors=100, radius=3):

    """ Given a query of length w, search for similar patterns in the timeseries ts. Patterns with a distance less than (radius * min_dist)
        from the query are considered similar(neighbors). This function supports multi-dimensional queries and time series. 
        The distance is calculated based on the multi-dimensional distance profile as described at 
        (https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf). This function is implemented based on the univariate 
        apporaches ofthe matrix profile foundation library.
    
    :param query: The query that will be compared against the time series. Can be univariate or multi-dimensional.
    :param ts: A time series. Can be univariate or multi-dimensional.
    :param w: The subsequence window size (should be the length of the query).
    :param min_dist: The minimum distance that will be multiplied with radius to compute
                     maximum distance allowed for a subsequence to be considered similar.
    :param exclusion_zone: The exclusion zone to use.
    :param max_neighbors: The maximum amount of neighbors to find for each of the top k motifs.
    :param radius: The radius to multiply min_dist with in order to create the maximum distance allowed for a subsequence to be considered similar.
    """
    
    window_size = w
    ts = ts.T
    query = query.T
    dims = ts.shape[0]
    data_len = ts.shape[1]
    dp_len = data_len - window_size + 1
    
    if exclusion_zone is None:
        exclusion_zone = 0

    # compute distance profile using mass2 for first appearance
    # create the multi dimensional distance profile

    md_distance_profile = np.zeros((dims, dp_len), dtype='complex128')
    for i in range(0, dims):
        ts_i = ts[i, :]
        query_i = query[i, :]
        md_distance_profile[i, :] = mass2(ts_i, query_i)

    D = md_distance_profile
    D.sort(axis=0, kind="mergesort")
    D_prime = np.zeros(dp_len)
    for i in range(dims):
        D_prime = D_prime + D[i]
        D[i, :] = D_prime / (i + 1)

    # reassign to keep compatibility with the rest of the code
    distance_profile = D[dims - 1, :]

    # find up to max_neighbors taking into account the radius and exclusion zone
    neighbors = []
    n_dists = []
    for j in range(max_neighbors):
        neighbor_idx = np.argmin(distance_profile)
        neighbor_dist = distance_profile[neighbor_idx]
        not_in_radius = not ((radius * min_dist) >= neighbor_dist)

        # no more neighbors exist based on radius
        if core.is_nan_inf(neighbor_dist) or not_in_radius:
            break

        # add neighbor and apply exclusion zone
        neighbors.append(neighbor_idx)
        n_dists.append(np.real(neighbor_dist))
        distance_profile = core.apply_exclusion_zone(
            exclusion_zone,
            False,
            window_size,
            data_len,
            neighbor_idx,
            distance_profile
        )
        
    # return the list of neighbor indices and the respective distances
    return neighbors, n_dists


def pairwise_dist(q1, q2):
    
    """ Calculates the distance between two time series sequences q1, q2. The distance is calculated based on the multi-dimensional distance profile.
        This function allows for the comparison of univariate and multi-dimensional sequences.
    :param q1: A time series sequence.
    :param q2: A time series sequence.
    """
    min_dist = float('inf')
    m = len(q1)
    _, nn_dist = find_neighbors(q1, q2, m, exclusion_zone=None, min_dist=min_dist, max_neighbors=1)
    pair_dist = nn_dist[0]
    return pair_dist
