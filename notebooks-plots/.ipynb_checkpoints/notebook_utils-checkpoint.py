import numpy as np
import operator
import pandas as pd


def read_cascades_file(cascades_filename):
    """
    Returns
    -------
    cascades : list(np.array((None, 2)))
        list of user_str, timestamp array (one array per cascade)
    """
    f = open(cascades_filename, "r")
    cascades = []
    for line in f.readlines():
        u_t = line.strip("\n").split(",")
        u = list(map(int, u_t[0::2]))  # int
        t = list(map(float, u_t[1::2]))  # float
        cascade = np.vstack([u, t]).transpose()
        cascades.append(cascade)
    f.close()
    return cascades


def get_engagement_counts(true_cascades, fake_cascades):
    # distribution of engagement counts
    u_t = {}
    for cas in true_cascades:
        for u in cas[:,0]:
            u = int(u)
            if u in u_t: u_t[u] += 1
            else: u_t[u] = 1
    sorted_t = np.array(sorted(u_t.items(), key=operator.itemgetter(1), reverse=True), dtype=np.int32)

    u_f = {}
    for cas in fake_cascades:
        for u in cas[:,0]:
            u = int(u)
            if u in u_f: u_f[u] += 1
            else: u_f[u] = 1
    sorted_f = np.array(sorted(u_f.items(), key=operator.itemgetter(1), reverse=True), dtype=np.int32)
    return u_t, u_f, sorted_t, sorted_f



def find_tweetid_for_userid(list_userids):
    
    
    return list_tweetids

