import pandas as pd
import snap
import numpy as np
import math
import sys
sys.path.append("../")
import os
import time
import json
from parameter_estimation.load_data_util import load_data_for_parameter_estimation
from parameter_estimation.mic_expectation_maximization import train as mic_train
from parameter_estimation.mic_expectation_maximization import last_evaluation
from parameter_estimation import data_io
# np.random.seed(0)


if __name__ == '__main__':
    start = time.time()
    """Input setting
    --------------------
    """
    data = 'test_data'
    cascades_filename = '../data/{}/cascades.txt'.format(data)
    labels_filename = '../data/{}/labels.txt'.format(data)
    train_cascade_ids_filename = '../data/{}/train_ids.txt'.format(data)
    save_pi_file = '../output/{}/pi.txt'.format(data)
    save_edges_file = '../output/{}/learned_graph.tsv'.format(data)  # save_graph_file = '../output/{}/learned.graph'
    save_idx2u_file = '../output/{}/idx2u.txt'.format(data)

    user_max = 1024  # 1000 (kwon) | 5000 (tma)
    extra_users_len = 0  # 10 (kwon) | 0 (tma)
    edge_thr = 5  # 1 (kwon) | 5 (tma)
    lookback_count = 5  # 10 (kwon) | 5 (tma)
    max_iter = 10  # 10 (both)
    cascade_count = 100
    num_negative_samples = None  # unused | 100 (both)

    if not os.path.exists('../output/{}'.format(data)):
        os.makedirs('../output/{}'.format(data))
    
    """Print information, create output dir if not exists
    --------------------
    """
    print("input params...")
    print(cascades_filename)
    print(labels_filename)
    print(train_cascade_ids_filename)
    print("User max, Extra users len, Edge thr, Cascade count",
          user_max, extra_users_len, edge_thr, cascade_count)
    print("Num neg samples, lookback_count, max_iter", num_negative_samples,
          lookback_count, max_iter)
    
    if not os.path.exists('../output/{}/'):
        os.makedirs('../output/{}/') 

    """Load data
    --------------------
    """
    print("loading data...")
    u2idx, idx2u, train_cascades, test_cascades, train_labels, test_labels, base_graph = load_data_for_parameter_estimation(
        cascades_filename, labels_filename, train_cascade_ids_filename, user_max, extra_users_len, lookback_count, edge_thr, cascade_count)
    print("base_graph information...")
    print("num_nodes={}".format(base_graph.GetNodes()))
    print("num_edges={}".format(base_graph.GetEdges()))
    print("num_train_cascades={}".format(len(train_cascades)))
    print("num_test_cascades={}".format(len(test_cascades)))
    print("done loading data...")

    """Train MIC
    --------------------
    """
    st_time = time.time()
    # train MIC parameter estimation
    pi0, pi1 = mic_train(base_graph, train_cascades, train_labels,
                         num_negative_samples=num_negative_samples, lookback_count=lookback_count, max_iter=max_iter)
    et_time = time.time()
    print("Training time = {} for {} users", et_time-st_time)

    """Save learned parameters
    --------------------
    """
    data_io.save_estimated_parameters([pi0, pi1], base_graph, idx2u, save_pi_file, save_edges_file, save_idx2u_file)
    print('finished saving learned parameters..')

    """Compute assignment clusters
    --------------------
    """
    # unsupervised evaluation on train cascades
    last_evaluation(pi0, pi1, base_graph, train_cascades, train_labels, num_negative_samples=None, lookback_count=None)
    print("Program finished in {} seconds".format(round(time.time()-start, 3)))
