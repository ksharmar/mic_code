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
    data = 'tma'
    out = 'all_tma/tma_ll'
    cascades_filename = '../data/{}/cascades.txt'.format(data)
    labels_filename = '../data/{}/labels.txt'.format(data)
    train_cascade_ids_filename = '../data/{}/ll_train_ids.txt'.format(data)
    save_pi_file = '../output/{}/pi.txt'.format(out)
    save_edges_file = '../output/{}/learned_graph.tsv'.format(out)  # save_graph_file = '../output/{}/learned.graph'
    save_idx2u_file = '../output/{}/idx2u.txt'.format(out)
    save_resp_file = '../output/{}/resp.txt'.format(out)  # responsibilities (preds) [gamma_0, gamma_1, target_label]
    log_file = '../output/{}/log.txt'.format(out)

    user_max = 7000  # atleast 10 engagements
    extra_users_len, min_cas_length = 0, 1
    
    edge_thr = 5 # 5
    lookback_count = 5  # 5
    
    cascade_count = 992
    max_iter = 10
    num_negative_samples = None
    
    if not os.path.exists('../output/{}'.format(out)):
        os.makedirs('../output/{}'.format(out))
    
    logf = open(log_file, 'w')
    sys.stdout = data_io.Runlog(sys.stdout, logf)  # This will go to stdout and the file out.txt
    
    """Print information, create output dir if not exists
    --------------------
    """
    print("input params...")
    print(cascades_filename)
    print(labels_filename)
    print(train_cascade_ids_filename)
    print('user_max=', user_max)
    print('extra_users_len/min_cas_length=', extra_users_len, min_cas_length)
    print('edge_thr,lookback_count=', edge_thr, lookback_count)
    print('cascade_count, max_iter, num_negative_samples=', cascade_count, max_iter, num_negative_samples)

    """Load data
    --------------------
    """
    print("loading data...")
    u2idx, idx2u, train_cascades, train_labels, filtered_train_cids, test_cascades, test_labels, base_graph = load_data_for_parameter_estimation(
        cascades_filename, labels_filename, train_cascade_ids_filename, user_max, extra_users_len, lookback_count, edge_thr, cascade_count, min_cas_length)
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
    gamma_0, gamma_1, targets = last_evaluation(pi0, pi1, base_graph, train_cascades, train_labels, num_negative_samples, lookback_count)
    stacked = np.vstack([gamma_0, gamma_1, targets, filtered_train_cids]).transpose()
    np.savetxt(save_resp_file, stacked)
    
    tc_list, tl_list = [], []
    for tc, tl in zip(test_cascades, test_labels):
        if len(tc) == 0: continue
        tc_list.append(tc); tl_list.append(tl)
    gamma_0, gamma_1, targets = last_evaluation(pi0, pi1, base_graph, tc_list, tl_list, num_negative_samples, lookback_count)
    
    print('finished saving responsibilities.')
    print("Program finished in {} seconds".format(round(time.time()-start, 3)))
