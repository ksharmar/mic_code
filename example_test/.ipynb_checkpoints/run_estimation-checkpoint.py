inside main init
{
"cascades_filename" : "../datasets/twitter_ma/cascades.txt",
"labels_filename" : "../datasets/twitter_ma/labels.txt",
"train_cascade_ids_filename" : "../datasets/twitter_ma/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 100,
"extra_users_len" : 5,
"edge_thr" : 0,
"cascade_count" : 992,
"num_negative_samples" : 100,
"lookback_count" : 10,
"max_iter" : 30,
"save_pi" : "../models/twitter_ma/1_pi.txt",
"save_graph": "../models/twitter_ma/1_learned.graph"
}
53.75, 60 (acc, f1)

run estimation
{
"cascades_filename" : "../datasets/syn_dataset/cascades.txt",
"labels_filename" : "../datasets/syn_dataset/labels.txt",
"train_cascade_ids_filename" : "../datasets/syn_dataset/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 5,
"extra_users_len" : 0,
"num_negative_samples" :100,
"lookback_count" : 2,
"max_iter" : 2,
"save_pi" : "../models/syn_dataset/pi.txt",
"save_graph": "../models/syn_dataset/learned.graph"
}

new run estimation syntest
{
"cascades_filename" : "../datasets/syn_dataset/cascades.txt",
"labels_filename" : "../datasets/syn_dataset/labels.txt",
"train_cascade_ids_filename" : "../datasets/syn_dataset/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 5000,
"extra_users_len" : 0,
"edge_thr" : 30,
"cascade_count" : 1000,
"num_negative_samples" : 100,
"lookback_count" : 5,
"max_iter" : 10,
"save_pi" : "../models/syn_dataset/new_syntest_pi.txt",
"save_graph": "../models/syn_dataset/new_syntest_learned.graph"
}

run_estimation_test
{
"cascades_filename" : "../datasets/syn_dataset/cascades.txt",
"labels_filename" : "../datasets/syn_dataset/labels.txt",
"train_cascade_ids_filename" : "../datasets/syn_dataset/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 5000,
"extra_users_len" : 0,
"edge_thr" : 10,
"cascade_count" : 1000,
"num_negative_samples" : 1000,
"lookback_count" : 5,
"max_iter" : 3,
"save_pi" : "../models/syn_dataset/pi.txt",
"save_graph": "../models/syn_dataset/learned.graph"
}

run_estimation_test0
{
"cascades_filename" : "../datasets/kwon/cascades.txt",
"labels_filename" : "../datasets/kwon/labels.txt",
"train_cascade_ids_filename" : "../datasets/kwon/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 1000,
"extra_users_len" : 10,
"edge_thr" : 0,
"cascade_count" : 111,
"num_negative_samples" : 100,
"lookback_count" : 5,
"max_iter" : 10,
"save_pi" : "../models/kwon/random_rerun_last_pi.txt",
"save_graph": "../models/kwon/random_rerun_learned.graph"
}

run_estimation_test1

{
"cascades_filename" : "../datasets/twitter_ma/cascades.txt",
"labels_filename" : "../datasets/twitter_ma/labels.txt",
"train_cascade_ids_filename" : "../datasets/twitter_ma/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 5000,
"extra_users_len" : 0,
"edge_thr" : 0,
"cascade_count" : 992,
"num_negative_samples" : 100,
"lookback_count" : 5,
"max_iter" : 10,
"save_pi" : "../models/twitter_ma/random_rerun_pi.txt",
"save_graph": "../models/twitter_ma/random_rerun_learned.graph"
}

run_estimation_test2

{
"cascades_filename" : "../datasets/twitter_ma/cascades.txt",
"labels_filename" : "../datasets/twitter_ma/labels.txt",
"train_cascade_ids_filename" : "../datasets/twitter_ma/train_ids.txt",
"method_type" : "mic_train",
"user_max" : 5000,
"extra_users_len" : 0,
"edge_thr" : 5,
"cascade_count" : 992,
"num_negative_samples" : 100,
"lookback_count" : 5,
"max_iter" : 10,
"save_pi" : "../models/twitter_ma/random_rerun_pi.txt",
"save_graph": "../models/twitter_ma/random_rerun_learned.graph"
}


import math, sys, os, time, json
sys.path.append("../")
from parameter_estimation.dic_expectation_maximization import train as dic_train
from parameter_estimation.mic_expectation_maximization import train as mic_train, get_soft_assignment, set_index_dict
from parameter_estimation.load_data_util import load_data_for_parameter_estimation
import numpy as np
import snap
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# np.random.seed(0)

# TODO handle if there are no u->v edges in base graph for test cascades (because base_graph is built from train_cascades)

def main(input_params):
    cascades_filename = input_params['cascades_filename']
    labels_filename = input_params['labels_filename']
    train_cascade_ids_filename = input_params['train_cascade_ids_filename']
    method_type = input_params['method_type']
    user_max = input_params['user_max']
    extra_users_len = input_params['extra_users_len']
    edge_thr = input_params['edge_thr']
    cascade_count = input_params['cascade_count']
    num_negative_samples = input_params['num_negative_samples']
    lookback_count = input_params['lookback_count']
    max_iter = input_params['max_iter']
    save_pi = input_params['save_pi']
    save_graph = input_params['save_graph']

    print("input params...")
    print(cascades_filename)
    print(labels_filename)
    print(train_cascade_ids_filename)
    print(method_type)
    print("User max, Extra users len, Edge thr, Cascade count", user_max, extra_users_len,
        edge_thr, cascade_count)
    print("Num neg samples, lookback_count, max_iter", num_negative_samples,
        lookback_count, max_iter)


    print("loading data...")
    u2idx, idx2u, train_cascades, test_cascades, train_labels, test_labels, base_graph = \
        load_data_for_parameter_estimation(cascades_filename, labels_filename, \
            train_cascade_ids_filename, user_max, extra_users_len,
            lookback_count, edge_thr, cascade_count)
    print("base_graph information...")
    print("num_nodes={}".format(base_graph.GetNodes()))
    print("num_edges={}".format(base_graph.GetEdges()))
    print("num_train_cascades={}".format(len(train_cascades)))
    print("num_test_cascades={}".format(len(test_cascades)))
    print("done loading data...")

    if method_type == "dic_train":
        # train DIC parameter estimation
        dic_train(base_graph, train_cascades, lookback_count=lookback_count, max_iter=max_iter)
    elif method_type == "mic_train":
        st_time = time.time()
        # train MIC parameter estimation
        pi0, pi1 = mic_train(base_graph, train_cascades, train_labels,
            num_negative_samples=num_negative_samples, \
            lookback_count=lookback_count, max_iter=max_iter)
        if save_pi is not None and save_pi != "":
            np.savetxt(save_pi, [pi0, pi1])
            print("saved_pi at location: " + save_pi)
        et_time = time.time()
        print("Training time = {} for {} users", et_time-st_time)
    if save_graph is not None and save_graph != "":
        snap.SaveEdgeList(base_graph, save_graph, "Save as tab-separated list of edges")
        act_0 = []
        act_1 = []
        u_str = []
        v_str = []
        for EI in base_graph.Edges():
            u = EI.GetSrcNId()
            v = EI.GetDstNId()
            act_0.append(base_graph.GetFltAttrDatE(EI, "act_prob_0"))
            act_1.append(base_graph.GetFltAttrDatE(EI, "act_prob_1"))
            u_str.append(idx2u[u])
            v_str.append(idx2u[v])
        df = pd.DataFrame()
        df['u'] = u_str
        df['v'] = v_str
        df['act0'] = act_0
        df['act1'] = act_1
        df.to_csv(save_graph+"df")
        np.savetxt(save_graph+"0", np.array(act_0))
        np.savetxt(save_graph+"1", np.array(act_1))
        idx2u_df = pd.DataFrame()
        idx2u_df['all_users'] = idx2u
        idx2u_df.to_csv(save_graph+"userdf")
        # np.savetxt(save_graph+"u", np.array(u_str))
        # np.savetxt(save_graph+"v", np.array(v_str))
        # FOut = snap.TFOut(save_graph)
        # base_graph.Save(FOut)
        # FOut.Flush()
        print("saved_graph at location: " + save_graph)
    print("done parameter estimation...")

    print("run analysis...")
    if method_type == "mic_train":
        # unsupervised evaluation on train cascades
        index_dict_list = set_index_dict(train_cascades)
        _, gamma1 = get_soft_assignment(train_cascades, pi0, pi1, base_graph,
            lookback_count, num_negative_samples, index_dict_list)
        pred_train_labels = (gamma1 >= 0.5)*1
        # report clustering
        precision, recall, fscore, support = precision_recall_fscore_support(train_labels, pred_train_labels)
        acc = accuracy_score(train_labels, pred_train_labels)
        classification_report = pd.concat(map(pd.DataFrame, [[acc,acc], fscore, precision, recall, support]), axis=1)
        classification_report.columns = ["accuracy", "f1-score", "precision", "recall", "support"]
        print("Clustering results: Classification report")
        print(classification_report)
        # report flipped clustering
        precision, recall, fscore, support = precision_recall_fscore_support(
            train_labels, ~np.array(pred_train_labels, dtype=np.bool))
        acc = accuracy_score(train_labels, ~np.array(pred_train_labels, dtype=np.bool))
        report_flip = pd.concat(map(pd.DataFrame, [[acc,acc], fscore, precision, recall, support]), axis=1)
        report_flip.columns = ["accuracy", "f1-score", "precision", "recall", "support"]
        print("Flipped prediction groups")
        print(report_flip)
    print("done analysis...")
    # 1) Krockner core periphery 1024 nodes is enough (with avg
    # cascade length of 115). 2) Parameter Recoverability. Variation with
    # sample size. (3 subfigures -> 50-50, 20-80, 35-65) distributions. MAEEdges
    # (Component A, B), MAE-Mixing Weights. 3) Cascade separability.
    # Variation with sample size. (3 subfigures -> 50-50, 20-80,
    # 35-65) distributions. F1 (Component A, B), Accuracy with standard
    # deviations over random runs.


if __name__=="__main__":
    start = time.time()
    input_json_filepath = sys.argv[1]
    with open(input_json_filepath, "r") as f:
        input_params = json.load(f)
    main(input_params)
    end = time.time()
    print("Program finished in {} seconds".format(round(end-start, 3)))
