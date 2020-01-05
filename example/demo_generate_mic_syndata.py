import snap
import pandas as pd
import numpy as np
# np.random.seed(31)
import random
# random.seed(0)
import sys
sys.path.append("../")
from parameter_estimation import data_io
from simulator.sample_distribution import *
from simulator import dic_simulator

if __name__=="__main__":
    
    """Input parameters
    """
    graph_dir = '../data/test_data/'
    edge_file = graph_dir + 'graph.txt'
    act_0_file = graph_dir + 'act_prob_0.txt'
    act_1_file = graph_dir + 'act_prob_1.txt'
    edge_file_header = 3
    num_nodes = 1024
    
    pi0 = 0.5
    save_dir = graph_dir + 'll_{}/'.format(pi0)
    labels_file = save_dir + 'labels.txt'
    cascades_file = save_dir + 'cascades.txt'
    train_ids_file = save_dir + 'train_ids.txt'
    
    num_unique_seedsets = 1  # eg. 1024 unique seed sets x (5, 10) times for cascades
    min_replications = 1
    max_replications = 2
    obs_steps = None
    
    # First: generate graph.txt (list of edges) from snap library c++ cmd line as follows
    """
    # ./krongen -o:outpath.txt -m:"0.9 0.5; 0.5 0.3" -i:10
    Core-periphery (-m:..) with 1024 nodes (i:10)
    print('done graph edges generation')
    """
    
    # Second: generate diffusion parameters (eg. act_prob_0/1 for dic model) on each edge above as follows
    """
    act_prob = []
    edge_df = pd.read_csv(edge_file, sep='\t', header=edge_file_header)
    for i in range(len(edge_df)):
        act_prob.append(np.random.uniform(0.0, 1.0, 2))
    act_prob = np.array(act_prob)
    np.savetxt(graph_dir + 'act_prob_0.txt', act_prob[:, 0])
    np.savetxt(graph_dir + 'act_prob_1.txt', act_prob[:, 1])
    print("done assigning and saving activation probabilities/diffusion parameters for K=2")
    sys.exit()
    """
    
    
    # Third: generate mixture cascades under dic
    """
    load back graph edges and activations
    """
    base_graph = data_io.load_base_graph_from_files(num_nodes, edge_file, edge_file_header, act_0_file, act_1_file)
    
    """
    sample seed sets
    """
    unique_seed_sets = sample_seed_sets(np.arange(num_nodes), 2.5, num_unique_seedsets)
    seed_sets = np.array(replicate_seed_sets(unique_seed_sets, min_replications, max_replications))
    print('finished seed set sampling')
    
    """
    generate labels at mix dist pi0 for num_cascades
    """ 
    # generate label for each seed set (label=1 with prob pi1, label=0 with prob pi0)
    labels = np.random.choice(np.arange(2), len(seed_sets), p=[pi0, 1 - pi0])
    np.savetxt(labels_file, labels, fmt='%d')
    print('finished saving labels')
        
    """
    simulate cascades
    """
    cascades = []
    total_len = 0
    for i, (seedset, label) in enumerate(zip(seed_sets, labels)):
        if label == 0: cons = 'act_prob_0'
        else: cons = 'act_prob_1'
        cas = dic_simulator.simulate(seedset, num_simulations=1, network=base_graph, 
                                     num_nodes=num_nodes, act_prob_constant=cons, obs_steps=obs_steps)[0]
        total_len += cas.shape[0]
        cascades.append(cas)
        if i % 100 == 0:
            print("generating = {}/{}".format(i, len(labels)))
    print("done generating %d cascades" % (len(cascades)))
    print("average length of cascade %f" % (total_len/len(cascades)))

    # write cascades (each line is a cascade: u, t, u, t, ..)for userindex timestamp pairs
    f = open(cascades_file, "w")
    for cascade in cascades:
        c = np.array(np.reshape(cascade, -1), dtype=str)
        f.write(",".join(c))
        f.write("\n")
    print("wrote cascades:", cascades_file, " number:", len(cascades))
    
    np.savetxt(train_ids_file, np.arange(len(cascades)), fmt='%d')
    
    print("finished program.")
    



    
    
  



