# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append("../")
from simulator.dic_simulator import simulate
from simulator.sample_distribution import sample_from_power_law, sample_seed_sets
from simulator.greedy_clef import select_seeds_greedy_clef
import snap
import pandas as pd
import pickle
import json


def simulate_and_save(seed_sets, estimator, save_file):
    json_obj = []  
    for seed_set in seed_sets:
        exp, std = estimator.get_expected_influence_per_timestep(seed_set, num_simulations)
        json_obj.append((list(seed_set), list(exp), list(std)))
    w = open(save_file, 'w')
    w.write(json.dumps(json_obj))
    # a = json.loads(w.read())
    print('saved to {}'.format(save_file))
    

if __name__=='__main__':
    start = time.time()
    """Input setting
    --------------------
    """
    data = 'twitter_ma'
    fake_component = 0
    
    pi_file = '../output/{}/pi.txt'
    edges_file = '../output/{}/learned_graph.tsv'
    idx2u_file = '../output/{}/idx2u.txt'
    infl_users_file = '../output/{}/selected_influential_users.tsv'
    
    save_simcascades_trueinf_theta_t = '../output/{}/simcascades_trueinf_theta_t.txt'
    save_simcascades_trueinf_theta_f = '../output/{}/simcascades_trueinf_theta_f.txt'
    
    save_simcascades_fakeinf_theta_t = '../output/{}/simcascades_fakeinf_theta_t.txt'
    save_simcascades_fakeinf_theta_f = '../output/{}/simcascades_fakeinf_theta_f.txt'
    
    save_simcascades_users_theta_t = '../output/{}/simcascades_users_theta_t.txt'
    save_simcascades_users_theta_f = '../output/{}/simcascades_users_theta_f.txt'
        
    num_seed_sets = 10
    num_simulations = 500
    obs_steps = None
    
    """Load graph and activation probabilities (learned and saved from parameter estimation)
    And Load selected influential users for seed set sampling if needed (estimated for fake component as specified in input) 
    --------------------
    """
    pi0, pi1, base_graph, idx2u, u2idx = data_io.load_estimated_parameters(pi_file, edges_file, idx2u_file)
    fake_influential_users = data_io.load_selected_infl_users(infl_users_file, component=fake_component)
    true_influential_users = data_io.load_selected_infl_users(infl_users_file, component=1-fake_component)
    num_nodes = base_graph.GetNodes()
    assert num_nodes == len(idx2u), 'inconsistent num of nodes in base_graph and in idx2u'
    
    fake_infl_estimator = InfluenceEstimator(base_graph, num_nodes, 'act_prob_{}'.format(fake_component), obs_steps)
    true_infl_estimator = InfluenceEstimator(base_graph, num_nodes, 'act_prob_{}'.format(1 - fake_component), obs_steps)
    
    """Sample seed sets and get simulated cascades for each seed set [true infl -> theta_t and theta_f]
    """
    print('true infl -> theta_t, theta_f')
    seed_sets = sample_seed_sets(true_influential_users, alpha=2.5, num_seed_sets=num_seed_sets)
    
    simulate_and_save(seed_sets, true_infl_estimator, save_simcascades_trueinf_theta_t)
    simulate_and_save(seed_sets, fake_infl_estimator, save_simcascades_trueinf_theta_f)

    """Sample seed sets and get simulated cascades for each seed set [fake infl -> theta_t and theta_f]
    """
    print('fake infl -> theta_t, theta_f')
    seed_sets = sample_seed_sets(fake_influential_users, alpha=2.5, num_seed_sets=num_seed_sets)
    
    simulate_and_save(seed_sets, true_infl_estimator, save_simcascades_fakeinf_theta_t)
    simulate_and_save(seed_sets, fake_infl_estimator, save_simcascades_fakeinf_theta_f)
    
    
    """Sample seed sets and get simulated cascades for each seed set [any users -> theta_t and theta_f]
    """
    print('users -> theta_t, theta_f')
    seed_sets = sample_seed_sets(np.arange(len(idx2u)), alpha=2.5, num_seed_sets=num_seed_sets)
    
    simulate_and_save(seed_sets, true_infl_estimator, save_simcascades_users_theta_t)
    simulate_and_save(seed_sets, fake_infl_estimator, save_simcascades_users_theta_f)
    
    print("finished.")


