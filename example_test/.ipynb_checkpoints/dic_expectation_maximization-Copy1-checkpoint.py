#!/usr/bin/env python
# coding=utf-8

"""
Method: Expectation maximization (Saito: Relaxation) for parameter estimation in discrete IC model
Continuous time input: Relaxation (allows a lookback num of users) instead of discrete timesteps as input
Intference: Unobserved diffusion graph links and unobserved edge influence weights 
(u->v possible if u before v in cascade)

Input: Training cascades [time ordered list of (user, time of activation)]
Output: Infered diffusion graph links with weights
"""

# import operator
# import networkx as nx
# import math, sys, os, time
# import numpy as np
# import tensorflow as tf
# np.random.seed(0)

# class EM_IC(object):
    

#     """ Model training functions """
    
#     def __init__(self, dataloader, W):
#         self._u2idx, self._idx2u = dataloader.get_useridx()
#         self._train_cascades = dataloader.get_train_cascades()
#         self._net = dataloader.get_structure()
#         # tr_labels = dataloader.get_train_labels()
#         self._W = W
        
#     def train(self, maxiter, freq):
        
#         print("\n\nstart: training")
        
#         pvs_userlist_dict = {}; # key="v-cascadeindex" value="userlist of potential influencers for v in the cascade s"
#         for e in self._net.edges:
#             u = e[0]; v = e[1]   
#             edge = self._net[u][v]
#             # need pvs only those needed in the update equation of kuv in the summation term
#             for cascadeind in edge['suvpl']:
#                 # u in parents(v) & A_s_previous:
#                 if str(v)+"-"+str(cascadeind) not in pvs_userlist_dict: # avoiding recompuatation
#                     j = self._get_index_in_cascade(v, self._train_cascades[cascadeind])
#                     lookback = 0 if self._W is None or j-self._W < 0 else j-self._W # print lookback, j
#                     pvs_userlist_dict[str(v)+"-"+str(cascadeind)] = self._train_cascades[cascadeind][lookback:j]
#         for e in self._net.edges:
#             u = e[0]; v = e[1]
#             edge = self._net[u][v]
#             # edge['ratio'] = 1.0 / (len(edge['suvmi']) + len(edge['suvpl']))
#             edge['ratio'] = 1.0 / (edge['suvmi'] + len(edge['suvpl']))
#             # print edge['suvmi'], edge['suvpl']
#             edge['kuv'] = np.random.rand() # assert(edge['kuv'] >= 0.0 and edge['kuv'] <= 1.0)
    
#         # print pvs_userlist_dict
      
#         oldL = float("-inf")  
#         L = self._compute_ll()
#         print("start, ll:%f", L)
#         for step in range(maxiter):
#             # E-step
#             pvs_dict = {}
#             for key, value in pvs_userlist_dict.items():
#                 pvs_dict[key] = self._computePv(long(key.split("-")[0]), value)
#             # M-step
#             for e in self._net.edges:
#                 edge = self._net[e[0]][e[1]]
#                 temp_sum = 0
#                 for cascadeind in edge['suvpl']:
#                     temp_sum += 1.0 / pvs_dict[str(e[1])+"-"+str(cascadeind)]
#                 edge['kuv'] = edge['ratio'] * edge['kuv'] * temp_sum
#                 # assert(edge['kuv'] >= 0.0 and edge['kuv'] <= 1.0)
            
#             if step % freq == 0:
#                 L = self._compute_ll()
#                 if oldL > L:
#                     break
#                 oldL = L
#                 print("iter:%d, ll:%f", step, L)
#                 pass
            
#         print("\n\ndone: training")
#         return self._net
    
#     def _get_index_in_cascade(self, v, cascade):
#         i = 0
#         for node in cascade:
#             if node == v: break
#             i+= 1
#         return i
        
#     def _computePv(self, v, ulist):
#         """Compute 1 - \Pi_u(1-P_uv)"""
#         prod = 1.0
#         assert len(ulist) > 0
#         for u in ulist:
#             # if self._follows_graph.has_edge(u, v): # this check is reqd when called from compute_ll
#             p_uv = 0.0 if not self._net.has_edge(u, v) else self._net[u][v]['kuv']
#             prod = prod * (1.0 - p_uv)
#         return 1.0 - prod # + 10e-5
  
#     def _compute_ll(self):
#         ll = 0.0
#         for casindex, cascade in enumerate(self._train_cascades):
#             for j in range(1, len(cascade)):
#                 lookback = 0 if self._W is None or j-self._W < 0 else j-self._W # trained, tuned for convergence on this.. eval?
#                 ll += math.log(10e-10 + self._computePv(cascade[j], cascade[lookback:j]))
#             active_set = set(cascade)
#             inactive = set(range(len(self._idx2u))) - active_set
#             for user in inactive:
#                 ll += math.log(10e-10 + 1 - self._computePv(user, cascade))
#         return ll

# def main(_):
#     t1 = time.time()
    
#     # Create parser
#     parser = argparse.ArgumentParser(description='Model training')
#     # Add arguments
#     parser.add_argument('-d', '--datapkl', help='data pickle path.')
#     parser.add_argument('-s', '--savepath', help='save nxG and mixwgts path.')
#     parser.add_argument('-w', '--window', help='lookback.', default=None, type=int)
#     # flags.DEFINE_integer("num_inactive", 5, "compute ll number of inactive users.", type=int)
#     parser.add_argument('-m', '--max_iter', default=100, help='max tr iters.', type=int)
#     parser.add_argument('-f', '--freq', default=5, help='convergence test freq.', type=int)

#     # Parse arguments
#     args = parser.parse_args()
#     dataloader = pkl.load(open(args.datapkl, "rb"))
    
#     # read data
#     model = EM_IC(dataloader, args.W)
#     nxG = model.train(args.max_iter, args.freq)
    
#     # save learned parameters
#     nx.write_gpickle(nxG, "em.nxg."+datapkl+".pkl")
  
#     t2 = time.time()
#     print("Program finished in {} seconds".format(round(t2-t1,3)))
    
import json

if __name__=="__main__":
    import json
    input_json_filepath = sys.argv[0]
    input_params = json.load(open(input_json_filepath, "r"))
    print(input_params)