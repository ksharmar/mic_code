#!/usr/bin/env python
# coding=utf-8
 
import logging
import operator
import networkx as nx
import math, sys, os, time
import numpy as np
import tensorflow as tf
np.random.seed(0)
import random

class Embedded_IC(object):
    """Embedded_IC for IC model"""

    def __init__(self, dataloader, W, num_neg_samples):
        self._u2idx, self._idx2u = dataloader.get_useridx()
        self._train_cascades = dataloader.get_train_cascades()
        self._net = dataloader.get_structure()
        # tr_labels = dataloader.get_train_labels()
        self._W = W # None for the paper EIC
        self.num_negs = num_neg_samples
        self._user_size = len(self._idx2u)
        
    """ Model training functions """
    
    def _buildComputationGraph(self, emb_dim, lr):
        u = tf.placeholder(tf.int32)
        v = tf.placeholder(tf.int32)
        p_v = tf.placeholder(tf.float32)
        alpha_d = tf.placeholder(tf.float32)

        emb_z = tf.Variable(tf.random_uniform([self._user_size, emb_dim], -1.0, 1.0), name="sender_emb")
        emb_w = tf.Variable(tf.random_uniform([self._user_size, emb_dim], -1.0, 1.0), name="receiver_emb")
        global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        
        u_emb = tf.nn.embedding_lookup(emb_z, u)
        v_emb = tf.nn.embedding_lookup(emb_w, v)
        
        u_0 = tf.slice(u_emb, [0], [1])
        v_0 = tf.slice(v_emb, [0], [1])
        u_1_n = tf.slice(u_emb, [1], [-1])
        v_1_n = tf.slice(v_emb, [1], [-1])
        f = tf.sigmoid(-u_0 - v_0 - tf.reduce_sum(tf.square(tf.subtract(u_1_n, v_1_n)))) # p_uv
        
        loss1 = - tf.multiply(alpha_d, tf.multiply(p_v, tf.log(f + 1e-5)) + tf.multiply(tf.subtract(1.0, p_v), tf.log(1.0 - f +1e-5))) 
        loss2 = - tf.multiply(alpha_d, tf.log(1.0 - f +1e-5))
        
        # lr = tf.train.exponential_decay(lr, global_step, 1000, 0.96, staircase=True)
        train_1 = tf.train.GradientDescentOptimizer(lr).minimize(loss1, global_step=global_step)
        train_2 = tf.train.GradientDescentOptimizer(lr).minimize(loss2, global_step=global_step)
        
        self.u = u
        self.v = v
        self.p_v = p_v
        self.alpha_d = alpha_d
        self.emb_z = emb_z
        self.emb_w = emb_w
        self.u_emb = u_emb
        self.v_emb = v_emb
        self.f = f
        self.loss1 = loss1
        self.loss2 = loss2
        self.train_1 = train_1
        self.train_2 = train_2
        self.global_step = global_step
    
    def train(self, maxiter, freq, emb_dim, lr):
        
        self._buildComputationGraph(emb_dim, lr)
        self._session = tf.Session()
        
        num_cascades = len(self._train_cascades)
        nbProbas = self._sum_nbProbas() / num_cascades
        alpha_dict = {}
        for d, cas in enumerate(self._train_cascades):
            alpha_dict[d] = (len(cas)-1) / nbProbas
        
        pvs_userlist_dict = {} # list of users that are potential influencers of v in cascade s
        for cascadeind, cas in enumerate(self._train_cascades):
            for j in range(1, len(cas)):
                v = cas[j]
                if str(v)+"-"+str(cascadeind) not in pvs_userlist_dict:
                    lookback = 0 if self._W is None or j-self._W < 0 else j-self._W # print lookback, j
                    pvs_userlist_dict[str(v)+"-"+str(cascadeind)] = cas[lookback:j] # all before v are influencers
        
        init = tf.global_variables_initializer()
        self._session.run(init)
        
        oldL = float("-inf")
        L = self._compute_ll(self._train_cascades)
        logging.info("iter:%d, ll:%f", -1, L)  
        print("start, ll:%f", L)
        # sys.exit()
        
        for step in range(maxiter): 
            print(step)
            # sample D
            d = np.random.randint(low=0, high=num_cascades)
            cascade = self._train_cascades[d]
            # sample V
            v = np.random.randint(low=0, high=self._user_size)
            if v == cascade[0]:
                v = (v+1) % self._user_size
                    
            # prepare compute
            alpha = alpha_dict[d] # 100
            v_active = v in set(cascade)
           
            # gradient updates (one update: sgd)

            if v_active:
                p_vd = self._computePvd(v, pvs_userlist_dict[str(v)+"-"+str(d)]) # TODO !!!
            for u in cascade: # can shuffle this - but not shuffled in paper
                if u == v: break
                # updates z_u and w_v through gradient back-prop
                if not v_active:
                    feed_dict = {self.u:u, self.v:v, self.alpha_d:alpha}
                    loss, step, _ = self._session.run([self.loss2, self.global_step, self.train_2], feed_dict=feed_dict)
                    logging.info("loss_step_2: %f", loss)
                    print("loss_step_2: %f", loss)
                else:
                    feed_dict={self.u:u, self.v:v}
                    p_uv = self._session.run(self.f, feed_dict=feed_dict)
                    p_v = p_uv/p_vd 
                    feed_dict = {self.u:u, self.v:v, self.p_v: p_v, self.alpha_d:alpha}
                    loss, step, _ = self._session.run([self.loss1, self.global_step, self.train_1], feed_dict=feed_dict)
                    logging.info("loss_step_1: %f", loss)
                    print("loss_step_1: %f", loss)

            if step % freq == 0:
#                 L = self._compute_ll(self._train_cascades) # UxU inferred or on self._net
#                 if oldL > L:
#                     break
#                 oldL = L
#                 logging.info("iter:%d, ll:%f", step, L)
#                 print("iter:%d, ll:%f", step, L)
                pass
        return self._net
    
    def _sum_nbProbas(self):
        running_sum = 0.0
        for cas in self._train_cascades:
            for i in range(len(cas)):
                running_sum += self._user_size - 1 - i
        return running_sum   
    
    def _computePvd(self, v, ul):
        """Compute 1 - \Pi_u(1-P_uv)"""
        prod = 1.0
        assert len(ul) > 0
        for u in ul:
            feed_dict={self.u:u, self.v:v}
            probs = self._session.run(self.f, feed_dict=feed_dict)
            p_uv = probs # 0.0 if not self._net.has_edge(u, v) else probs # TODO U x U or with has edge check
            prod = prod * (1.0 - p_uv)
        return 1.0 - prod # + 10e-5
    
    def _compute_ll(self, cascade_list):
        ll = 0.0
        for casindex, cascade in enumerate(cascade_list):
            # print(casindex, ll)
            for j in range(1, len(cascade)):
                lookback = 0 if self._W is None or j-self._W < 0 else j-self._W # trained, tuned for convergence on this.. eval?
                ll += math.log(10e-5 + self._computePvd(cascade[j], cascade[lookback:j]))
            active_set = set(cascade)
            inactive = set(range(self._user_size)) - active_set
            inactive = set(random.sample(inactive, self.num_negs))
            for user in inactive:
                ll += math.log(10e-5 + 1.0 - self._computePvd(user, cascade))
        return ll
    
    1, 0
    
def main(_):
    logging.basicConfig(level=logging.INFO)
    t1 = time.time()
    
    # Create parser
    parser = argparse.ArgumentParser(description='Model training')
    # Add arguments
    parser.add_argument('-d', '--datapkl', help='data pickle path.')
    parser.add_argument('-s', '--savepath', help='save nxG and mixwgts path.')
    parser.add_argument('-w', '--window', help='lookback.', default=None, type=int)
    # flags.DEFINE_integer("num_inactive", 5, "compute ll number of inactive users.", type=int)
    parser.add_argument('-m', '--max_iter', default=100, help='max tr iters.', type=int)
    parser.add_argument('-f', '--freq', default=5, help='convergence test freq.', type=int)
    parser.add_argument('-e', '--emb_dim', default=16, help='embedding dimension', type=int)
    parser.add_argument('-lr', '--learning rate', default=10e-2, help='learning rate', type=float)

    # Parse arguments
    args = parser.parse_args()
    dataloader = pkl.load(open(args.datapkl, "rb"))
    
    # read data
    model = Embedded_IC(dataloader, args.W)
    nxG = model.train(args.max_iter, args.freq, args.emb_dim)
    
    # save learned parameters
    nx.write_gpickle(nxG, "em.nxg."+datapkl+".pkl")
  
    t2 = time.time()
    print("Program finished in {} seconds".format(round(t2-t1,3)))

    
if __name__=="__main__":
    main()