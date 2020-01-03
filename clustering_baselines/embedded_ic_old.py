#!/usr/bin/env python
# coding=utf-8
 
import logging
import operator
import networkx as nx
import math, sys, os, time
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.random.set_random_seed(0)

flags = tf.app.flags

flags.DEFINE_string("train_data", None, "training file.")
flags.DEFINE_string("test_data", None, "testing file.")
flags.DEFINE_string("follows_data", None, "follows data file.")
flags.DEFINE_string("save_path", None, "path to save the model.")

flags.DEFINE_integer("emb_dim", 25, "embedding dimension for users.")
flags.DEFINE_integer("usermax", 20, "max # top active users considered.")
flags.DEFINE_integer("maxiter", 100, "maximum train iterations.")
flags.DEFINE_integer("freq", 1, "convergence test frequency.")
flags.DEFINE_integer("stop_criteria", 1, "|ll-difference| <= stop_criteria --> terminate.")
flags.DEFINE_float("lr", 10e-4, "learning rate.") # 10e-4 in paper
FLAGS = flags.FLAGS

class Options(object):
        
    def __init__(self):
        self.train_data = FLAGS.train_data
        self.test_data = FLAGS.test_data
        self.follows_data = FLAGS.follows_data
        self.save_path = FLAGS.save_path
        self.maxiter = FLAGS.maxiter 
        self.freq = FLAGS.freq
        self.usermax = FLAGS.usermax
        self.emb_dim = FLAGS.emb_dim
        self.stop_criteria = FLAGS.stop_criteria
        self.lr = FLAGS.lr

class Embedded_IC(object):
    """EIC (Bourigault) for IC model"""
    
    """ Load cascades (observations) in train and test set and create buildIndex for users """
    def __init__(self, options, session):
        self._options = options
        self._u2idx = {}
        self._idx2u = []
        self._buildIndex()
        # userlist, user-part times in each cascade, user=> his set of paticipated cascades
        self._train_cascades, self._train_cascades_times, self._upart_cas_tr = self._readFromFile(options.train_data)
        self._test_cascades, self._test_cascades_times, self._upart_cas_te = self._readFromFile(options.test_data)
        self._options.train_size = len(self._train_cascades)
        self._options.test_size = len(self._test_cascades)
        logging.info("done: reading cascades data: train size=%d, test set=%d" % (self._options.train_size, self._options.test_size))
        # self._follows_graph = self._buildGraph(options.follows_data) # not needed in Embedded_IC (U x U infered by parameters on users instead of edges)
        self._buildComputationGraph()
        self._session = session
        
    def _buildIndex(self):
        # compute an index of the users that appear at least once in the training and testing cascades (cut-off to maxusers).
        opts = self._options
        activity_count = {}
        for line in open(opts.train_data):
            if len(line.strip()) == 0:
                continue
            timesteps = line.strip().split("|")
            for timestep in timesteps:
                chunks = timestep.split(" ")
                for chunk in chunks:
                    user, timestamp = chunk.split("-")
                    if user not in activity_count: activity_count[user] = 1
                    else: activity_count[user] += 1
        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            timesteps = line.strip().split("|")
            for timestep in timesteps:
                chunks = timestep.split(" ")
                for chunk in chunks:
                    user, timestamp = chunk.split("-")
                    if user not in activity_count: activity_count[user] = 1
                    else: activity_count[user] += 1
        # retain top K users # user_set = train_user_set | test_user_set
        resorted_users = sorted(activity_count.items(), key=operator.itemgetter(1), reverse=True)
        retained_user_np = np.array(resorted_users[0:opts.usermax])[:,0]
        pos = 0
        for user in retained_user_np:
            self._u2idx[user] = pos
            pos += 1
            self._idx2u.append(user)
        opts.user_size = len(self._idx2u)
        logging.info("\n\ndone: building user index: user_size=%d" % (opts.user_size))
        
    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        opts = self._options
        t_cascades = [] # users
        upart_cascades = {}
        t_cascades_times = {}
        casindex = 0
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []; timeslist = []
            activations = line.strip().split(" ")
            for active in activations:
                uname, timestamp = active.split("-")
                timestamp = long(float(timestamp))
                if uname not in self._u2idx:
                    continue
                user = self._u2idx[uname]
                if user in userlist:
                    continue # repeat occurence ignored
                userlist.append(user)
                timeslist.append(timestamp)          
            if len(userlist)>=1:
                t_cascades.append(userlist)
                for user, timestamp in zip(userlist, timeslist):
                    if user not in upart_cascades:
                        upart_cascades[user] = set()
                    upart_cascades[user].add(casindex)
                    t_cascades_times[str(user)+"-"+str(casindex)] = timestamp
                casindex += 1
        return t_cascades, t_cascades_times, upart_cascades

#     def _buildGraph(self, filename):
        
#         """read follows graph (if available) Uses co-participation graph (edges from co-participation in follow graph retained)"""
#         opts = self._options
#         num_users = opts.user_size
#         user_participation = self._upart_cas_tr
#         upart_times = self._train_cascades_times

#         follow_net = nx.DiGraph()
#         if filename is not None:
#             for line in open(filename):
#                 line = line.strip()
#                 uname, vname = line.split(" ")[0], line.split(" ")[1]
#                 if uname in self._u2idx and vname in self._u2idx:
#                     u = self._u2idx[uname]; v = self._u2idx[vname]
#                     if u == v: continue
#                     bool_edge_participates = False
#                     bothactive = user_participation[u] & user_participation[v]
#                     for b in bothactive:
#                         if upart_times[str(u)+"-"+str(b)] < upart_times[str(v)+"-"+str(b)]:
#                             bool_edge_participates = True
#                     if bool_edge_participates: follow_net.add_edge(u, v, kuv=np.random.rand(), suvpl=0, suvmi=0)
#         else:
#             # co-participation graph in train cascades (maybe: with timestamp-based pruning to limit potential influencers)
#             for u in range(0, num_users):
#                 for v in range(0, num_users):
#                     if u==v or follow_net.has_edge(u, v):
#                         continue
#                     bothactive = user_participation[u] & user_participation[v]
#                     for b in bothactive:
#                         if upart_times[str(u)+"-"+str(b)] < upart_times[str(v)+"-"+str(b)]:
#                             follow_net.add_edge(u, v, kuv=np.random.rand(), suvpl=0, suvmi=0)
                            
            
#         # update cascade counts for every selected diffusion edge:
#         # suvpl is set of cascades where u is active and v is active at sometime after u in the cascade
#         # suvmi is the number of cascades with u active and v inactive in the cascade = num cascade with u active - both active
#         for e in follow_net.edges:
#             u = e[0]; v = e[1]
#             edge = follow_net[u][v]
#             uactive = user_participation[u]
#             vactive = user_participation[v]
#             bothactive = uactive & vactive
#             suvmi = len(uactive - bothactive)
#             suvpl = set()
#             for b in bothactive:
#                 if upart_times[str(u)+"-"+str(b)] < upart_times[str(v)+"-"+str(b)]:
#                         suvpl.add(b)
#             follow_net[u][v]['suvmi'] = suvmi
#             follow_net[u][v]['suvpl'] = suvpl
            
#         logging.info("done: reading social graph: #edges=%s", len(follow_net.edges))
#         return follow_net
    
     
    """ Model training functions """    
        
    def _buildComputationGraph(self):
        opts = self._options
        u = tf.placeholder(tf.int32)
        v = tf.placeholder(tf.int32)
        p_v = tf.placeholder(tf.float32)
        alpha_d = tf.placeholder(tf.float32)

        emb_z = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -1.0, 1.0), name="sender_emb")
        emb_w = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -1.0, 1.0), name="receiver_emb")
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
        
        lr = opts.lr
        # lr = tf.train.exponential_decay(opts.lr, global_step, 1000, 0.96, staircase=True)
        train_1 = tf.train.GradientDescentOptimizer(lr).minimize(loss1, global_step=global_step)
        train_2 = tf.train.GradientDescentOptimizer(lr).minimize(loss2, global_step=global_step)
        
        self.u = u
        self.v = v
        self.p_v = p_v
        self.emb_z = emb_z
        self.emb_w = emb_w
        self.u_emb = u_emb
        self.v_emb = v_emb
        self.global_step = global_step
        self.f = f
        self.lr = lr
        self.loss1 = loss1
        self.loss2 = loss2
        self.train_1 = train_1
        self.train_2 = train_2
        self.alpha_d = alpha_d
        
    def _sum_nbProbas(self):
        opts = self._options
        num_users = opts.user_size
        running_sum = 0.0
        for cas in self._train_cascades:
            for i in range(len(cas)):
                running_sum += num_users - 1 - i
        return running_sum   

    def _sampleU(self, v, s):
        ul = set()
        for node in s:
            if node == v:
                break
            ul.add(node)
        return ul
    
    def _computePvd(self, v, ul):
        prod = 1.0
        assert len(ul) > 0
        for u in ul:
            feed_dict={self.u:u, self.v:v}
            p_uv = self._session.run(self.f, feed_dict=feed_dict)
            prod = prod * (1.0 - p_uv)
        return 1.0 - prod + 10e-5

    def compute_ll(self):
        opts = self._options
        ll = 0.0
        for casindex, cascade in enumerate(self._train_cascades):
            for j in range(1, len(cascade)):
                ll += math.log(10e-10+self._computePvd(cascade[j], cascade[:j]))
            active_set = set(cascade)
            inactive = set(range(opts.user_size)) - active_set
            for user in inactive:
                ll += math.log(10e-10+1-self._computePvd(user, cascade))
        return ll
                    
    def train(self):
                
        opts = self._options
        session = self._session
        learning_rate = opts.lr
        num_cascades = len(self._train_cascades)
        num_users = opts.user_size
        nbProbas = self._sum_nbProbas() / num_cascades
        alpha_dict = {}
        for d, cas in enumerate(self._train_cascades):
            alpha_dict[d] = (len(cas)-1) / nbProbas
        pvs_userlist_dict = {} # list of users that are potential influencers of v in cascade s
        for cascadeind, s in enumerate(self._train_cascades):
            for v in s[1:]:
                if str(v)+"-"+str(cascadeind) not in pvs_userlist_dict:
                    pvs_userlist_dict[str(v)+"-"+str(cascadeind)] = self._sampleU(v, self._train_cascades[cascadeind]) # & parents[v]
        # iterate through edges of follow graph to get parents (in general - not needed here)    
        
        init = tf.global_variables_initializer()
        session.run(init)
        self.saver = tf.train.Saver()
        oldL = float("-inf")
        L = self.compute_ll()
        logging.info("iter:%d, ll:%f", -1, L)
        for step in range(opts.maxiter):   
            print "step", step
            # sample D
            d = np.random.randint(low=0, high=num_cascades)
            cascade = self._train_cascades[d]
            # sample V
            v = np.random.randint(low=0, high=num_users)
            if v == cascade[0]:
                v = (v+1) % opts.user_size
                    
            # prepare compute
            alpha = alpha_dict[d] # 100
            v_active = v in set(cascade)
           
            # gradient updates

            if v_active:
                p_vd = self._computePvd(v, pvs_userlist_dict[str(v)+"-"+str(d)])
            for u in cascade: # can shuffle this - but not shuffled in paper
                if u == v: break
                # updates z_u and w_v through gradient back-prop
                if not v_active:
                    feed_dict = {self.u:u, self.v:v, self.alpha_d:alpha}
                    loss, step, _ = session.run([self.loss2, self.global_step, self.train_2], feed_dict=feed_dict)
                    logging.info("loss_step_2: %f", loss)
                else:
                    feed_dict={self.u:u, self.v:v}
                    p_uv = self._session.run(self.f, feed_dict=feed_dict)
                    p_v = p_uv/p_vd 
                    feed_dict = {self.u:u, self.v:v, self.p_v: p_v, self.alpha_d:alpha}
                    loss, step, _ = session.run([self.loss1, self.global_step, self.train_1], feed_dict=feed_dict)
                    logging.info("loss_step_1: %f", loss)

            if step % opts.freq == 0:
                L = self.compute_ll() # UxU inferred
                # if abs(L - oldL) <= opts.stop_criteria:
                #    break
                # if oldL > L:
                #   break
                oldL = L
                logging.info("iter:%d, ll:%f", step, L)
            

def main(_):
    logging.basicConfig(level=logging.INFO)
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Embedded_IC(options, session)
        model.train()
    
if __name__=="__main__":
    tf.app.run()
    # python embedded_ic.py --train_data "sample_train_cascades.txt" --test_data "sample_test_cascades.txt" --follows_data "sample_follows_data.txt"
    # python embedded_ic.py --train_data "./data/kwon/train_cascades.txt" --test_data "sample_test_cascades.txt"
    # Notes
    # 1. very low loss (0) for kwon dataset (need to change sampling or wait and see)
    # 2. change to the sgd eqns in the paper instead
    # 3. change v, d, u (uar) sampling
    # 4. add periodic updates to the weights (exp rep)