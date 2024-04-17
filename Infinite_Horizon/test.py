from games.SIR import SIR_Game
from games.ring import ring_Game
import numpy as np 
import torch 

import logging

def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create file handler and set level to debug
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

if __name__=="__main__":
    logger = setup_logger("example_logger", log_file="test.log", level=logging.DEBUG)
    W = np.array([
        [0.8,0.4],
        [0.4, 0.8]
    ])
    K = len(W)
    game = SIR_Game(W,K=2)
    game1 = ring_Game(W,K=2)

    '''
    test on the SIR model first, and then we test on the ring game 
    '''
    MAX_ITER = 1000

    mf_lst = [ ]
    policy_lst = [ ]
    lr = 4.0
    tol = 1.0 
    threshold = 1e-5

    for i in range(MAX_ITER):
        '''
        obtain the stable mean field under the current policy, 
        and do the policy improvement
        '''
      
        game.mean_field = game.random_mf()
        game.pop_inf(iter=1000)
        pi_prev = game.pi.copy()
        temp = np.zeros((game.K,game.nstate,game.naction))
        for k in range(game.K):
            Q = game.Gamma_q_func(k)
            for s in range(game.nstate):
                q = Q[s]
                temp[k,s] = game.mirror(k,s,q,eta=lr)
        
        game.pi = temp.copy()
        policy_lst.append(pi_prev)
        mf_lst.append(game.mean_field)
        # print(game.pi)
        # print(pi_prev)
        tol_1 = np.sum(np.abs(game.pi-pi_prev))
        tol = min(tol,tol_1)
        if (i+1)%10 ==0 : 
            logger.info("The {} th iteration, the convergence is: tol= {}".format(i+1, tol)) 
        if tol<threshold:
            logger.info("Converge to the equilibrium with tol ={} ".format(tol))
            break 
    
    print("The mean field is ")
    print(" --------------")
    print(game.mean_field)
    print("--------------")
    print("The policy is ")
    print("--------------")
    print("The first population")
    print(game.pi[0])
    print("--------------")
    print("The second  population")
    print(game.pi[1])
    print("--------------")
        
        
        
        
    
            
            
            
    
    