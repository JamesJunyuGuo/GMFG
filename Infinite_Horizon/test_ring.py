from games.SIR import SIR_Game
from games.ring import ring_Game
from games.controller import ring_control
import numpy as np 
import torch 

import logging
import  matplotlib.pyplot as plt 
plt.style.use('ggplot')

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
    # control = ring_control(np.arange(11))
    game = ring_Game(W,K=2,nstate=21)
  

    '''
    test on the SIR model first, and then we test on the ring game 
    '''
    MAX_ITER = 1000

    mf_lst = [ ]
    policy_lst = [ ]
    lr = 8.0
    tol = 1.0 
    threshold = 1e-4*2

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
        tol_1 = np.max(np.abs(game.pi-pi_prev))
        tol = min(tol,tol_1)
        if (i+1)%10 ==0 : 
            logger.info("The {} th iteration, the convergence is: tol= {}".format(i+1, tol)) 
        if tol<threshold:
            logger.info("Converge to the equilibrium with tol ={} ".format(tol))
            break 
    
    
    fig,axs = plt.subplots(5,1,dpi=200)
    
    fig,axs = plt.subplots(4,1,dpi=100)
    mf = np.mean(game.mean_field,axis=0)
    pi = game.pi[0]
    axs[0].plot(mf,label = "Mean Field ",color = 'blue')
    axs[0].set_title("Mean Field")
    axs[1].plot(pi[:,0])
    axs[1].set_title("Turn Left")
    axs[2].plot(pi[:,1])
    axs[2].set_title("Stay Still")
    axs[3].plot(pi[:,2])
    axs[3].set_title("Turn Right")

    fig.tight_layout()
    plt.savefig("./result/ring_{}.png".format(game.nstate))
    plt.show()
        
        
        
        
    
            
            
            
    
    