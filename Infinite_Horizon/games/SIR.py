import numpy as np
from scipy.optimize import minimize_scalar
import time
import logging
from scipy.stats import entropy 
def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    '''
    Create console handler and set level to debug
    '''
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    '''
    Create file handler and set level to debug
    '''
   
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

'''
Now we define the transition kernel
'''


class SIR_Game:
    def __init__(self,W,mu, r, pi,K=2,scale = [3,-3]):
        '''
        W is the connection matrix
        K is the number of clusters
        r1 is the reward for infection
        r2 is the reward for wearing a mask
        mu is the initial state distribution (which is a matrix)
        self.lam is the scale of the regularizer
        '''

       
        self.nstate = 2
        self.naction = 2
        self.pi = pi #(2,2,2)
        self.mean_field = mu #(2,2)
        self.w = W # (2,2)
        self.discount = 0.95
        self.z = np.zeros((2,2))
       
        self.K = K
        self.r = np.array(r)
    
        self.lam = 0.2
        self.scale = scale
        self.update_z()
        R_max = np.max(abs(self.r))
        self.R_max = R_max + 2
        

    def h_func(self,x):
        '''
        Define the regulizer, where x is the policy
        this functino is strongly convex with respect to x 
        '''
    
        return self.lam * entropy(x)
    
    
    def transition_probs(self,k,s,a):
        
        
        pass 
    
    '''
    get the transition matrix for policy evaluation
    '''
    
    def get_transition(self,k):
        transition = np.zeros((self.nstate,self.nstate))
        for s in range(self.nstate):
            for s1 in range(self.nstate):
                for a in range(self.naction):
                    transition[s,s1] = self.transition_probs(k,s,a)[s1]@self.pi[k,s,a]
        return transition
    
      


    def reward(self,s,a,k):
        if k in range(self.K):
            return -self.r[k]*(s==1) -(a==0) + self.scale[k]* np.log(self.mean_field[k,s])+ self.R_max
            
        else:
            raise ValueError("choose the right population")
    '''
    This is the reward function for the kth population
    H = 0(Healthy) ,S =1 (Sick)
    Y =0 (Yes) , N =1 (Not)
    '''
            
    def update_z(self):
      self.z = self.w @self.mean_field /self.K 
    
      
    '''
     define the transition Matrix P( |s,a,z)
     Under the current policy for the k th population, the state transition matrix  P(s'|s)
    '''
    
   
    def Population_update(self):
        '''
        One step population update for the k clusters
        pi is a mixed policy here
        '''
        ans = np.zeros((self.K,self.nstate))
        for k in range(self.K):
            P = self.transition(k)
            for s in range(self.nstate):
                for s1 in range(self.nstate):
                    ans[k][s] += self.mean_field[k][s1]*self.pi[k][s1]@P[s1,:,s]
        self.mean_field = ans.copy()
        return ans
      
      
    def pop_inf(self,iter =1000):
        '''
        this is the stabilized populaion distribution under the current policy
        '''
        for _ in range(iter):
            self.Population_update()


    def Vh_func(self,k):
        '''
        use policy evaluation to compute the regularized value function, solve the regularized value function 
        '''
        P = self.get_transition(k)
        reward = np.zeros((self.nstate))
        for state in range(self.nstate):
            #compute the expection of the reward
            reward[state] += self.h_func(self.pi[k,state,:])
            for action in range(self.naction):
                reward[state] += self.reward(state, action,k)*self.pi[k,state,action]
        A = np.zeros((self.nstate,self.nstate))
        A = np.eye(self.nstate) - self.discount* P
        ans = np.linalg.solve(A, reward)
        return ans


    def Qh_func(self,s,a,k):
      #use the regularized value function to compute the regularized q function
        ans = self.reward(s, a, k)+self.h_func(self.pi[k,s,:])
        V = self.Vh_func(k)
        P = self.transition(k)
        for s1 in range(2):
            ans += self.discount* V[s1]*P[s,a,s1]
        return ans

    def qh_func(self,s,a,k):
        ans = self.Qh_func(s, a, k)
        ans -= self.h_func(self.pi[k,s,:])
        return ans

    def Gamma_q_func(self,k):
        ans = np.zeros((self.nstate,self.naction))
        for s in range(self.nstate):
            for a in range(self.naction):
                ans[s,a] = self.Qh_func(s, a, k)
        return ans


    def mirror(self,k,s,eta=0.1):
      #one step policy mirror acent
        q = self.Gamma_q_func(k)
        pi = self.pi[k,:,:]
        def softmax(input):
            temp = input-np.max(input)
            return np.exp(temp)/np.sum(np.exp(temp))
        
        def func(theta):
            u = np.concatenate([theta,[0]])
            v = softmax(u)
            res = (u@q[s] + self.lam*entropy(v))*2*eta - np.sum((v-pi[s])**2)
            return res*-1
        
        ans = minimize_scalar(func, method='bounded')
        theta = ans.x 
        v = np.concatenate([theta,[0]])
        
        return softmax(v)
    
    
    def OMD(self,k,s,tau=0.5):
        vec = np.zeros((self.naction))
        vec = self.Gamma_q_func(k)[s]
        def softmax(input):
            input -= np.max(input)
            '''
            Avoid explosion in the exponential computation 
            '''
            return np.exp(input)/np.sum(np.exp(input))
        
        ans = softmax(tau*vec)
        return ans 
    


# Example usage:
if __name__ == "__main__":
    logger = setup_logger("example_logger", log_file="example.log", level=logging.DEBUG)
    r = [2,2]
    K = len(r)
    mu = np.array([[0.75,0.25],[0.8,0.2]])
    pi1 = np.array([[0.5,0.5],[0.5,0.5]])
    pi2 = np.array([[0.5,0.5],[0.5,0.5]])
    pi = np.array([pi1,pi2])
    W = np.eye(K)*0.2 + np.ones((K,K))*0.6
    scale = [-0.2,0.2]
    obj = GMFG_exact(W,mu,r,pi,K,scale)
    logger.info("Initializing the Game")
    tol = 1
    obj.lam = 0.5
    MAX_ITER = 100000

    lr_lst = [2.0 for i in range(MAX_ITER)]
    policy_lst = []
    mf_lst = []
    '''
    In each iteration, we first obtain the stable mean field under the current policy,
    and then we use the Q-Learning algorithm to obtain the Q-function
    Finally, we leverage the PMA algorithm to update the policy
    '''


    for iter in range(100000):
        t1 = time.time()
        obj.mean_field = mu
        obj.pop_inf()
        obj.update_z()
        
        '''
        Initialize the Mean Field and update the aggregate impact 
        '''
        temp = np.zeros((obj.K,obj.nstate,obj.naction))
        pi_temp = obj.pi.copy()
        for k in range(obj.K):
            for s in range(obj.nstate):
                temp[k,s,:] = (obj.mirror(k, s,eta=lr_lst[iter]))
        obj.pi = temp.copy()
        tol = min(tol,np.sum((np.abs(pi_temp-temp))))
        policy_lst.append(pi_temp)
        x  = obj.mean_field.copy()
        mf_lst.append(x.copy())
        del temp,x
        if (iter+1)% 100 ==0:
            logger.info("The best converge is {} in iteration {}".format(tol,iter+1))
        if tol<1e-8:
            logger.info("The best converge is {}".format(tol))
            break 
    
   
    
   

    
    
        
        