import numpy as np
from scipy.optimize import minimize_scalar
import time
import logging

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

def Prob(z):
    P = np.zeros((2,2,2))
    P[1,:,0] = 0.2
    P[1,:,1] = 0.8
    P[0,1,1] = 0.3 + 0.5*z[1]
    P[0,1,0] = 1- P[0,1,1]
    P[0,0,1] = 0.1 + 0.2*z[1]
    P[0,0,0] = 1 - P[0,0,1]
    return P


class GMFG_exact:
    def __init__(self,W,mu,r, pi,K=2,scale = [3,-3]):
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
    
        return -(x*np.log(x)).sum()*self.lam
      


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
      self.z = self.w @self.mean_field 
      self.z /= self.K
      
    '''
     define the transition Matrix P( |s,a,z)
     Under the current policy for the k th population, the state transition matrix  P(s'|s)
    '''
    
   
    def transition(self,k):
        z = self.z[k]
        P = Prob(z)
        return P
    
    '''
    get the transition matrix for policy evaluation
    '''
    def get_transition(self,k):
        transition = np.zeros((self.nstate,self.nstate))
        for s in range(self.nstate):
            for s1 in range(self.nstate):
                transition[s,s1] = self.transition(k)[s,:,s1]@self.pi[k,s,:]
        return transition
    
    


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
        for i in range(iter):
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
        #mirror descent step
        def func(u):
            res = (u*q[s,0]+(1-u)*q[s,1]+(-u*np.log(u)-(1-u)*np.log(1-u))*self.lam)*2*eta-((u-pi[s,0])**2+((1-u)-pi[s,1])**2)
            return res*-1
        interval = (0,1)
        ans = minimize_scalar(func, bounds=interval, method='bounded')
        return np.array([ans.x,1-ans.x])
    
    
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
    
    def randomized_policy(self):
        '''
        The shape of the policy should be (self.K ,self.nstate,self.naction)
        '''
        random_policy = np.ones((self.K,self.nstate,self.naction))/self.naction
        self.pi = random_policy
    
    def randomized_mf(self):
        '''
        The shape of the mean field should be (self.K, self.nstate)
        '''
        random_mf = np.ones((self.K,self.nstate))/self.nstate
        
        self.mean_field = random_mf
        
    

def evaluate(game : GMFG_exact,id=0) : 
    game.randomized_mf()
    V_func_1 = game.Vh_func(id)
    return V_func_1@game.mean_field[id] 
    
    

    
