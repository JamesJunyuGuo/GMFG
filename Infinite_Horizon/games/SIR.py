import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy 

class SIR_Game:
    def __init__(self,W,mu=None, pi=None,K=2, Controller = None):
        '''
        W is the connection matrix
        K is the number of clusters
        mu is the initial state distribution (which is a matrix)
        self.lam is the scale of the regularizer
        '''

        self.nstate = 3
        self.naction = 3
        self.K = K
        if pi:
            self.pi = pi #(2,2,2)
        
        else:
            self.pi = self.random_policy()
        if mu:
            self.mean_field = mu 
        else:
            self.mean_field = self.random_mf() #(2,2)
        self.w = W # (2,2)
        self.discount = 0.99
        
        self.z = np.zeros((self.K,self.nstate))
        self.lam = 1.0
        self.update_z()
        self.controller = Controller
    
    def z_action(self ):
        res = np.zeros((self.K,  self.naction))
        for k in range(self.K ):
        
            res[k] = self.mean_field[k]@ self.pi[k]
        
        z_action = 1/self.K * self.w @res 
        return z_action

    def h_func(self,x):
        '''
        Define the regulizer, where x is the policy
        this functino is strongly convex with respect to x 
        '''
    
        return self.lam * entropy(x)
    
    
    def random_policy(self):
        res = np.ones((self.K,self.nstate,self.naction))/self.naction
        return res 
    
    def random_mf(self):
        res = np.ones((self.K,self.nstate))/self.nstate
        return res 
    
    
    def transition_probs(self,k,s,a):
        probs = np.zeros((self.nstate))
        z_a = self.z_action()
        
        
        '''
        we should let the probability of the action determine the transition rate 
        '''
        if s ==0 :
            self.update_z()
            #placeholder 
            probs[1] = min(self.z[k][1]*3, 1)
            probs[0] = 1- probs[1]
            
        
        
        elif s==1:
            probs[2] = 0.4
            probs[1] = 0.6
            
            
        
        else:
            probs[0] = 0.4
            probs[2] = 0.6
        return probs  
    
    '''
    get the transition matrix for policy evaluation
    '''
    
    def get_transition(self,k):
        transition = np.zeros((self.nstate,self.nstate))
        for s in range(self.nstate):
            for a in range(self.naction):
                transition[s] += self.transition_probs(k,s,a)*self.pi[k,s,a]
        return transition
    
    
    def reward(self,s,a,k):
        c = [2,1,1]
        if k in range(self.K):
            r_a = 1 - (a-1)**2 *0.5 * c[s]
            r_s = abs(s-1)
            return r_a + r_s
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
            
            for s in range(self.nstate):
                for a in range(self.naction):
                    P = self.transition_probs(k,s,a)
                    ans[k] += self.mean_field[k][s]*self.pi[k][s][a]*P
        self.mean_field = ans.copy()
        return ans 
      
      
    def pop_inf(self,iter =100):
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
        
        ans +=self.discount* V@self.transition_probs(k,s,a)
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


    def mirror(self,k,s,q,eta=0.1):
        prev = self.pi[k,s]
        # q = self.Gamma_q_func(k)[s]
        def softmax(x):
            temp = np.concatenate([x,[0.0]])
            temp = temp-np.max(temp)
            return np.exp(temp)/np.sum(np.exp(temp))
        def func(u):
            def entropy(x):
                return -np.sum(x*np.log(x))
            pi = softmax(u)
            return np.sum((pi-prev)**2) -2*eta*((q@pi) +self.lam* entropy(pi))
        initial_guess = [0,0]
        bound = [(-10,10),(-10,10) ]
        result = minimize(func, initial_guess,bounds=bound)
        theta = result.x 
        return softmax(theta)
        

    
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
    


   