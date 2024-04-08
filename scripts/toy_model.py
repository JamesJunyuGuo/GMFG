

import numpy as np
from scipy.optimize import minimize_scalar


#transition matrix 
def Prob(z):
    P = np.zeros((2,2,2))
    P[1,:,0] = 0.3
    P[1,:,1] = 0.7
    P[0,0,1] = z[1]*0.8+0.1
    P[0,0,0] = 0.9- z[1]*0.8
    P[0,1,1] = 0.3+z[1]*0.55
    P[0,1,0] = 0.7-0.55* z[1]
    return P



class Game:
    def __init__(self,W,mu,pi,r,K=2):

        #W is the connection matrix
        #K is the number of clusters
        #r1 is the reward for infection
        #r2 is the reward for wearing a mask
        #mu is the initial state distribution (which is a matrix)
        self.nstate = 2
        self.naction = 2
        self.pi = pi
        self.mean_field = mu
        self.w = W
        self.discount = 0.95
        self.z = np.zeros((2,2))
        #we set the population number as 2 for the time being 
        self.K = K
        self.r = r
        #this is the scale of the regulizer 
        self.lam = 0.1
        self.update_z()

    def h_func(self,x):
    #regulizer
    #x is the policy 
        return -(x*np.log(x)).sum()*self.lam
        # this function is strongly convex corresponding to x


    def reward(self,s,a,k):
        if k==0 or k==1:
            return -self.r[k]*(s)-a
            
        else:
            print("Error")
            
      #this is the reward function for the kth population
    # H = 0(Healthy) ,S =1 (Sick)
    # Y =0 (Yes) , N =1 (Not)
    def update_z(self):
      self.z = self.w @self.mean_field 
      self.z /= self.K
      
    
    
    #define the transition Matrix P( |s,a,z)
    def transition(self,k):
        z = self.z[k]
        P = Prob(z)
        return P

#under the current policy for the k th population, the state transition matrix 
    def get_transition(self,k):
      # this function is used to for the policy evaluation
        transition = np.zeros((self.nstate,self.nstate))
        for s in range(self.nstate):
            for s1 in range(self.nstate):
                transition[s,s1] = self.transition(k)[s,:,s1]@self.pi[k,s,:]
        return transition


    def Population_update(self):
        #one step population update for the k clusters
        #pi is a mixed policy here
        ans = np.zeros((self.K,self.nstate))
        for k in range(self.K):
            P = self.transition(k)
            for s in range(2):
                for s1 in range(2):
                    ans[k][s] += self.mean_field[k][s1]*self.pi[k][s1]@P[s1,:,s]
        self.mean_field = ans.copy()
        return ans
      
      
    def pop_inf(self,iter =1000):
      #this is the stabilized populaion distribution under the current policy
        for i in range(iter):
            self.Population_update()


    def Vh_func(self,k):
      #use policy evaluation to compute the regularized value function
        P = self.get_transition(k)
        reward = np.zeros((self.nstate))
        for state in range(self.nstate):
            #compute the expection of the reward
            reward[state] += self.h_func(pi[k,state,:])
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
        ans = np.zeros((2,2))
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


print(np.zeros((2,2)))