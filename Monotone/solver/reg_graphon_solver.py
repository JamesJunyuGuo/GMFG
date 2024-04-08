import numpy as np
from scipy.stats import entropy

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.base import MeanField
from solver.base import Solver
from solver.policy.finite_policy import QMaxPolicy, QSoftMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy


class reg_DiscretizedGraphonExactSolverFinite(Solver):
    """
    Exact solutions for finite state spaces
    """

    def __init__(self, eta=0, num_alphas=101, regularization=1, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.regularization = regularization
        self.num_alphas = num_alphas
        self.alphas = np.linspace(1 / num_alphas / 2, 1 - 1 / num_alphas / 2, self.num_alphas)

    def solve(self, mfg: FiniteGraphonMeanFieldGame, mu: MeanField):
        def softmaxx(a):
            exp_a = np.exp(a)
            softmax_a = exp_a / np.sum(exp_a)
            return softmax_a

        Q_alphas = []
        V_alphas = []
        #policy_alphas = []

        for alpha in self.alphas:
            Vs = []
            Qs = []
            #policies = []
            curr_V = [0 for _ in range(mfg.agent_observation_space[1].n)]

            for t in range(mfg.time_steps).__reversed__():
                Q_t = []
                next_curr_V = []
                #curr_policy=[]
                for x in range(mfg.agent_observation_space[1].n):
                    x = tuple([alpha, x])
                    Q_tx = np.array([mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) *
                                     np.vdot(curr_V, mfg.transition_probs(t, x, u, mu))
                                     for u in range(mfg.agent_action_space.n)])
                    Q_t.append(Q_tx)
                    temp_policy = softmaxx(Q_tx)
                    temp_value = entropy(temp_policy/self.regularization)+ np.vdot(Q_tx,temp_policy)
                    next_curr_V.append(temp_value)
                    #curr_policy.append(temp_policy)
                curr_V = next_curr_V

                Vs.append(curr_V)
                Qs.append(Q_t)
                #policies.append(curr_policy)

            Vs.reverse()
            Qs.reverse()
            #policies.reverse()
            Q_alphas.append(Qs)
            V_alphas.append(Vs)
            #policy_alphas.append(policies)

        def get_policy(Qs):
            if self.regularization != 0:
                return QSoftMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs, 1/self.regularization) #change 1 / self.eta to 1/regularization
            elif self.eta != 0:
                return QSoftMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs, 1/self.eta)
            else:
                return QMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs)
            
        
        policy = DiscretizedGraphonFeedbackPolicy(mfg.agent_observation_space, mfg.agent_action_space,
                                                  [
                                                      get_policy(Qs)
                                                      for Qs, alpha in zip(Q_alphas, self.alphas)
                                                  ], self.alphas)
        

        return policy, {"Q": Q_alphas}
