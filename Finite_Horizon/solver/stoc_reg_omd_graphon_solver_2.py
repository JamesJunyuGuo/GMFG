import numpy as np
from scipy.stats import entropy

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from solver.policy.random_policy import RandomFinitePolicy
from simulator.mean_fields.base import MeanField
from solver.base import Solver
from solver.policy.finite_policy import QMaxPolicy, QSoftMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy
from evaluator.base import PolicyEvaluator
from games.mfg_wrapper import MFGGymWrapper



class stoc_reg_DiscretizedGraphonExactOMDSolverFinite_2(Solver):
    """
    Exact OMD solutions for finite state spaces
    use OMD to solve for the exact solutions
    """

    def __init__(self, eta=1, num_alphas=20, regularization=1, K=20,**kwargs):
        super().__init__(**kwargs)
        self.num_alphas = num_alphas
        #print(num_alphas)
        self.alphas = np.linspace(1 / num_alphas / 2, 1 - 1 / num_alphas / 2, num_alphas)
        self.y = None
        self.regularization =regularization
        self.omd_coeff1 = 0.5
        self.omd_coeff2 = 0.5/regularization
        self.eta = eta
        self.K = K

    def solve(self, mfg: FiniteGraphonMeanFieldGame, mu: MeanField, pi: DiscretizedGraphonFeedbackPolicy, **kwargs):
        data_cache = []
        env = MFGGymWrapper(mfg, mu, time_obs_augment=False)
        temp_policy = RandomFinitePolicy(mfg.agent_observation_space, mfg.agent_action_space)
        for alpha in self.alphas:
            alpha_data_log = []
            for _ in range(self.K):
                temp_data_log = []
                observation = env.reset()
                observation = (alpha,observation[1])
                done = 0
                while not done:
                    action = temp_policy.act(env.t, observation)
                    next_observation, reward, done, _ = env.step(action)
                    temp_data_log.append([observation,action,reward,next_observation])
                    observation = next_observation

                alpha_data_log.append(temp_data_log)
            data_cache.append(alpha_data_log)
            #print("finish agent initialization")

        Q_alphas = []

        #print("finish data collection")

        for agent_idx,alpha in zip(range(len(self.alphas)), self.alphas):
            Vs = []
            Qs = []
            curr_V = [0 for _ in range(mfg.agent_observation_space[1].n)]

            for t in range(mfg.time_steps).__reversed__():
                Q_t_pi = []
                for x in range(mfg.agent_observation_space[1].n):
                    x = tuple([alpha, x])

                    Q_tx_pi = []
                    temp_data = [data_cache[agent_idx][repeat][t] for repeat in range(self.K)]
                    for u in range(mfg.agent_action_space.n):
                        transit_est = []
                        temp_samples = []
                        for index in range(self.K):
                            if temp_data[index][0][1]== x[1] and temp_data[index][1]== u:
                                temp_samples.append(temp_data[index][3][1])
                        #print(temp_samples[0])
                        if len(temp_samples):
                            for temp_S in range(mfg.agent_observation_space[1].n):
                                temp_count = temp_samples.count(temp_S)/len(temp_samples)
                                transit_est.append(temp_count)
                        else:
                            #print(alpha)
                            #print(t)
                            for temp_S in range(mfg.agent_observation_space[1].n):
                                transit_est.append(1/mfg.agent_observation_space[1].n)
                        #print(transit_est)
                        temp_Q_est = mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) * np.vdot(curr_V, transit_est)
                        Q_tx_pi.append(temp_Q_est)
                    #           np.vdot(curr_V, mfg.transition_probs(t, x, u, mu))
                    #Q_tx_pi = [mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) *
                    #           np.vdot(curr_V, mfg.transition_probs(t, x, u, mu))
                    #           for u in range(mfg.agent_action_space.n)]
                    
                    #The Q-function at time t is Q_t_pi
                    Q_t_pi.append(Q_tx_pi)

                #curr_V = [np.vdot(Q_t_pi[x], pi.pmf(t, tuple([alpha, x]))) for x in range(len(curr_V))]
                curr_V = [np.vdot(Q_t_pi[x], pi.pmf(t, tuple([alpha, x])))+entropy(pi.pmf(t, tuple([alpha, x]))) for x in range(len(curr_V))]
                Vs.append(curr_V)
                Qs.append(Q_t_pi)

            Vs.reverse()
            Qs.reverse()
            Q_alphas.append(Qs)

        if self.y is None:
            self.y = self.omd_coeff2 * np.array(Q_alphas)
        else:
            self.y = self.omd_coeff1*self.y + self.omd_coeff2 * np.array(Q_alphas)

        policy = DiscretizedGraphonFeedbackPolicy(mfg.agent_observation_space, mfg.agent_action_space,
                                                  [
                                                      QSoftMaxPolicy(mfg.agent_observation_space, mfg.agent_action_space, Qs, 1 / self.eta)
                                                      for Qs, alpha in zip(self.y, self.alphas)
                                                  ], self.alphas)

        return policy, {"Q": self.y}