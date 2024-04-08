import numpy as np
from scipy.stats import entropy

from games.graphon_mfg import FiniteGraphonMeanFieldGame
from simulator.mean_fields.base import MeanField
from solver.base import Solver
from solver.policy.finite_policy import QMaxPolicy, QSoftMaxPolicy
from solver.policy.graphon_policy import DiscretizedGraphonFeedbackPolicy
from evaluator.base import PolicyEvaluator



class reg_DiscretizedGraphonExactOMDSolverFinite(Solver):
    """
    Exact OMD solutions for finite state spaces
    """

    def __init__(self, eta=1, num_alphas=100, regularization=1, **kwargs):
        super().__init__(**kwargs)
        self.num_alphas = num_alphas
        self.alphas = np.linspace(1 / num_alphas / 2, 1 - 1 / num_alphas / 2, num_alphas)
        self.y = None
        self.regularization =regularization
        self.omd_coeff1 = 0.5
        self.omd_coeff2 = 0.5/regularization
        self.eta = eta

    def solve(self, mfg: FiniteGraphonMeanFieldGame, mu: MeanField, pi: DiscretizedGraphonFeedbackPolicy, **kwargs):
        Q_alphas = []

        for alpha in self.alphas:
            Vs = []
            Qs = []
            curr_V = [0 for _ in range(mfg.agent_observation_space[1].n)]

            for t in range(mfg.time_steps).__reversed__():
                Q_t_pi = []
                for x in range(mfg.agent_observation_space[1].n):
                    x = tuple([alpha, x])
                    Q_tx_pi = [mfg.reward(t, x, u, mu) + (1 - mfg.done(t, x, u, mu)) *
                               np.vdot(curr_V, mfg.transition_probs(t, x, u, mu))
                               for u in range(mfg.agent_action_space.n)]
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
