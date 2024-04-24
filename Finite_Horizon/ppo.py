import time
import pickle
import pathlib

from solver.policy.random_policy import RandomFinitePolicy
from solver.stoc_reg_omd_graphon_solver import stoc_reg_DiscretizedGraphonExactOMDSolverFinite
from solver.stoc_reg_omd_graphon_solver_2 import stoc_reg_DiscretizedGraphonExactOMDSolverFinite_2
from solver.reg_omd_graphon_solver import reg_DiscretizedGraphonExactOMDSolverFinite
from solver.omd_graphon_solver import DiscretizedGraphonExactOMDSolverFinite
from solver.reg_graphon_solver import reg_DiscretizedGraphonExactSolverFinite
from solver.graphon_solver import DiscretizedGraphonExactSolverFinite
from evaluator.graphon_evaluator import DiscretizedGraphonEvaluatorFinite
from evaluator.reg_graphon_evaluator import reg_DiscretizedGraphonEvaluatorFinite
from evaluator.stochastic_evaluator import StochasticEvaluator
from games.finite.beach import BeachGraphon
from games.finite.cyber import CyberGraphon
from games.finite.cyber_het import HeterogeneousCyberGraphon
from games.finite.investment import InvestmentGraphon
from games.finite.sis import SISGraphon
from games.graphons import uniform_attachment_graphon, er_graphon, ranked_attachment_graphon, power_law_graphon, cutoff_power_law_graphon, sbm_graphon, exp_graphon
from simulator.graphon_simulator import DiscretizedGraphonExactSimulatorFinite
from simulator.stochastic_simulator import StochasticSimulator
from games.mfg_wrapper import MFGGymWrapper

""" Initialize """

def const_graphon_1(x,y):
    return 1
def const_graphon_2(x,y):
    return 0.5
def const_graphon_3(x,y):
    return 0

regularization = 1 
num_alphas = 30
game = BeachGraphon(**{"graphon": exp_graphon})


def env_creator(env_config=None):
    return MFGGymWrapper(game, mu, time_obs_augment=True)

register_env("MFG-v0", env_creator)