import time

from solver.policy.random_policy import RandomFinitePolicy
from solver.stoc_reg_omd_graphon_solver import stoc_reg_DiscretizedGraphonExactOMDSolverFinite
from solver.stoc_reg_omd_graphon_solver_2 import stoc_reg_DiscretizedGraphonExactOMDSolverFinite_2
from solver.reg_omd_graphon_solver import reg_DiscretizedGraphonExactOMDSolverFinite
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
""" Initialize """

regularization = 1 
num_alphas = 30
game = BeachGraphon(**{"graphon": sbm_graphon})
simulator = DiscretizedGraphonExactSimulatorFinite(**{})
evaluator = reg_DiscretizedGraphonEvaluatorFinite(**{'regularization':regularization,'num_alphas':num_alphas})
#solver = stoc_reg_DiscretizedGraphonExactOMDSolverFinite(**{'regularization':regularization, 'num_alphas': num_alphas, 'K': 10})
solver = stoc_reg_DiscretizedGraphonExactOMDSolverFinite_2(**{'regularization':regularization, 'num_alphas': 10, 'K': 150})
#solver = reg_DiscretizedGraphonExactOMDSolverFinite(**{'regularization':regularization,'num_alphas':num_alphas})
eval_solver = reg_DiscretizedGraphonExactSolverFinite(**{'regularization':regularization,'num_alphas':num_alphas})
iterations = 50

logs = []

""" Initial mean field and policy """
print("Initializing. ", flush=True)
policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
mu, info = simulator.simulate(game, policy)
print("Initialized. ", flush=True)

""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
    policy, info = solver.solve(game, mu, policy)
    if i >= iterations-3:
        log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, mu, policy)
    eval_results_opt = evaluator.evaluate(game, mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

