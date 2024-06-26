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
# game1 = BeachGraphon(**{"graphon": const_graphon_1})
# game2 = BeachGraphon(**{"graphon": const_graphon_2})
# game3 = BeachGraphon(**{"graphon": const_graphon_3})
simulator = DiscretizedGraphonExactSimulatorFinite(**{})
evaluator = reg_DiscretizedGraphonEvaluatorFinite(**{'regularization':regularization,'num_alphas':num_alphas})
#solver = stoc_reg_DiscretizedGraphonExactOMDSolverFinite(**{'regularization':regularization, 'num_alphas': num_alphas, 'K': 10})
solver1 = stoc_reg_DiscretizedGraphonExactOMDSolverFinite_2(**{'regularization':regularization, 'num_alphas': 5, 'K': 300})
# solver2 = stoc_reg_DiscretizedGraphonExactOMDSolverFinite_2(**{'regularization':regularization, 'num_alphas': 10, 'K': 100})
solver3 = reg_DiscretizedGraphonExactOMDSolverFinite(**{'regularization':regularization,'num_alphas':num_alphas})
solver4 = DiscretizedGraphonExactOMDSolverFinite(**{'num_alphas':num_alphas})
# solver5 = PPOSolver(**{'total_iteration':20,'eta':0.2})

eval_solver = reg_DiscretizedGraphonExactSolverFinite(**{'regularization':regularization,'num_alphas':num_alphas})
iterations = 25

big_logs = {}


logs = []





""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
        
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver3.solve(game, mu,policy)
        
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "our_policy1": logs}
'''
logs = []
""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        print("Initializing the policy")
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver2.solve(game, mu, policy)
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "our_policy2": logs}
'''
'''
logs = []
""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        print("Initializing the policy")
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver3.solve(game, mu, policy)
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "our_policy_exact": logs}

logs = []
""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        print("Initializing the policy")
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver4.solve(game, mu)
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "unreg_policy": logs}

logs = []
""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        print("Initializing the policy")
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver3.solve(game1, mu, policy)
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game1, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "uni_graphon1": logs}

logs = []
""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        print("Initializing the policy")
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver3.solve(game2, mu, policy)
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game2, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "uni_graphon2": logs}

logs = []
""" Outer iterations """
for i in range(iterations):

    """ New policy """
    log = {}
    t = time.time()
    if i==0:
        print("Initializing the policy")
        policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    else:
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver3.solve(game3, mu, policy)
        if i >= iterations-3:
            log = {**log, "solver": info}

    """ New mean field """
    print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
    mu, info = simulator.simulate(game3, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Nominal mean field """
    print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
    nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
    if i >= iterations-3:
        log = {**log, "simulator": info}

    """ Evaluation of the policy under its induced mean field """
    print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
    best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
    if i >= iterations-3:
        log = {**log, "best_response": info}
    print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
    eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
    eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
    log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
    print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                        eval_results_pi["eval_mean_returns"],
                                                        eval_results_opt["eval_mean_returns"]), flush=True)

    logs.append(log)

big_logs = {**big_logs, "uni_graphon3": logs}
'''
filepath = './integral_result_new/beach_exp4_8_100_5/'
pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
with open(filepath + 'logs.pkl', 'wb') as f:
    pickle.dump(big_logs, f, 4)
