import time
import pickle
import numpy as np

from solver.policy.random_policy import RandomFinitePolicy
from games.graphons import uniform_attachment_graphon, er_graphon, ranked_attachment_graphon, power_law_graphon, cutoff_power_law_graphon, sbm_graphon, exp_graphon

def integral_reg_run_experiment(**config):
    """ Initialize """

    game = config["game"](**config["game_config"])
    simulator = config["simulator"](**config["simulator_config"])
    evaluator = config["evaluator"](**config["evaluator_config"])
    solver = config["solver"](**config["solver_config"])
    eval_solver = config["eval_solver"](**config["eval_solver_config"])
    
    
    df=open(config["est_add"]+'/logs.pkl','rb')
    est_data=pickle.load(df)
    df.close()
    df=open(config["est_add"]+'/config.pkl','rb')
    data_collect_config=pickle.load(df)
    df.close()

    est_recovery_rate = float(est_data['est_recovery_rate'])
    #est_recovery_rate = 0.2
    est_infection_rate = float(est_data['est_infection_rate'])
    #est_infection_rate = 0.8
    if data_collect_config["graphon_name"]=='exp':
        est_theta = float(est_data['est_theta'])
        #est_theta = 3
        def exp_graphon_theta(x,y):
            return exp_graphon(x,y,est_theta)
        est_graphon = exp_graphon_theta
    
    if data_collect_config["graphon_name"]=='sbm':
        est_a = float(est_data['est_a'])
        est_b = float(est_data['est_b'])
        def sbm_graphon_a_b(x,y):
            return sbm_graphon(x, y, est_a, est_b, 0)
        est_graphon = sbm_graphon_a_b

    def const_graphon_1(x,y):
        return 1
    def const_graphon_2(x,y):
        return 0.5
    def const_graphon_3(x,y):
        return 0
    est_game = config["game"](**{"graphon": est_graphon,
                "infection_rate": est_infection_rate,
                "recovery_rate": est_recovery_rate,
            })
    cont_game_1 = config["game"](**{"graphon": const_graphon_1})
    cont_game_2 = config["game"](**{"graphon": const_graphon_2})
    cont_game_3 = config["game"](**{"graphon": const_graphon_3})
    
    #print(est_game.recovery_rate)
    #print(est_game.infection_rate)
    #print(est_theta)

    #print(game.recovery_rate)
    #print(game.infection_rate)

    big_logs = {}


    logs = []
    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        if i==0:
            print("Initializing the policy")
            policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
        else:
            print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
            policy, info = solver.solve(est_game, mu, policy)
            if i >= config["iterations"]-3:
                log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(est_game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Nominal mean field """
        print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
        nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
        eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)
    
    big_logs = {**big_logs, "our_policy": logs}


    '''    
    logs = []
    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        if i==0:
            print("Initializing the policy")
            policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
        else:
            print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
            policy, info = solver.solve(cont_game_1, mu, policy)
            if i >= config["iterations"]-3:
                log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(cont_game_1, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Nominal mean field """
        print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
        nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
        eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)
    
    
    big_logs = {**big_logs, "const_policy_1": logs}
    

    logs = []
    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        if i==0:
            print("Initializing the policy")
            policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
        else:
            print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
            policy, info = solver.solve(cont_game_2, mu, policy)
            if i >= config["iterations"]-3:
                log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(cont_game_2, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Nominal mean field """
        print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
        nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
        eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)
    
    
    big_logs = {**big_logs, "const_policy_2": logs}

    
    logs = []
    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        if i==0:
            print("Initializing the policy")
            policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
        else:
            print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
            policy, info = solver.solve(cont_game_3, mu, policy)
            if i >= config["iterations"]-3:
                log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(cont_game_3, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Nominal mean field """
        print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
        nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
        eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)
    
    
    big_logs = {**big_logs, "const_policy_3": logs}

    logs = []
    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        if i==0:
            print("Initializing the policy")
            policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
        else:
            print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
            policy, info = solver.solve(game, mu, policy)
            if i >= config["iterations"]-3:
                log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Nominal mean field """
        print("Loop {}: {} Now simulating nonimal mean field. ".format(i, time.time()-t), flush=True)
        nominal_mu, info = simulator.simulate(game, policy) #nominal_mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, nominal_mu) #best_response, info = eval_solver.solve(game, nominal_mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)#eval_results_pi = evaluator.evaluate(game, nominal_mu, policy)
        eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)#eval_results_opt = evaluator.evaluate(game, nominal_mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)

    big_logs = {**big_logs, "nominal_policy": logs}
    '''
    


    return big_logs, data_collect_config
