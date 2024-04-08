import pickle
import numpy as np

from datacollection import dc_args_parser
from solver.policy.random_policy import RandomFinitePolicy
from games.mfg_wrapper import MFGGymWrapper
from games.fixed_mfg_wrapper import FIXMFGGymWrapper

def datacollect(config):
    results,simu_info = datacollect_procedure(**config)

    
    with open(config['experiment_directory'] + 'logs.pkl', 'wb') as f:
        pickle.dump(results, f, 4)
    with open(config['experiment_directory'] + 'simu_info.pkl', 'wb') as f:
        pickle.dump(simu_info, f, 4)
    #config["game"] = config["game_name"]
    config["game_config"]=None
    with open(config['experiment_directory'] + 'config.pkl', 'wb') as f:
        pickle.dump(config, f, 4)
    


def datacollect_procedure(**config):
    """ Initialize """
    game = config["game"](**config["game_config"])
    simulator = config["simulator"](**config["simulator_config"])
    evaluator = config["evaluator"](**config["evaluator_config"])
    solver = config["solver"](**config["solver_config"])
    eval_solver = config["eval_solver"](**config["eval_solver_config"])

    agentnum = config["agentnum"]
    episodenum = config["episodenum"]
    agentsampler = config["agentsampler"]

    if agentsampler == 'grid':
        alphas = np.linspace(1/agentnum/2, 1-1/agentnum/2, agentnum)
    elif agentsampler == 'random':
        alphas = np.random.uniform(0,1,agentnum)
    else:
        raise NotImplementedError

    """ Initial mean field and policy """
    print("Initializing. ", flush=True)
    policy = RandomFinitePolicy(game.agent_observation_space, game.agent_action_space)
    mu, muinfo = simulator.simulate(game, policy)
    print("Initialized. ", flush=True)

    '''
    print("Now solving for policy. ")
    policy, info = solver.solve(game, mu)

    print("Now simulating mean field. ")
    mu, muinfo = simulator.simulate(game, policy)
    '''

    simu_info = {"agentnum": agentnum, "agentsampler": agentsampler, "episodenum": episodenum, "mu": muinfo, 'sampledalphas': alphas}

    logs = []

    for _ in range(episodenum):
        temp_log = []
        for i in range(agentnum):
            obs_log = []
            reward_log = [] 
            action_log = []
            env = FIXMFGGymWrapper(game, mu, alphas[i], time_obs_augment=False)
            done = 0
            #reward_sum = 0
            observation = env.reset()
            obs_log.append(observation)

            while not done:
                action = policy.act(env.t, observation)
                action_log.append(action)
                observation, reward, done, _ = env.step(action)
                #reward_sum = reward_sum + reward
                obs_log.append(observation)
                reward_log.append(reward)

            temp_log.append({"alpha":alphas[i], "observation": obs_log, "action": action_log, "reward": reward_log})
        
        logs.append(temp_log)

    return logs, simu_info

    

    



    '''
    logs = []

    """ Outer iterations """
    for i in range(config["iterations"]):

        """ New policy """
        log = {}
        t = time.time()
        print("Loop {}: {} Now solving for policy. ".format(i, time.time()-t), flush=True)
        policy, info = solver.solve(game, mu)
        if i >= config["iterations"]-3:
            log = {**log, "solver": info}

        """ New mean field """
        print("Loop {}: {} Now simulating mean field. ".format(i, time.time()-t), flush=True)
        mu, info = simulator.simulate(game, policy)
        if i >= config["iterations"]-3:
            log = {**log, "simulator": info}

        """ Evaluation of the policy under its induced mean field """
        print("Loop {}: {} Now solving for best response. ".format(i, time.time()-t), flush=True)
        best_response, info = eval_solver.solve(game, mu)
        if i >= config["iterations"]-3:
            log = {**log, "best_response": info}
        print("Loop {}: {} Now evaluating exploitability. ".format(i, time.time()-t), flush=True)
        eval_results_pi = evaluator.evaluate(game, mu, policy)
        eval_results_opt = evaluator.evaluate(game, mu, best_response)
        log = {**log, "eval_pi": eval_results_pi, "eval_opt": eval_results_opt}
        print("Loop {}: {} policy {} optimal {} ".format(i, time.time()-t,
                                                         eval_results_pi["eval_mean_returns"],
                                                         eval_results_opt["eval_mean_returns"]), flush=True)

        logs.append(log)

    

    return logs
    '''

if __name__ == '__main__':
    config = dc_args_parser.parse_config()
    datacollect(config)
