import argparse
import pathlib

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


def parse_args():
    parser = argparse.ArgumentParser(description="Approximate MFGs")
    parser.add_argument('--game', help='game setting')
    parser.add_argument('--graphon', help='graphon')
    parser.add_argument('--solver', help='solver', choices=['exact', 'boltzmann', 'ppo', 'omd', 'reg', 'regomd'])
    parser.add_argument('--simulator', help='simulator', choices=['exact', 'stochastic'])
    parser.add_argument('--evaluator', help='evaluator', choices=['exact', 'stochastic', 'reg'], default='exact')
    parser.add_argument('--eval_solver', help='eval solver', choices=['exact', 'ppo', 'reg'], default='exact')

    parser.add_argument('--iterations', type=int, help='number of outer iterations', default=500)
    parser.add_argument('--total_iterations', type=int, help='number of inner solver iterations', default=5000)

    parser.add_argument('--eta', type=float, help='temperature parameter', default=1.)

    parser.add_argument('--id', type=int, help='experiment name', default=None)

    parser.add_argument('--results_dir', help='results directory')
    parser.add_argument('--exp_name', help='experiment name')
    parser.add_argument('--verbose', type=int, help='debug outputs', default=0)
    parser.add_argument('--num_alphas', type=int, help='number of discretization points', default=50)

    #parser.add_argument('--env_params', type=dict, help='Environment parameter set: gmfg para, graphon para', default={})
    parser.add_argument('--p', type=float, help='er graphon p', default=0.5)
    parser.add_argument('--a', type=float, help='sbm graphon a', default=0.9)
    parser.add_argument('--b', type=float, help='sbm graphon b', default=0.3)
    parser.add_argument('--c', type=float, help='sbm graphon c', default=0.9)
    parser.add_argument('--theta', type=float, help='exp graphon theta', default=3.0)

    parser.add_argument('--infection_rate', type=float, help='SIS infection_rate', default=0.8)
    parser.add_argument('--recovery_rate', type=float, help='SIS recovery_rate', default=0.2)

    parser.add_argument('--regularization', type=float, help='regularization parameter', default=0.)

    return parser.parse_args()


def generate_config(args):
    return generate_config_from_kw(**{
        'game': args.game,
        'graphon': args.graphon,
        'solver': args.solver,
        'simulator': args.simulator,
        'evaluator': args.evaluator,
        'eval_solver': args.eval_solver,
        'iterations': args.iterations,
        'total_iterations': args.total_iterations,
        'eta': args.eta,
        'results_dir': args.results_dir,
        'exp_name': args.exp_name,
        'id': args.id,
        'verbose': args.verbose,
        'num_alphas': args.num_alphas,
        'p':args.p,
        'a':args.a,
        'b':args.b,
        'c':args.c,
        'theta':args.theta,
        'infection_rate':args.infection_rate,
        'recovery_rate':args.recovery_rate,
        'regularization': args.regularization
    })


def generate_config_from_kw(**kwargs):
    if kwargs['results_dir'] is None:
        kwargs['results_dir'] = "./results/"

    if kwargs['exp_name'] is None:
        kwargs['exp_name'] = "%s_%s_%s_%s_%s_0_0_%f_%d" % (
            kwargs['game'], kwargs['graphon'], kwargs['solver'], kwargs['simulator'], kwargs['evaluator'],
            kwargs['eta'], kwargs['num_alphas'])

    if 'id' in kwargs and kwargs['id'] is not None:
        kwargs['exp_name'] = kwargs['exp_name'] + "_%d" % (kwargs['id'])

    experiment_directory = kwargs['results_dir'] + kwargs['exp_name'] + "/"
    pathlib.Path(experiment_directory).mkdir(parents=True, exist_ok=True)

    if kwargs['game'] == 'Cyber-Graphon':
        game = CyberGraphon
    elif kwargs['game'] == 'Cyber-Het-Graphon':
        game = HeterogeneousCyberGraphon
    elif kwargs['game'] == 'Beach-Graphon':
        game = BeachGraphon
    elif kwargs['game'] == 'SIS-Graphon':
        game = SISGraphon
    elif kwargs['game'] == 'Investment-Graphon':
        game = InvestmentGraphon
    else:
        raise NotImplementedError

    if kwargs['graphon'] == 'unif-att':
        graphon = uniform_attachment_graphon
    elif kwargs['graphon'] == 'rank-att':
        graphon = ranked_attachment_graphon
    elif kwargs['graphon'] == 'er':
        def er_graphon_p(x,y):
            return er_graphon(x,y,kwargs['p'])
        graphon = er_graphon_p
    elif kwargs['graphon'] == 'power':
        graphon = power_law_graphon
    elif kwargs['graphon'] == 'cutoff-power':
        graphon = cutoff_power_law_graphon
    elif kwargs['graphon'] == 'sbm':
        def sbm_graphon_abc(x,y):
            return sbm_graphon(x,y,kwargs['a'],kwargs['b'],kwargs['c'])
        graphon = sbm_graphon_abc
    elif kwargs['graphon'] == 'exp':
        def exp_graphon_theta(x,y):
            return exp_graphon(x,y,kwargs['theta'])
        graphon = exp_graphon_theta
    else:
        raise NotImplementedError

    if kwargs['solver'] == 'exact' or kwargs['solver'] == 'boltzmann':
        from solver.graphon_solver import DiscretizedGraphonExactSolverFinite
        solver = DiscretizedGraphonExactSolverFinite
    elif kwargs['solver'] == 'reg':
        from solver.reg_graphon_solver import reg_DiscretizedGraphonExactSolverFinite
        solver = reg_DiscretizedGraphonExactSolverFinite
    elif kwargs['solver'] == 'omd':
        from solver.omd_graphon_solver import DiscretizedGraphonExactOMDSolverFinite
        solver = DiscretizedGraphonExactOMDSolverFinite
    elif kwargs['solver'] == 'regomd':
        from solver.reg_omd_graphon_solver import reg_DiscretizedGraphonExactOMDSolverFinite
        solver = reg_DiscretizedGraphonExactOMDSolverFinite
    elif kwargs['solver'] == 'ppo':
        from solver.ppo_solver import PPOSolver
        solver = PPOSolver
    else:
        raise NotImplementedError

    if kwargs['simulator'] == 'exact':
        simulator = DiscretizedGraphonExactSimulatorFinite
    elif kwargs['simulator'] == 'stochastic':
        simulator = StochasticSimulator
    else:
        raise NotImplementedError

    if kwargs['evaluator'] == 'exact':
        evaluator = DiscretizedGraphonEvaluatorFinite
    elif kwargs['evaluator'] == 'stochastic':
        evaluator = StochasticEvaluator
    elif kwargs['evaluator'] == 'reg':
        evaluator = reg_DiscretizedGraphonEvaluatorFinite
    else:
        raise NotImplementedError

    if kwargs['eval_solver'] == 'exact':
        from solver.graphon_solver import DiscretizedGraphonExactSolverFinite
        eval_solver = DiscretizedGraphonExactSolverFinite
    elif kwargs['eval_solver'] == 'reg':
        from solver.reg_graphon_solver import reg_DiscretizedGraphonExactSolverFinite
        eval_solver = reg_DiscretizedGraphonExactSolverFinite
    elif kwargs['eval_solver'] == 'ppo':
        from solver.ppo_solver import PPOSolver
        eval_solver = PPOSolver
    else:
        raise NotImplementedError
    
    if kwargs['game'] == 'SIS-Graphon':
        temp_game_config =  {
                "graphon": graphon,
                "infection_rate": kwargs['infection_rate'],
                "recovery_rate": kwargs['recovery_rate'],
            }
    else:
        temp_game_config =  {
                "graphon": graphon,
            }
    

    return {
            # === Algorithm modules ===
            "game": game,
            "solver": solver,
            "simulator": simulator,
            "evaluator": evaluator,
            "eval_solver": eval_solver,
            "graphon": graphon,

            # === General settings ===
            "iterations": kwargs['iterations'],

            # === Default module settings ===
            "game_config": temp_game_config,
            "solver_config": {
                "total_iterations": kwargs['total_iterations'],
                "eta": kwargs['eta'],
                'verbose': kwargs['verbose'],
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
                'regularization': kwargs['regularization'],
            },
            "eval_solver_config": {
                "total_iterations": kwargs['total_iterations'],
                "eta": 0,
                'verbose': kwargs['verbose'],
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
                'regularization': kwargs['regularization'],
            },
            "simulator_config": {
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "evaluator_config": {
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
                'regularization': kwargs['regularization'],
            },

            "experiment_directory": experiment_directory,
        }
    
    '''
    if kwargs['game'] == 'Cyber-Graphon' or kwargs['game'] == 'Cyber-Het-Graphon' or kwargs['game'] == 'Beach-Graphon':
        return {
            # === Algorithm modules ===
            "game": game,
            "solver": solver,
            "simulator": simulator,
            "evaluator": evaluator,
            "eval_solver": eval_solver,

            # === General settings ===
            "iterations": kwargs['iterations'],

            # === Default module settings ===
            "game_config": {
                "graphon": graphon,
                "env_params": kwargs['env_params'],
            },
            "solver_config": {
                "total_iterations": kwargs['total_iterations'],
                "eta": kwargs['eta'],
                'verbose': kwargs['verbose'],
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "eval_solver_config": {
                "total_iterations": kwargs['total_iterations'],
                "eta": 0,
                'verbose': kwargs['verbose'],
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "simulator_config": {
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "evaluator_config": {
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },

            "experiment_directory": experiment_directory,
        }
    else:
        return {
            # === Algorithm modules ===
            "game": game,
            "solver": solver,
            "simulator": simulator,
            "evaluator": evaluator,
            "eval_solver": eval_solver,

            # === General settings ===
            "iterations": kwargs['iterations'],

            # === Default module settings ===
            "game_config": {
                "graphon": graphon,
            },
            "solver_config": {
                "total_iterations": kwargs['total_iterations'],
                "eta": kwargs['eta'],
                'verbose': kwargs['verbose'],
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "eval_solver_config": {
                "total_iterations": kwargs['total_iterations'],
                "eta": 0,
                'verbose': kwargs['verbose'],
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "simulator_config": {
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },
            "evaluator_config": {
                'num_alphas': kwargs['num_alphas'] if 'num_alphas' in kwargs else 101,
            },

            "experiment_directory": experiment_directory,
        }
    '''


def parse_config():
    args = parse_args()
    return generate_config(args)
