import pickle

from experiments import args_parser
from experiments.reg_trainer import reg_run_experiment


def reg_run(config):
    results = reg_run_experiment(**config)
    
    '''
    with open(config['experiment_directory'] + 'logs.pkl', 'wb') as f:
        pickle.dump(results, f, 4)
    with open(config['experiment_directory'] + 'config.pkl', 'wb') as f:
        pickle.dump(config, f, 4)
    '''

if __name__ == '__main__':
    config = args_parser.parse_config()
    reg_run(config)
