import pickle

from experiments import flex_args_parser
from experiments.flex_reg_trainer import flex_reg_run_experiment


def flex_reg_run(config):
    results = flex_reg_run_experiment(**config)
    
    
    with open(config['experiment_directory'] + 'logs.pkl', 'wb') as f:
        pickle.dump(results, f, 4)
    with open(config['experiment_directory'] + 'config.pkl', 'wb') as f:
        pickle.dump(config, f, 4)
    

if __name__ == '__main__':
    config = flex_args_parser.parse_config()
    flex_reg_run(config)
