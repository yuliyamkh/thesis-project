from base import LangChangeModel
import argparse
import agentpy as ap
import numpy as np
import os
from typing import List, Union, Dict

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mechanism', help='Mechanism of language change')
arg_parser.add_argument('--output_dir', default='output_data', help='Output directory')
arg_parser.add_argument('--exp_id', help='Id of the experiment')
arg_parser.add_argument('--simulations', help='Number of simulation runs')


def run_experiment(parameters: Dict,
                   mechanism_name: str,
                   exp_name_suffix: str,
                   experiment_id: int,
                   out_dir: str):

    sample = ap.Sample(parameters=parameters, n=50)
    exp = ap.Experiment(LangChangeModel, sample=sample, iterations=3, record=True)
    exp_results = exp.run(n_jobs=-1, verbose=10)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    exp_results.save(exp_name=f'{mechanism_name}_{exp_name_suffix}',
                     exp_id=experiment_id, path=out_dir)


def run_experiments(mechanism_name: str, min_N: int, max_N: int,
                    k: int, initial_p: float, p_range: List[Union[int, float]],
                    s_range: List[float], n_range: List[float],
                    out_dir: str, experiment_id: int, sims: int):

    common_parameters = {
        'agents': ap.IntRange(min_N, max_N),
        'lingueme': ('A', 'B'),
        'memory_size': 10,
        'initial_frequency': initial_p,
        'number_of_neighbors': k,
        'interactor_selection': False,
        'replicator_selection': False,
        'neutral_change': False,
        'steps': int(sims)
    }

    if mechanism_name == 'neutral_change':
        for r_prob in p_range:
            parameters = {**common_parameters, 'rewiring_probability': r_prob, 'neutral_change': True}
            run_experiment(parameters=parameters,
                           mechanism_name=mechanism_name,
                           exp_name_suffix=r_prob,
                           experiment_id=experiment_id,
                           out_dir=out_dir)

    if mechanism_name == 'replicator_selection':
        for r_prob in p_range:
            for s in s_range:
                parameters = {**common_parameters, 'rewiring_probability': r_prob, 'replicator_selection': True,
                              'selection_pressure': s}
                run_experiment(parameters=parameters,
                               mechanism_name=mechanism_name,
                               exp_name_suffix=f"{r_prob}_{s}",
                               experiment_id=experiment_id,
                               out_dir=out_dir)

    if mechanism_name == 'interactor_selection':
        for r_prob in p_range:
            for s in s_range:
                for n in n_range:
                    parameters = {**common_parameters, 'rewiring_probability': r_prob, 'interactor_selection': True,
                                  'selection_pressure': s, 'n': n, 'leaders': None}
                    run_experiment(parameters=parameters,
                                   mechanism_name=mechanism_name,
                                   exp_name_suffix=f"{r_prob}_{s}_{n}",
                                   experiment_id=experiment_id,
                                   out_dir=out_dir)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    mechanism = args.mechanism
    output_dir = args.output_dir
    exp_id = args.exp_id
    simulations = args.simulations

    run_experiments(mechanism_name=mechanism,
                    min_N=5, max_N=10,
                    initial_p=0.2, k=4,
                    p_range=[0, 0.01, 1],
                    s_range=np.arange(0.1, 1.1, 0.1),
                    n_range=[0.1, 0.2],
                    experiment_id=exp_id,
                    out_dir=output_dir,
                    sims=simulations)
