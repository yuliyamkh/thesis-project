from model import LangChangeModel
import argparse
import agentpy as ap
import numpy as np
import os
from typing import List, Union, Dict

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--mechanism', type=str, help='Mechanism of language change')
arg_parser.add_argument('--output_dir', default='output_data', help='Output directory')
arg_parser.add_argument('--exp_id', type=int, help='Id of the experiment')
arg_parser.add_argument('--simulations', type=int, help='Number of simulation runs')
arg_parser.add_argument('--min_N', type=int, default=10, help='Minimal population size')
arg_parser.add_argument('--max_N', type=int, default=10000, help='Maximal population size')
arg_parser.add_argument('--in_p', type=float, default=0.2, help='Initial probability of the innovation')


def run_experiment(parameters: Dict, mechanism_name: str,
                   exp_name_suffix: str, experiment_id: int,
                   out_dir: str) -> None:

    """
    Run a single experiment.
    """
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
                    out_dir: str, experiment_id: int, sims: int) -> None:

    """
    Run experiments for a specific set of parameters according to
    a certain mechanism of language change. Save the output data.

    Parameters:
    -----------
    mechanism_name:     Mechanism of language change: neutral_change, replicator_selection, or interactor_selection
    min_N:              Minimal population size
    max_N:              Maximal population size
    k:                  Number of neighbours
    initial_p:          Initial probability of the innovation
    p_range:            Range of rewiring probability values
    s_range:            Range of selection strength/pressure values
    n_range:            Range of proportions of leaders
    out_dir:            Output directory
    experiment_id:      ID of the experiment
    sims:               Number of simulation runs
    """

    common_parameters = {
        'agents': ap.IntRange(min_N, max_N),
        'lingueme': ('A', 'B'),
        'memory_size': 10,
        'initial_frequency': initial_p,
        'number_of_neighbors': k,
        'interactor_selection': False,
        'replicator_selection': False,
        'neutral_change': False,
        'steps': sims
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
    min_N = args.min_N
    max_N = args.max_N
    in_p = args.in_p

    run_experiments(mechanism_name=mechanism,
                    min_N=min_N, max_N=max_N,
                    initial_p=in_p, k=4,
                    p_range=[0, 0.01, 1],
                    s_range=np.arange(0.1, 1.1, 0.1),
                    n_range=[0.1, 0.2],
                    experiment_id=exp_id,
                    out_dir=output_dir,
                    sims=simulations)
