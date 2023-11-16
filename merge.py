import argparse
import pandas as pd
import os

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--source', default='output_data', help='Source directory where simulation results are stored')
arg_parser.add_argument('--mechanism', help='Name of the mechanism')
arg_parser.add_argument('--out_dir', default='final_output', help='Output directory')


def merge(mechanism_name: str, dir_path: str, directories: list, out_dir_path: str) -> None:
    """
    Merge multiple outputs of experiments for a specific
    mechanism of language change into one output file

    Parameters:
    -----------
    mechanism_name:     Name of the mechanism: neutral_change, replicator_selection, or interactor_selection
    dir_path:           Directory name where the output of the experiments is stored
    directories:        All directories stored in out_dir
    out_dir:            The path to the directory where the output should be saved
    """

    dfs = []
    for directory_name in directories:
        if directory_name.startswith(mechanism_name):
            path = os.path.join(dir_path, directory_name)
            filepath_params = os.path.join(path, "parameters_constants.json")
            filepath_reporters = os.path.join(path, "reporters.csv")
            filepath_params_sample = os.path.join(path, "parameters_sample.csv")

            # Read parameters constants
            params = pd.read_json(filepath_params)
            rewiring_probability = params.rewiring_probability.unique()[0]

            # Read parameters sample
            params_sample = pd.read_csv(filepath_params_sample)
            params_sample = params_sample.to_dict()
            population = params_sample['agents']

            # Read and extend reporters
            reporters = pd.read_csv(filepath_reporters)
            population_size = [population[sample_id] for sample_id in reporters.sample_id]

            reporters['population_size'] = population_size
            reporters['rewiring_probability'] = [rewiring_probability] * len(population_size)
            reporters['replicator_selection'] = [mechanism_name == 'replicator_selection'] * len(population_size)
            reporters['interactor_selection'] = [mechanism_name == 'interactor_selection'] * len(population_size)
            reporters['neutral_change'] = [mechanism_name == 'neutral_change'] * len(population_size)

            if mechanism_name == 'replicator_selection' or mechanism_name == 'interactor_selection':
                selection_pressure = params.selection_pressure.unique()[0]
                reporters['selection_pressure'] = [selection_pressure] * len(population_size)
                reporters = reporters.drop(reporters[reporters['selection_pressure'] == 0].index)

            if mechanism_name == 'interactor_selection':
                n = params.n.unique()[0]
                reporters['n'] = [n] * len(population_size)

            dfs.append(reporters)

    result = pd.concat(dfs, ignore_index=True)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # Save the DataFrame as a CSV file in the specified directory
    result.to_csv(os.path.join(out_dir_path, f'{mechanism_name}_output_test.csv'))
    print(f'Data saved to {out_dir_path}/{mechanism_name}_output_test.csv')


if __name__ == '__main__':
    args = arg_parser.parse_args()
    source = args.source
    dirs = os.listdir(source)
    out_dir = args.out_dir
    mechanism = args.mechanism
    merge(mechanism_name=mechanism,
          dir_path=source,
          directories=dirs,
          out_dir_path=out_dir)
