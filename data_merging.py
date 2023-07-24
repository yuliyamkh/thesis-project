import pandas as pd
import os

parent_directory = "C:/Users/avror/OneDrive/Dokumente/ABM/output_data/output"
directories_names = os.listdir(parent_directory)


def merge(mechanism_name: str):
    """
    Merge multiple outputs into one dataframe
    :param mechanism_name: name of the mechanism
    :return: dataframe of the .csv format
    """

    dfs = []
    for directory_name in directories_names:
        if directory_name.startswith(mechanism_name):
            path = os.path.join(parent_directory, directory_name)
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
            reporters['interactor_selection'] = [False] * len(population_size)
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
    if not os.path.exists('output'):
        os.makedirs('output')

    # Save the DataFrame as a CSV and Excel files in the specified directory
    result.to_csv(os.path.join('output', f'{mechanism_name}_output.csv'))
    result.to_excel(os.path.join('output', f'{mechanism_name}_output.xlsx'))


merge(mechanism_name='neutral_change')
