import matplotlib.pyplot as plt
from typing import Type, Any

"""
Helper functions
"""


def selection(relative_freq_innovation: float, b: float) -> float:
    """
    Generate positive replicator selection
    :param b: selection pressure
    :param relative_freq_innovation: relative frequency of using innovation
    :return: updated relative frequency of using innovation
    """
    updated_relative_freq_innovation = relative_freq_innovation

    if 0 <= relative_freq_innovation <= 1 / (1 + b):
        updated_relative_freq_innovation = (1 + b) * relative_freq_innovation
    if 1 / (1 + b) <= relative_freq_innovation <= 1:
        updated_relative_freq_innovation = 1

    return updated_relative_freq_innovation


def batch_simulate(num_sim: int, model: Type[Any], params: dict) -> plt.Figure:
    """
    Run the model through many iterations and plot the results
    :param num_sim: number of iterations
    :param model: model
    :param params: parameters
    :return: plot
    """

    fig, ax = plt.subplots()

    for i in range(1, num_sim+1):
        print(f'Simulation number: {i}')
        model_i = model(params)
        results_i = model_i.run()
        data = results_i.variables.LangChangeModel
        data['average_x'].plot(linewidth=0.8)

    plt.ylim((0, 1))
    plt.xlabel('t')

    text = (f'agents = {params["agents"]}\n'
                         f'neighbors = {params["number_of_neighbors"]}\n'
                         f'network = {params["network_density"]}\n'
                         f'steps = {params["steps"]}\n'
                         f'simulations = {num_sim}')

    ax.annotate(text, xy=(1, 1), xytext=(-100, -15), fontsize=8,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='left', verticalalignment='top')

    plt.show()
