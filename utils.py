import matplotlib.pyplot as plt
from typing import Type, Any


def batch_simulate(num_exp: int, model: Type[Any], params: dict, mech: str) -> None:
    """
    Run the model through many iterations and plot the results

    Parameters:
    -----------
    num_sim:    Number of simulation runs
    model:      Agent-based model
    params:     A dictionary of the parameters of the model
    mech:       Mechanism name
    """

    fig, ax = plt.subplots()

    for i in range(1, num_exp + 1):
        print(f'Simulation number: {i}')
        model_i = model(params)
        results_i = model_i.run()
        data = results_i.variables.LangChangeModel
        data['A'].plot(linewidth=0.8)

    plt.ylim((0, 1))
    plt.xlabel('time')
    plt.xlim((0, params['steps']))

    text = None
    if mech == 'neutral_change':
        text = (f'agents = {params["agents"]}\n'
                f'neighbors = {params["number_of_neighbors"]}\n'
                f'network = {params["rewiring_probability"]}\n'
                f'steps = {params["steps"]}\n'
                f'simulations = {num_exp}')

    if mech == 'replicator_selection':
        text = (f'agents = {params["agents"]}\n'
                f'neighbors = {params["number_of_neighbors"]}\n'
                f'network = {params["rewiring_probability"]}\n'
                f'selection pressure = {params["selection_pressure"]}\n'
                f'steps = {params["steps"]}\n'
                f'simulations = {num_exp}')

    if mech == 'interactor_selection':
        text = (f'agents = {params["agents"]}\n'
                f'neighbors = {params["number_of_neighbors"]}\n'
                f'network = {params["rewiring_probability"]}\n'
                f'selection pressure = {params["selection_pressure"]}\n'
                f'leaders = {params["n"]}\n'
                f'steps = {params["steps"]}\n'
                f'simulations = {num_exp}')

    ax.annotate(text, xy=(1, 1), xytext=(-120, -15), fontsize=8,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='left', verticalalignment='top')

    title = mech.split('_')
    capitalized_parts = [part.capitalize() for part in title]
    name = ' '.join(capitalized_parts)
    plt.suptitle(name)

    plt.show()
