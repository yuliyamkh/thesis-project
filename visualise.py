import argparse
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from labellines import labelLines

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--directory', help='Directory path where simulation results are stored')
arg_parser.add_argument('--mechanism', help='Mechanism name')


def plot_data(data: DataFrame, rewiring_probabilities: list, selection_pressures=None,
              n_values=None, mechanism="neutral_change") -> None:
    """
    Plot the proportion of innovation A (y-axis) against population size (x-axis)
    for each social network type, represented by different rewiring probabilities.

    Parameters:
    -----------
     data:                        A DataFrame containing experiment results for a specific mechanism.
    rewiring_probabilities:      A list of rewiring probabilities.
    selection_pressures:         An optional list of selection pressures.
    n_values:                    An optional list of proportions of leaders.
    """

    max_N = data['population_size'].max()
    min_N = data['population_size'].min()

    fig, axs = plt.subplots(1, len(rewiring_probabilities))

    if mechanism == 'interactor_selection':
        for i, rewiring_probability in enumerate(rewiring_probabilities):
            ax = axs[i]
            data_is = data[data['rewiring_probability'] == rewiring_probability]
            for j, n_value in enumerate(n_values):
                n_data = data_is[data_is['n'] == n_value]
                for selection_pressure in selection_pressures:
                    selection_data = n_data[n_data['selection_pressure'] == selection_pressure]
                    selection_data.plot(x='population_size', y='final_A', ax=ax)

                    ax.set_ylabel('L', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
                    ax.set_xlabel('N', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
                    ax.set_xlim([min_N, max_N])
                    ax.set_ylim([0, 1])

                    if rewiring_probability == 0.00:
                        ax.set_title('Regular Network', fontsize=12)
                    elif rewiring_probability == 0.01:
                        ax.set_title('Small-world Network', fontsize=12)
                    elif rewiring_probability == 1:
                        ax.set_title('Random Network', fontsize=12)
                    else:
                        ax.set_title(f'Rewiring Probability {rewiring_probability}', fontsize=12)

                    ax.legend_ = None

    elif mechanism == 'replicator_selection':
        for ax, rewiring_probability in zip(axs, rewiring_probabilities):
            data_rs = data[data['rewiring_probability'] == rewiring_probability]
            for selection_pressure in selection_pressures:
                selection_data = data_rs[data_rs['selection_pressure'] == selection_pressure]
                selection_data.plot(x='population_size', y='final_A', ax=ax, label=str(round(selection_pressure, 1)))

                ax.set_ylabel('L', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
                ax.set_xlabel('N', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
                ax.set_xlim([min_N, max_N])
                ax.set_ylim([0, 1])

                if rewiring_probability == 0.00:
                    ax.set_title('Regular Network', fontsize=12)
                elif rewiring_probability == 0.01:
                    ax.set_title('Small-world Network', fontsize=12)
                elif rewiring_probability == 1:
                    ax.set_title('Random Network', fontsize=12)
                else:
                    ax.set_title(f'Rewiring Probability {rewiring_probability}', fontsize=12)

            labelLines(ax.get_lines(), zorder=2.5, fontsize=6)
            ax.legend_ = None

    else:
        for ax, rewiring_probability in zip(axs, rewiring_probabilities):
            data_ng = data[data['rewiring_probability'] == rewiring_probability]
            data_ng.plot(x='population_size', y='final_A', ax=ax)

            ax.set_ylabel('L', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
            ax.set_xlabel('N', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
            ax.set_xlim([min_N, max_N])
            ax.set_ylim([0, 1])

            if rewiring_probability == 0.00:
                ax.set_title('Regular Network', fontsize=12)
            elif rewiring_probability == 0.01:
                ax.set_title('Small-world Network', fontsize=12)
            elif rewiring_probability == 1:
                ax.set_title('Random Network', fontsize=12)
            else:
                ax.set_title(f'Rewiring Probability {rewiring_probability}', fontsize=12)

            ax.legend_ = None

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = arg_parser.parse_args()
    directory = args.directory
    mech = args.mechanism

    df = pd.read_csv(directory)

    dat = None
    if mech == 'neutral_change':
        dat = df.groupby(['rewiring_probability', 'population_size'])['final_A'].mean().reset_index()
        plot_data(mechanism=mech, data=dat, rewiring_probabilities=df['rewiring_probability'].unique())

    if mech == 'replicator_selection':
        dat = df.groupby(['rewiring_probability', 'selection_pressure', 'population_size'])[
            'final_A'].mean().reset_index()
        plot_data(mechanism=mech, data=dat,
                  rewiring_probabilities=df['rewiring_probability'].unique(),
                  selection_pressures=df['selection_pressure'].unique())

    if mech == 'interactor_selection':
        dat = df.groupby(['rewiring_probability', 'selection_pressure', 'n', 'population_size'])[
            'final_A'].mean().reset_index()
        plot_data(mechanism=mech, data=dat,
                  rewiring_probabilities=df['rewiring_probability'].unique(),
                  selection_pressures=df['selection_pressure'].unique(),
                  n_values=df['n'].unique())
