import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from labellines import labelLines

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--directory', help='Directory path where simulation results are stored')

df = pd.read_csv(r'C:\Users\avror\PycharmProjects\ABM_AgentPy\output\neutral_change_output.csv')
df2 = pd.read_csv(r"C:\Users\avror\PycharmProjects\ABM_AgentPy\output\replicator_selection_output.csv")
df3 = pd.read_csv(r"C:\Users\avror\PycharmProjects\ABM_AgentPy\output\interactor_selection_output.csv")
data = df.groupby(['rewiring_probability', 'population_size'])['final_A'].mean().reset_index()
data2 = df2.groupby(['rewiring_probability', 'selection_pressure', 'population_size'])['final_A'].mean().reset_index()
data3 = df3.groupby(['rewiring_probability', 'selection_pressure', 'n', 'population_size'])['final_A'].mean().reset_index()

# Create a list of rewiring probabilities
rewiring_probabilities = df['rewiring_probability'].unique()
selection_pressures = df2['selection_pressure'].unique()
n_values = df3['n'].unique()

# Create subplots
fig, axs = plt.subplots(1, len(rewiring_probabilities))
mechanism = 'interactor_selection'
# Loop over each subplot and plot the data
for ax, rewiring_probability in zip(axs, rewiring_probabilities):

    if mechanism == 'neutral_change':
        # Filter data for the current rewiring probability
        data_pl = data[data['rewiring_probability'] == rewiring_probability]
        # Plot data
        data_pl.plot(x='population_size', y='final_A', ax=ax, legend=False)

        # Set title
        ax.set_title(f'Rewiring Probability: {rewiring_probability}')

        # Set labels
        ax.set_ylabel('L', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
        ax.set_xlabel('N', fontdict={'fontsize': 12, 'fontstyle': 'italic'})

        ax.set_xlim([0, 10000])
        ax.set_ylim([0, 1])

        if rewiring_probability == 0.00:
            ax.set_title('Regular Network', fontsize=12)
        if rewiring_probability == 0.01:
            ax.set_title('Small-world Network', fontsize=12)
        if rewiring_probability == 1:
            ax.set_title('Random Network', fontsize=12)

    if mechanism == 'replicator_selection':
        # Filter data for the current rewiring probability
        data_sp = data2[data2['rewiring_probability'] == rewiring_probability]
        # Loop over each selection_pressure and plot on the same axes
        for selection_pressure in selection_pressures:
            # Filter data for the current selection_pressure
            selection_data = data_sp[data_sp['selection_pressure'] == selection_pressure]

            # Plot data with label being the selection_pressure
            selection_data.plot(x='population_size', y='final_A', ax=ax,
                                label=str(round(selection_pressure, 1)))

        # Set labels
        ax.set_ylabel('L', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
        ax.set_xlabel('N', fontdict={'fontsize': 12, 'fontstyle': 'italic'})

        ax.set_xlim([0, 10000])
        ax.set_ylim([0, 1])

        labelLines(ax.get_lines(), zorder=2.5, fontsize=6)
        ax.legend_ = None

        if rewiring_probability == 0.00:
            ax.set_title('Regular Network', fontsize=12)
        if rewiring_probability == 0.01:
            ax.set_title('Small-world Network', fontsize=12)
        if rewiring_probability == 1:
            ax.set_title('Random Network', fontsize=12)

    if mechanism == 'interactor_selection':
        # Loop over each subplot and plot the data
        for i, rewiring_probability in enumerate(rewiring_probabilities):
            ax = axs[i]

            # Filter data for the current rewiring probability
            data_is = data3[data3['rewiring_probability'] == rewiring_probability]

            # Loop over each n value and plot on the same axes
            for j, n_value in enumerate(n_values):
                # Filter data for the current n value
                n_data = data_is[data_is['n'] == n_value]

                # Loop over each selection_pressure and plot on the same axes
                for selection_pressure in selection_pressures:
                    # Filter data for the current selection_pressure
                    selection_data = n_data[n_data['selection_pressure'] == selection_pressure]

                    # Plot data with label being the selection_pressure
                    selection_data.plot(x='population_size', y='final_A', ax=ax)

                    # Set title and legend
                    # Set labels
                    ax.set_ylabel('L', fontdict={'fontsize': 12, 'fontstyle': 'italic'})
                    ax.set_xlabel('N', fontdict={'fontsize': 12, 'fontstyle': 'italic'})

                    ax.set_xlim([0, 10000])
                    ax.set_ylim([0, 1])

                    if rewiring_probability == 0.00:
                        ax.set_title('Regular Network', fontsize=12)
                    if rewiring_probability == 0.01:
                        ax.set_title('Small-world Network', fontsize=12)
                    if rewiring_probability == 1:
                        ax.set_title('Random Network', fontsize=12)

                ax.legend_ = None

# Show plot
plt.tight_layout()
plt.show()
exit()
