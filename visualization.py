import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

results = pd.read_csv('output/Interactor_selection_2/variables_LangChangeModel.csv')
parameters_sample = pd.read_csv('output/Interactor_selection_2/parameters_sample.csv')
parameters_sample_dict = parameters_sample.to_dict()['agents']
parameter_constants = pd.read_json('output/Interactor_selection_2/parameters_constants.json')
reporters = pd.read_csv('output/Interactor_selection_2/reporters.csv')

population_sizes = []
for sample_id in reporters.sample_id:
    population_sizes.append(parameters_sample_dict[sample_id])
reporters['population_size'] = population_sizes
reporters.to_csv('reporters/Interactor_selection_2')

parameter_constants = parameter_constants.drop(1)
parameter_constants = parameter_constants.drop(columns='lingueme').to_dict()

# Generate text for plot
text = []
for parameter, value in parameter_constants.items():
    for k, v in value.items():
        string = f'{parameter}: {v}'
        text.append(string)
text = '\n'.join(text)


def draw_text(ax, txt):
    """
    Draw a text-box
    """
    at = AnchoredText(txt,
                      loc='lower left',
                      prop=dict(size=6.5), frameon=True,
                      )
    ax.add_artist(at)


# Create clusters for population size
clusters = [1 if 0 < agents < 100 else 2
            if 100 <= agents < 1000 else 3
            if 1000 <= agents < 10000 else 4
            if 10000 <= agents < 100000 else 5
            for agents in parameters_sample.agents]

# Store clusters and corresponding values in a dictionary
clusters_dict = {1: '1-99', 2: '100-999', 3: '1000-9999', 4: '10000', 5: '100000'}

# Add clusters to results
results['clusters'] = [clusters[sample_id] for sample_id in results.sample_id]

# fig, axes = plt.subplots()

data = results.groupby(['clusters', 'sample_id', 'iteration'])['A']
group_names = [name for name, group in data]

for cluster in results.clusters.unique():
    fig1, axes1 = plt.subplots()
    merged_df = pd.DataFrame()
    for group_name in group_names:
        if group_name[0] == cluster:
            group = data.get_group(group_name)
            group_df = group.to_frame()
            group_df = group_df.rename(columns={'x': f'x_{group_name[2]}'})
            group_df = group_df.reset_index(drop=True)
            merged_df = pd.concat([merged_df, group_df], axis=1)
            plt.plot(merged_df)

    draw_text(ax=axes1, txt=f"Population: {clusters_dict[cluster]}")
    plt.xlabel('time, t', weight='bold')
    plt.ylabel('innovation frequency, x', weight='bold')
    plt.ylim((0, 1))
    plt.xlim(0, 100000)
    plt.show()

    # average = merged_df.mean(axis=1)
    # average.plot(ax=axes, label=f'{clusters_dict[cluster]} agents')

# plt.legend(fontsize=7)
# draw_text(ax=axes, txt=text)
# plt.xlim(0, 1000)
# plt.ylim((0, 1))
# plt.ylabel('innovation frequency, x', weight='bold')
# plt.xlabel('time, t', weight='bold')
# plt.show()
exit()
