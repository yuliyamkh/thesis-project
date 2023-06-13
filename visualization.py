import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

results = pd.read_csv('ap_output/LangChangeModel_12/variables_LangChangeModel.csv')
parameters_sample = pd.read_csv('ap_output/LangChangeModel_12/parameters_sample.csv')
parameters_sample_dict = parameters_sample.to_dict()['agents']
parameter_constants = pd.read_json('ap_output/LangChangeModel_12/parameters_constants.json')
reporters = pd.read_csv('ap_output/LangChangeModel_12/reporters.csv')

population_sizes = []
for sample_id in reporters.sample_id:
    population_sizes.append(parameters_sample_dict[sample_id])
# reporters['population_size'] = population_sizes
# reporters.to_csv('reporters/reporters_LangChangeModel_12')

parameter_constants = parameter_constants.drop(1)
parameter_constants = parameter_constants.drop(columns='lingueme').to_dict()

# Generate text for plot
text = []
for parameter, value in parameter_constants.items():
    for k, v in value.items():
        string = f'{parameter}: {v}'
        text.append(string)
text = '\n'.join(text)


def draw_text(ax):
    """
    Draw a text-box
    """
    at = AnchoredText(text,
                      loc='lower left',
                      prop=dict(size=6.5), frameon=False,
                      )
    ax.add_artist(at)


# Create clusters for population size
clusters = [1 if 0 < agents < 100 else 2
            if 100 <= agents < 1000 else 3
            if 1000 <= agents < 10000 else 4
            for agents in parameters_sample.agents]

# Store clusters and corresponding values in a dictionary
clusters_dict = {1: '1-99', 2: '100-999', 3: '1000-9999', 4: '10000'}

# Add clusters to results
results['clusters'] = [clusters[sample_id] for sample_id in results.sample_id]

fig, axes = plt.subplots()

data = results.groupby(['clusters', 'sample_id', 'iteration'])['x']
group_names = [name for name, group in data]

for cluster in results.clusters.unique():
    merged_df = pd.DataFrame()
    for group_name in group_names:
        if group_name[0] == cluster:
            group = data.get_group(group_name)
            group_df = group.to_frame()
            group_df = group_df.rename(columns={'x': f'x_{group_name[2]}'})
            group_df = group_df.reset_index(drop=True)
            merged_df = pd.concat([merged_df, group_df], axis=1)

    average = merged_df.mean(axis=1)
    average.plot(ax=axes, label=f'{clusters_dict[cluster]} agents')

plt.legend(fontsize=7)
draw_text(ax=axes)
plt.xlim(0, 1000)
plt.ylim((0, 1))
plt.ylabel('innovation frequency, x', weight='bold')
plt.xlabel('time, t', weight='bold')
plt.show()
exit()
