import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv("ap_output/LangChangeModel_2/variables_LangChangeModel.csv")
shape = data.shape
params = pd.read_csv("ap_output/LangChangeModel_2/parameters_sample.csv")
params_sample = data.sample_id.unique().tolist()
iterations = data.iteration.unique().tolist()


def visualize(parameters_sample_i):
    fig, axes = plt.subplots()

    sample_i = data[data.sample_id == parameters_sample_i]
    for j in iterations:
        sample_j = sample_i[sample_i.iteration == j]
        sample_j.plot(x='t', y='average_belief', linewidth=0.5, ax=axes)

    plt.ylim((0, 1))
    plt.xlabel('Interactions')
    plt.show()


for i in params_sample:
    visualize(parameters_sample_i=i)
