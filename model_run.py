from base import LangChangeModel
import agentpy as ap

# Parameters setup
parameters = {'agents': ap.IntRange(10, 10000),
              'lingueme': ('A', 'B'),
              'memory_size': 10,
              'initial_frequency': 0.2,
              'number_of_neighbors': 4,
              'rewiring_probability': 0.01,
              'interactor_selection': False,
              'replicator_selection': False,
              'neutral_change': True,
              'selection_pressure': 0.8,
              'n': 2,
              'steps': 100000
              }

# Perform experiment by running the model
# for multiple iterations and parameter combinations
sample = ap.Sample(parameters=parameters, n=40)
exp = ap.Experiment(LangChangeModel, sample=sample, iterations=3, record=True)
exp_results = exp.run(n_jobs=-1, verbose=10)
exp_results.save(exp_name='Neutral_change', exp_id=2, path="output")
