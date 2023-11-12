# Thesis Project for Master of Arts in General Linguistics
***Exploring the Effects of Population Size on Language Change: An Integrated Approach using Agent-Based Modeling and Statistical Analysis***

**Seminar für Sprachwissenschaft | Universität Tübingen**

**Abstract**

The effect of population size on language change has been the subject of a series of linguistic studies. However, the results of the studies vary, with some claiming that population size is a significant factor influencing language change, and others claiming that it is not. In this thesis, the relationship between population size and language change is investigated using a hybrid approach which integrates agent-based modeling and statistical analysis. The implemented agent-based model is a multi-speaker Moran model, and it is used to simulate language change under different conditions. The conditions are represented by the different mechanisms of language change: neutral change (drift), replicator selection, and interactor selection. The data generated by the multi-speaker Moran model are quantitatively analysed using multiple linear regression. The results of the statistical analysis show that population size significantly effects language change if selection is involved. 

## The multi-speaker Moran model
### Usage / Examples

To run first experiments with [the multi-speaker Moran model](https://github.com/yuliyamkh/ABM_AgentPy/blob/master/model.py), make sure to install the packages listed in [requirements.txt](https://github.com/yuliyamkh/ABM_AgentPy/blob/master/requirements.txt). After that, you can perform the following step: ```python model.py```. Consider that this command runs the default model with the following parameter setup:

| Argument          | Description                            | Default        |
|-------------------|----------------------------------------|----------------|
| ```--ps```        | Number of agents                       | 10             |
| ```--ifs```       | Initial innovation frequency           | 0.2            |
| ```--n_ns```      | Average number of neighbours per agent | 4              |
| ```--rp```        | Rewiring probability                   | 0              |
| ```--sp```        | Selection pressure/strength            | 0.1            |
| ```--nls```       | Proportion of leaders                  | 0.1            |
| ```--sim_steps``` | Number of simulation steps             | 100000         |
| ```--exp_num```   | Number of experiments                  | 5              |
| ```--mech```      | Mechanism name                         | neutral_change |



