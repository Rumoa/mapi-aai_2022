# Readme
These folders contain the source code to reproduce the results that appear in the submitted report.
In this file we show how to run them. These files are prototypes and (may) contain bugs.


## Minimax

The folder python-ataxx contains the cloned repository https://github.com/kz04px/python-ataxx  as well as files to produce the 
results. The file from the original repository that has been modified is "python-ataxx/ataxx/players.py" to fix bugs in the
alpha beta and negamax functions as well as providing more functionality like the mobility evaluation function. 
The code that has been modified will include a header as well as comments.

The code to reproduce the results are the files:
- alpha_vs_alpha.py
- min_vs_min.py
- nega_vs_nega.py
- players_vs_players.py

These files contain the possible cases that has been tested and run them. The results are saved in a .job file although
the information printed in the console can be also used to store the information.

The files inside the results folder contains the data generated for the report.
The file behavior.py contains the probability distribution of depth corresponding to the difficulty levels.



## Reinforcement-learning

Inside RL there are three folders: gym-env, stable-baseline3 and tf-agents . We use a custom environment of openai gym that we have to manually install it 
in our python environment.


## Custom openai gym environment

Installation instructions:
- We recommend to create a new python environment, using pyenv, conda, etc. We have used conda.
- Using our favourite python package manager, install openaigym.
- Locate the file "\_\_init\_\_.py" of the gym library. In our case, it was located at /home/antonio/miniconda3/envs/TFgpu/lib/python3.9/site-packages/gym/envs/reversi/\_\_init\_\_.py , being TFgpu the name of our conda environment
- Append the content of the file "append_to\_\_init\_\_.py" in the end of the \_\_init\_\_.py file. This register our custom environment in the gym package.
- Paste the folder called ataxx inside the folder of \_\_init\_\_.py . In our case: /home/antonio/miniconda3/envs/TFgpu/lib/python3.9/site-packages/gym/envs/ 

You should be able now to use the custom gym environment.


The \_\_init\_\_.py file contains the initialization parameters of the environment. Please read the comments to see the possible values.

The ataxx.py file is the modified original file from reversi environment as we state in the report. Some of the comments are of our authorship about the modifications while others are the original ones.
Take into account that this is a prototype, so there are old names in the class from the Reversi game. 

There must be more elegant approaches than modifying a \_\_init\_\_.py file as end user, but we didn't have the time to do it.

## Stable baselines3

The file generate_model_ppo.py generates the results using a ppo agent with the implementation from stable baselines3. It also uses callbacks to save the best model achieved.

## tf-agents
The file dqn.py generates results using a dqn network from the library tf-agents. It also saves the best model achieved iteratively.

We have also included the results that were used in the report. 



Antonio Ruiz Molero 2022