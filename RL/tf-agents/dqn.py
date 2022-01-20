import tensorflow as tf
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import PolicySaver
import joblib

import matplotlib.pyplot as plt

import gym
import random
import numpy as np

# Antonio Ruiz Molero 2022
# Most of the code is taken from the book  "Machine Learning TensorFlow Cookbook"
# from Audevart, Bahachewicz and Massaron.

# Compute the average return within a range


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# Create our custom environment.
env = gym.make('Ataxx-v0')


num_iterations = 25000  # @param

initial_collect_steps = 1000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 200000  # @param


batch_size = 128  # @param
learning_rate = 1e-4  # @param
log_interval = 200  # @param

num_eval_episodes = 2  # @param
eval_interval = 1000  # @param

# Prepare our environment ready for tf-agents
env_name = 'Ataxx-v0'
env = tf_agents.environments.suite_gym.load(env_name)
env = tf_agents.environments.tf_py_environment.TFPyEnvironment(env)

# Separate them in training and evaluation environment.
train_env = env
eval_env = env

# Creation of the q net where we specify the dimension of our action and observation spaces
q_net = tf_agents.networks.q_network.QNetwork(
    env.observation_spec(), env.action_spec())
train_step_counter = tf.Variable(0)
# Select the Adam optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

# Creation of the tf_agent with the q_network created before
tf_agent = tf_agents.agents.dqn.dqn_agent.DqnAgent(env.time_step_spec(),
                                                   env.action_spec(),
                                                   q_network=q_net,
                                                   optimizer=optimizer,
                                                   td_errors_loss_fn=tf_agents.utils.common.element_wise_squared_loss,
                                                   train_step_counter=train_step_counter)

tf_agent.initialize()


eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

# Initialization of replay buffer.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

print("Batch Size: {}".format(train_env.batch_size))

replay_observer = [replay_buffer.add_batch]
# Metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=1)

iterator = iter(dataset)

print(compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes))

tf_agent.train = common.function(tf_agent.train)
tf_agent.train_step_counter.assign(0)

final_time_step, policy_state = driver.run()

# Initialization of the saver
my_policy = tf_agent.collect_policy
saver = PolicySaver(my_policy, batch_size=None)

for i in range(1000):
    final_time_step, _ = driver.run(final_time_step, policy_state)

episode_len = []
step_len = []
return_list = []
return_step = []
# Start of the training
for i in range(num_iterations):
    # compute a  step
    final_time_step, _ = driver.run(final_time_step, policy_state)

    experience, _ = next(iterator)
    # compute the training loss and train
    train_loss = tf_agent.train(experience=experience)
    # step counter
    step = tf_agent.train_step_counter.numpy()

    # save data if needed or display it in the screen
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        episode_len.append(train_metrics[3].result().numpy())
        step_len.append(step)
        print('Average episode length: {}'.format(
            train_metrics[3].result().numpy()))
    # compute average return
    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, tf_agent.policy, num_eval_episodes)
        return_list.append(avg_return)
        return_step.append(step)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        # If the average return is equal to 1.0, perform a backup
        if avg_return == 1.0:
            saver.save('best/policy_%d' % step)

# Plotting of the length per episodes
plt.plot(step_len, episode_len)
plt.xlabel('Episodes')
plt.ylabel('Average Episode Length (Steps)')
plt.savefig("episodes_5x5.pdf")
plt.show()

# save the avg return in a job file.
joblib.dump([return_step, return_list], "returns.job")
plt.show()
