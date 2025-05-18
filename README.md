# Self-Learning-Snake-Game

# How to Run Program
Type into command line "pip install -r requirements.txt", this should install all necessary packages. Then type "python ai_snakegame.py" this will run the program. To exit, simply clost the window of the game. Enjoy!


# Project Proposal
A project for SysEng5211, the goal of this game is for a snake to get the highest score possible by eating an apple. The snake is trained with an agent uses Deep Q learning. The purpose of this game is to show how Deep Q Learning (DQL) can be used to successfully learn in an challenging enviroment. The agent has "danger" awareness, current direction, and food location, it uses these to learn the optimal moves with its neural network, experience replay, and episolon greedy exploration. The project is meant to show deep learning fundamentals in a gamified way, allowing the user to watch as the snake learns how to collect the most apples and get the highest score!


# What is Q-Learning?
When using QL a Q-table is developed. The table consist of state-action pairs and corresponds to the state (S) and Action (A) within the table. R is the reward function, t is the current time step, and alpha (α) and gamma (γ) represent learning rate and discount factor respectively. This formula approximates the optimal action-value function.

# Why Use Q-Learning?
QL has the benefit of free modeling, meaning the agent knows all possibles states and actions discovers the state transitions and rewards by “exploring” its environment. As this occurs the agent is also using temporal difference, which means it reevaluates its prediction after taking a step. Essentially making no need for a final outcome as the last action is interpreted for the next step.  

# Epsilon-Greedy Algorithm in Q-Learning
A Q-table is created with arbitrary values except the terminal states, which are set to zero.  After the initialization, the agent begins to step,  with each step an action (A)  is chosen from the Q-table (Q). As an action is taken, a reward is given and the state changes from S to S’. These values are used to update the Q-table, and this continues until a terminal state is reached.

# Deep Q-Learning
Deep Q learning (DQL) is an algorithm that is a part of the reinforcement learning domain. This algorithm is uses the processes of deep learning and reinforcement learning to train an agent to make decisions in complex environments.  
An agent is an entity that makes decisions while interacting with its environment to maximize a cumulative reward.  The feedback given to the agent is what determines the “next step” in its decision making process.
Two fundamental principles of DQL are the discount factor and the balance between exploration and exploitation of the agent.

# Deep Learning’s Role in Q-Learning
Since traditional QL has limitations, the use of neural networks allowed for the use of Deep Q networks (DQN). These can approximate the Q-value function with neural networks parameterized by weights instead of maintaining QL tables.  The network is able to take in a state as an input and output Q-values for all possible actions. This makes the DQN more scalable, able to take in continuous input data, and handle high dimensional state spaces.

# Architecture of Deep Q-Networks
The traditional DQN has 3 main parts, 
1.  the neural network,
2. experience relay, and
3. target network.
   
1. The network approximates the Q-value function Q(S, A; θ), where θ is the trainable parameters.
2. For training to be stable, the DQN stores previous experiences (s, a, r, s`) in a replay buffer. As training is happening small samples are randomly taken from the buffer and used to improve generalization.
3. A separate target network (θ-) is used to maintain stability with the DQN by periodically having the target weights updated with the main network weights.
   
The loss function measures the target and predicted Q-values. 

# How DQN is used to Train Agents

1. Initialize the replay buffer along with the target (θ-) and main (θ) network.  Also set the hyperparameters, such as learning rate, discount factor, and exploration rate.
2. The epsilon greedy algorithm is then used for the exploration and exploitation for the environment.
3. The DQN should begin to collect “experiences” within the environment, that will stored in the replay buffer.
4. Use the stored experiences to compute target Q-values using the target network, then update the main network by minimizing loss function.
5. Periodically update target network weights with the main networks to continue stability.
6. Gradually decrease exploration over time to exploitation.

# Snake Game

The purpose of the game is for a snake to eat as much food as possible, as the snake eats food, it grows by one block, while playing the game the snake is not allowed to run into itself or the edge of its environment if it does, the game ends. I am using Pygame and Pytorch to implement the game. 

I have considered the following:

Action: [1, 0, 0] is straight, 
[0, 1, 0] is turn right, and 
[0, 0, 1] is turn left.

The Rewards are labeled as:  
Eat food: +10, Game Over: -10, Else: 0

There are 11 total values for the state: danger straight, right, left (3), direction left, right, up, down (4), food left, right, up, down (4).  
They are kept as a boolean value for example: 

[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]

This means there is no danger, the snake is moving to the right, and there is food to right and down.

# DQN Implementation

For the model the inputs need to be able to take in 11 values along with a hidden layer in between, and output 3 values (action).  
Basic Rundown of DQN Process:
-Initialize Q value
-Choose an action (model predicts a more or a random move is chosen)
-Perform the action
-Measure Reward
-Update Q value (train model)
Repeat steps 1-4

