# Airstriker-genesis
In this project, we will use a Gym Environment run in Python 3.6 to apply reinforcement learning to the arcade game Airstriker-Genesis. The Gym Environment by Open AI converts Atari games into reinforcement learning environments, allowing us to apply and test different algorithms. The objective of this game is to avoid enemy spaceships, enemy attacks, and other foreign objects (asteroids) in the state space, while scoring points by destroying enemy spaceships. Attacks can only be dodged, not repelled.

Motivation: 
The purpose of this project is to develop a reinforcement learning agent to win the game (success is measured by specific performance parameters such as level completion and maximum score) by evaluating and enhancing existing reinforcement learning algorithms. 

Method: 
The project is an accumulated testing of different reinforcement learning methods, with the final code and results implementing Deep Q-network (DQN). We start by creating a random agent as the baseline for later comparison between the methods and to check that our environment is working properly. The first reinforcement learning agent we created is called ‘the Brute’3, because, like the name suggests, this algorithm only builds up a sequence of button presses that perform well in the game and outputs the best reward, without taking in any visual inputs from the screen.

Then, we attempted to implement DQN, which is composed of two neural networks: the target network and the online network. During the network's learning phase, these q-values will be used as identifiers to optimize the network. 

We need a preprocessing phase for Airstriker-Genesis to help the agent take input and learn. In this phase, we 1) change each frame of the screen from RGB to grayscale values for faster processing, and 2) stack frames to allow the agent to learn 4 frames at a time, instead of 1 (because we noticed that 1 frame does not provide enough information for the agent to make decisions).

The agent also saves a model every 20 episodes for experience replay.

Experiments: 
After evaluating the agent’s performance over the course of a couple of weeks, we have concluded that the main setback for our agent is the catastrophic forgetting problem. We decreased the learning rate a bit so that less recent episodes will not be forgotten, and had to retrain our model. As of the moment, this is our best solution, but in the future, we want to find algorithms that would solve this issue without having to retrain the model from scratch.

Results:
The agent’s evaluation still ran into issues but we had more promising prospects. The agent was able to beat level 1 of the game after the first overnight extensive training but was never able to advance any further despite our efforts. Our epsilon value went down to around 9.5% (meaning that the agent was barely exploring anymore and mainly just going for the greedy actions) at the time of writing this report and it seems like it was still not enough to allow the agent to beat level 2.  

Contributions: 
Can Ha An wrote report, researched on DQN, tested and troubleshooted agent, adjusted agent’s parameters
Le Gia Duc preprocessed game for agent (changed frame from RGB to grayscale values, stacked frames), implemented DQN algorithm, adjusted agent’s parameters)
