# **Chapter 6 Temporal-Difference Learning** &nbsp; &nbsp; :link: [Notes](./[NOTES]CH_6.pdf)

## Examples

### 6.2 Random Walk (*p.125*)

Empirically compare the prediction abilities of TD(0) and
constant-$\alpha$ MC when applied to the Random Walk environment; The left graph below shows the values learned after various numbers of episodes on a single run of TD(0), The right graph shows learning curves for the two methods for various values of $\alpha$. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/example_6_2_random_walk.py)


<p align="center">
    <img src='./plots/example_6_2/value_approx.png', width=34.5%>
    <img src='./plots/example_6_2/rms_compare.png', width=51.7%>
</p>

### 6.3 Random Walk under batch updating (*p.126*)

Batch-updating versions of TD(0) and constant- $\alpha$ MC were applied as follows to the random walk prediction example
(Example 6.2). After each new episode, all episodes seen so far were treated as a batch. They were repeatedly presented to the algorithm, either TD(0) or constant- $\alpha$ MC, with $\alpha$ sufficiently small that the value function converged. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/example_6_3_batch_updating.py)

<p align="center">
    <img src='./plots/example_6_3.png', width=75%>
</p>

### 6.5 Windy Gridworld (*p.130*)

A standard gridworld with start goal states, and a crosswind running upward through the middle of the grid. The actions are the standard four — *up, down, right, and left* — in the middle region the resultant next states are shifted upward by a “wind,” the strength of which varies from column to column. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/example_6_5_windy_gridworld.py)

- Train records:
<p align="center">
    <img src='./plots/example_6_5/step_episodes.png', width=45%>
    <img src='./plots/example_6_5/rewards.png', width=45%>
</p>

- Result:

<p align="center">
    <img src='./plots/example_6_5/result.gif', width=60%>
</p>

### 6.6 Cliff Walking
This gridworld example compares Sarsa and Q-learning, highlighting the difference between on-policy (Sarsa) and off-policy (Q-learning) methods. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions
causing movement up, down, right, and left. Reward is −1 on all transitions except those into the region marked “The Cliff.” Stepping
into this region incurs a reward of −100 and sends the agent instantly back to the start. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/example_6_6_clif_walking.py)

- The reward record of Q-learning and SARSA:
<p align="center">
    <img src='./plots/example_6_6/rewards.png', width=75%>
</p>

- Result: **SARSA**

<p align="center">
    <img src='./plots/example_6_6/SARSA.gif', width=60%>
</p>

- Result: **Q-learning**

<p align="center">
    <img src='./plots/example_6_6/Q_learning.gif', width=60%>
</p>

### Figure 6.3 Performance of TD methods on Cliff Walking

Interim and asymptonic performance of TD control methods on the cliff-walking task as a function of $\alpha$. All algorithms used an $\varepsilon$-greedy policy with $\varepsilon = 0.1$. Asymptotic performance is an average over 100,000 episodes whereas interim performance is an average over the first 100 episodes. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/figure_6_3_TD_methods_performance.py)

<p align="center">
    <img src='./plots/figure_6_3.png', width=75%>
</p>

## Exercise

### 6.9 Windy Gridworld with King's Move (*p.131*)

Re-solve the windy gridworld assuming eight possible actions, including the diagonal moves, rather than four. Can also include the ninth action that causes no movement at all other than that caused by wind. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/exercise_6_9_windy_king_move.py)

- Train records:
<p align="center">
    <img src='./plots/exercise_6_9/step_episodes.png', width=45%>
    <img src='./plots/exercise_6_9/rewards.png', width=45%>
</p>

- Result:

<p align="center">
    <img src='./plots/exercise_6_9/result.gif', width=60%>
</p>


### 6.10 Windy Gridworld with Stochastic Wind (*p.131*)

Re-solve the windy gridworld with King's move, assuming that the effect of the wind, if there's any, is stochastic, sometimes varying by 1 from the mean values given for each column. [Code](https://github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes/blob/main/chapter_06_temporal_difference_learning/exercise_6_10_stochastic_wind.py)

- Train records:
<p align="center">
    <img src='./plots/exercise_6_10/step_episodes.png', width=45%>
    <img src='./plots/exercise_6_10/rewards.png', width=45%>
</p>

- Result:

<p align="center">
    <img src='./plots/exercise_6_10/result.gif', width=60%>
</p>