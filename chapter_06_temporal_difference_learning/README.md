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

A standard gridworld with start goal states, and a crosswind running upward through the middle of the grid. The actions are the standard four — *up, down, right, and left* — in the middle region the resultant next states are shifted upward by a “wind,” the strength of which varies from column to column. [Code](/example_6_5_windy_gridworld.py)

- Train records:
<p align="center">
    <img src='./plots/example_6_5/step_episodes.png', width=45%>
    <img src='./plots/example_6_5/rewards.png', width=45%>
</p>

- Result:

<p align="center">
    <img src='./plots/example_6_5/result.gif', width=60%>
</p>