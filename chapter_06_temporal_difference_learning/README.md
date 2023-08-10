# **Chapter 6 Temporal-Difference Learning** &nbsp; &nbsp; :link: [Notes](./[NOTES]CH_6.pdf)

## Examples

### 6.2 Random Walk (*p.125*)

Empirically compare the prediction abilities of TD(0) and
constant-$\alpha$ MC when applied to the Random Walk environment; The left graph below shows the values learned after various numbers of episodes on a single run of TD(0), The right graph shows learning curves for the two methods for various values of $\alpha$. [Code](./example_6_2_random_walk.py)

<p align="center">
    <img src='./plots/example_6_2/value_approx.png', width=34.5%>
    <img src='./plots/example_6_2/rms_compare.png', width=51.7%>
</p>