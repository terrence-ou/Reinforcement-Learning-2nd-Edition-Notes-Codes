# **Chapter 5: Monte Carlo Methods** &nbsp; &nbsp; :link: [Notes](./%5BNOTES%5DCH_5.pdf)

## Examples

### 5.1 Blackjack (*p.94*)

Approximate state-value functions for the blackjack policy that sticks only on 20
or 21, computed by Monte Carlo policy evaluation. [Code](./example_5_1_blackjack.py)
<br>

**Usable ace:**

<p align="center">
    <img src='./plots/example_5_1/10000_episodes_usable_ace.png' width=45%>
    <img src='./plots/example_5_1/500000_episodes_usable_ace.png' width=45%>
</p>

**No usable ace:**

<p align="center">
    <img src='./plots/example_5_1/10000_episodes_no_usable_ace.png' width=45%>
    <img src='./plots/example_5_1/500000_episodes_no_usable_ace.png' width=45%>
</p>

### 5.3 Solving Blackjack (*p.99*)
Apply Monte Carlo ES to blackjack. The initial policy is to stick only on the player's sum is 20 or 21, and the initial action-value function is zero for all state-action pairs. [Code](./example_5_3_solving_blackjack.py)
<br>

**Usable ace:**

<p align="center">
    <img src='./plots/example_5_3/optimal_policy_usable_ace.png' width=45%>
    <img src='./plots/example_5_3/optimal_value_usable_ace.png' width=45%>
</p>

**No usable ace:**

<p align="center">
    <img src='./plots/example_5_3/optimal_policy_no_usable_ace.png' width=45%>
    <img src='./plots/example_5_3/optimal_value_no_usable_ace.png' width=45%>
</p>