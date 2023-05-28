# **Chapter 4: Dynamic Programming** &nbsp; &nbsp; :link: [Notes](./%5BNOTES%5DCH_4.pdf)
## **Examples**

### 4.1 Policy Evaluation Gridworld (*p.77*)
Convergence of iterative policy evaluation on a small gridworld. The last policy is guaranteed only to be an improvement over the random policy, but in this case it, and all policies after the third iteration, are optimal.
[Code](./example_4_1_policy_evaluation.py)
<p align="center">
    <img src='./plots/example_4_1.png' width=50%>
</p>

### 4.2 Jack's Car Rental (*p.81*)
**Important:** Before running this example, be sure to first install the Jack's Car Rental environment by:
```bash
cd gym_env
pip install .
```