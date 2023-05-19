# Reinforcement Learning 2nd Edition - Notes and Codes

*Reinforcement Learning - An Introduction, 2nd Edition*, written by Richard S. Sutton and Andrew G. Barto, is kind of bible of reinforcement learning. It is a required reading for students and researchers to get the appropriate context of the keep developing field of RL and AI. <br/>

Links to get or rent a hardcover or ebook: [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning/), [Amazon](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262039249/ref=sr_1_4?keywords=reinforcement+learning+an+introduction&qid=1684469272&sprefix=reinforcement+le%2Caps%2C82&sr=8-4&ufe=app_do%3Aamzn1.fos.006c50ae-5d4c-4777-9bc0-4513d670b6bc) (Paperback version if generally not recommended because the poor printing quality).<br/>

### **Motivation of this project:**
Although the authors have made the book extremely clear and friendly to readers at each level, this book is honestly still intimidating to RL or ML beginners because of the intense concepts, abstract examples and algorithms, and its volume. Therefore, as an RL researcher, I'm trying to extract key points and implement examples as well as exercises in the book to help more people better understand the valuable knowledge the book generously provides.<br/>
<br/>
My work mainly consists of:<br/>
- Turning examples into code and plots that are as close to that of in the book as possible;
- Implementing algorithms in `Python` and testing them with RL playground packages like [`Gymnasium`](https://gymnasium.farama.org/);
- Take notes and organize them as PDF files per chapter.
<br/>

## **Snapshot of chapters**:
---
### **Chapter 2: Multi-armed Bandits** &nbsp; &nbsp; :link: [link](/chapter_02_k_armed_bandits/)
 This chapter starts with bandit algorithm and introduces strategies like $\varepsilon$-greedy, Upper-Confidence-Bound, and Gradient Bandit to improve the the algorithm's performance.
- A k-armed bandit testbed:
<p align='center'>
    <img src='./chapter_02_k_armed_bandits/plots/example_2_1.png' width=65%>
</p>

- Parameter study (algorithm comparison) - stationary environment
<p align='center'>
    <img src='./chapter_02_k_armed_bandits/plots/example_2_6_summary.png' width=65%>
</p>

---
### **Chapter 3: Finite Markov Decision Process** &nbsp; &nbsp; :link: [link](/chapter_03_finite_MDP/)
This chapter introduces the fundamentals of the Markov Decision Process in finite states like agent-environment interaction, goals and rewards, returns and episodes, and policy and value function. It helps to build up a basic understanding of the components of reinforcement learning.
- Optimal solution to the `gridworld` example:
<div align='center'>
    <img src='./chapter_03_finite_MDP/plots/example_3_8/optimal_values.png' width=35%>
    <img src='./chapter_03_finite_MDP/plots/example_3_8/optimal_actions.png' width=35%>
</div>

---

### **Chapter 4: Dynamic Programming**
 *In Progress*