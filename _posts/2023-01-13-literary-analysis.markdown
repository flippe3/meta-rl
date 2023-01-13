---
layout: post
title:  "Literary Analysis"
date:   2023-01-13 16:39:12 +0100
categories: Meta-RL, Meta-Learning, Reinforcement Learning, learning to learn
---
A collection of all the papers I've been studying is [here](https://docs.google.com/spreadsheets/d/17ufJaHLAPp0CureI5rUQsRy7NVUZJ8_jqep1uzExxQA/edit#gid=0). In this post I've only written short summaries of the papers I found most interesting.

## Meta-Learning
### [Model Agnostic Meta Learning for Fast Adaption of Deep Networks](https://arxiv.org/abs/1703.03400)
This paper introduces a very clear meta-learning algorithm called MAML which uses gradient decent to
optimize a meta-learner for the network to be able to adapt faster to out of distribution(ood) data. It
is very general and works on any architecture given that it is trained using gradient descent. This also means
that MAML can be used for different problems such as supervised regression and classification but also reinforcement
learning. This paper serves as a great starting point for meta-learning.

Algorithms: Gradient descent, REINFORCE
Datasets: Omniglot, MiniImagenet, MuJoCo simulator (half-cheetah and ant locomotion), 2D goal positions (no dataset).
Code: [MAML](https://github.com/cbfinn/maml), [MAML\_RL](https://github.com/cbfinn/maml_rl)

## Reinforcement Learning

## Meta Reinforcement Learning
