# Implementation of the paper "WToE: Learning When to Explore in Multi-Agent Reinforcement Learning"

This is the code for the paper ["WToE: Learning When to Explore in Multi-Agent Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/10324374/).

## Environment
1. Grid Examples: The environment contains two basic grid environment (2-room and 4-room environments), which are implemented in the `GRID/ENV` file.
2. [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs): A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics. Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf). We test our algorithm in six scenarios, including `simple_adversary`,`simple_crypto`, `simple_push`, `simple_reference`, `simple_spread`, `simple_tag`.
3. [MAgent](https://github.com/geek-ai/MAgent): MAgent is a research platform for many-agent reinforcement learning. Unlike previous research platforms that focus on reinforcement learning research with a single agent or only few agents, MAgent aims at supporting reinforcement learning research that scales up from hundreds to millions of agents.

## Quick start
Please follow the instruction of 'README.md' file in different environments to install Python requirements.

## Cite our paper
```
@article{WToE,
  title={WToE: Learning When to Explore in Multi-Agent Reinforcement Learning},
  author={Dong, Shaokang and Mao, Hangyu and Yang, Shangdong and Zhu, Shengyu and Li, Wenbin and Hao, Jianye and Gao, Yang},
  journal={IEEE Transactions on Cybernetics},
  year={2024},
  volume={54},
  number={8},
  pages={4789-4801},
  doi={10.1109/TCYB.2023.3328732}
}
```
