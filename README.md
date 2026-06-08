# A working minimal version of diversity

The simplest setup I could find that produces behavioral diversity in RL. One agent, a small gridworld, 5 skills, where each skill learns a distinct action. The point was to understand what actually drives diversity before building anything bigger.

The short version: a total variation objective between skill policies works as a differentiable loss, not as a reward. PPO can't optimize it through the advantage channel because it doesn't depend on the sampled action, but it's differentiable with respect to the policy, so it can be optimized directly by backprop.


# Installation

```bash
git clone https://github.com/BoxingBytes/PufferLib.git 
cd Pufferlib
git checkout blog-diversity-2026-06-09

pip install -e . --no-build-isolation
python setup.py build_squared --inplace --force

puffer train puffer_squared # trains the environment
puffer eval --load-model-path latest # runs the environment with a specific skill. 
```

You can change the skill ID in the *pufferlib/pufferl.py* file at (line 1139):
```python
skill = skills[<skill_id>].unsqueeze(0).expand(num_agents, -1) # choose skill_id between 0 and 4
```

<video width="100%" preload="auto" muted controls playsinline autoplay>
  <source src="{{ 'v0_skill1.mp4' | relative_url }}" type="video/mp4">
</video>


# Pufferlib
![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pufferlib)
![Github Actions](https://github.com/PufferAI/PufferLib/actions/workflows/install.yml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

PufferLib is the reinforcement learning library I wish existed during my PhD. It started as a compatibility layer to make working with complex environments a breeze. Now, it's a high-performance toolkit for research and industry with optimized parallel simulation, environments that run and train at 1M+ steps/second, and tons of quality of life improvements for practitioners. All our tools are free and open source. We also offer priority service for companies, startups, and labs!

![Trailer](https://github.com/PufferAI/puffer.ai/blob/main/docs/assets/puffer_2.gif?raw=true)

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues. We're always looking for new contributors, too!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
