# METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

This is an implementation of the [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://arxiv.org/abs/2310.08887v2) paper using pufferlib. 

To use, git clone & pip install in editable mode: 
```bash
pip install -e .
```

Make sure metra is enabled in conf ```config/default.ini```, then 
```
puffer train puffer_convert_circle
```

The main code is implemented in the ```pufferlib/pufferl.py```file. Look for **metra** flags.
