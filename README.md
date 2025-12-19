# Vector Quantized-Elites: Unsupervised and Problem-Agnostic Quality-Diversity Optimization

This is an implementation of the [VQ-Elites]([https://arxiv.org/abs/2310.08887v2](https://arxiv.org/abs/2504.08057)) paper using pufferlib. 

To use, git clone & pip install in editable mode: 
```bash
pip install -e .
```

Make sure vq elites is enabled in conf ```config/default.ini```, then 
```
puffer train puffer_convert_circle
```

The main code is implemented in the ```pufferlib/pufferl.py```file. Most of the algorithm code is in the train method.
