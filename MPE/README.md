# WToE_MPE

## Requirements
 ```shell
$ conda create -n WToE-MPE python=3.6.12
$ conda activate WToE-MPE
$ pip install -r requirements.txt
```

## Quick Start

```shell
$ python simple_adversary/main.py --scenario-name=simple_adversary --algorithm=WToE
```

Directly run the main.py, then the algrithm will be trained on scenario 'simple_adversary' with 'WToE' algorithm.

## Algorithm
We impletent the WToE, MADDPG, PR2, NoisyNets algorithm.
Training usage: 
```shell
$ python simple_adversary/main.py --scenario-name=simple_adversary --algorithm=MADDPG
```
The number of agents can be adjusted in `common/arguments.py`.

You can choose different scenarios, including `simple_crypto`, `simple_push`, `simple_reference`, `simple_spread`, `simple_tag`, as follows:
```shell
$ python simple_crypto/main.py --scenario-name=simple_crypto --algorithm=MADDPG
```
Note that, the filename should be consistent with the name of scenario.