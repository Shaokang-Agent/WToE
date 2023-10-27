# WToE_GRID

### Environment
The environment contains two basic grid environment (2-rooms and 4-rooms environment), which are implemented in the `ENV` file.

### Requirement
```shell
$ conda create -n WToE-Grid python=3.7.9
$ conda activate WToE-Grid
$ pip install -r requirements.txt
```

### Algorithm
We impletent the WToE, EITI, Max-entropy Q-learning, $\epsilon$-greedy (baseline) algorithms.

### Multi-agent training
In the multi-agent setting, we test the exploration efficiency of different algorithms. 
Training usage: 
```shell
$ python Two_rooms/obstacle_WToE.py
``` 
or 
```shell
$ python Four_rooms/WToE.py
```
