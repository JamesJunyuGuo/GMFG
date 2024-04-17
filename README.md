# Graphon MFG 

This is the README file of the GMFG project. 
To run the code, configure the environment with conda virtual env

```shell
conda create --name gmfg
conda activate gmfg
pip install -r requirements.txt
```

See the *Finite_Horizon* folder for the implementation of the beach bar model with finite horizon.   

1. For the beach experiment and the Financial Network experiment, to change the reward function, refer to the following files.     
```shell
cd Finite_Horizon/
```
See  *games/Finite/beach.py* to change the *reward_g* function, and in *games/graphon_mfg.py* we will use the reward function quoting *reward_g* function. In this way we can chang the structure of the reward function, the same applies to other models such as the Cyber network and other models as well. 
2. To choose different Graphon functions, refer to */games/graphons.py* and you can define the graphon function you want and import the function in the main file *main.py* to initialize the game. 


See the *Infinite_Horizon* folder for the implementation of the ring  game and the epidemic model with infinite horizon.   
 

#### References 
$\cdot$ [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch.git)       

$\cdot$ [Monotone MFG](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d4c2f25bf0c33065b7d4fb9be2a9add1-Abstract-Conference.html)

