from games.SIR import SIR_Game
from games.ring import ring_Game
import numpy as np 
import torch 
W = np.array([
    [0.8,0.4],
    [0.4, 0.8]
])
K = len(W)
game = SIR_Game(W,K=2)
game1 = ring_Game(W,K=2)
game.pop_inf(iter=1000)
print(game.Vh_func(0))
print(game.Gamma_q_func(0))

q = game.Gamma_q_func(0)[0]
print(q)
print(game.mirror(0,0,q,eta = 2))
