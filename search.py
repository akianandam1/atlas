import random
import torch
from ModelVisuals import randoptimize
import multiprocessing
import time


m_1 = random.random()
m_2 = random.random()
m_3 = random.random()
x_1 = random.random()*2-1
v_1 = random.random()*2-1
v_2 = random.random()*2-1
vec = torch.tensor([x_1,0,0,1,0,0,0,0,0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1 + m_2*v_2)/m_3, 0], requires_grad = True)


i = 0
while i < 30:
    randoptimize(vec, m_1, m_2, m_3)
    time.sleep(2)
    i += 1
