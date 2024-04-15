import numpy as np
import math


def uniform_attachment_graphon(x, y):
    return 1 - np.maximum(x, y)


def ranked_attachment_graphon(x, y):
    return 1 - x * y


def er_graphon(x, y, p=0.5):
    return np.ones_like(x) * p


def power_law_graphon(x, y, alpha=0.5):
    return (1-alpha)**2 * np.power(x * y, -alpha)


def cutoff_power_law_graphon(x, y, alpha=0.5, c=0.1):
    return ((1-alpha) / (1-alpha * np.power(c, 1-alpha)))**2 * np.power(np.maximum(x,c) * np.maximum(y,c), -alpha)

def sbm_graphon(x, y, a = 0.9 , b = 0.3, c = 0.9):

    if (0<=x<0.7 and 0<=y<0.7) or (0.7<=x<=1 and 0.7<=y<=1):
        return a
    elif (0<=x<0.7 and 0.7<=y<=1) or (0<=y<0.7 and 0.7<=x<=1):
        return b
    else:
        raise NotImplementedError

    '''
    if (0<=x<0.3 and 0<=y<0.3) or (0.3<=x<0.6 and 0.3<=y<0.6) or (0.6<=x<=1 and 0.6<=y<=1):
        return 1
    elif (0.3<=x<0.6 and 0<=y<0.3) or (0.3<=y<0.6 and 0<=x<0.3):
        return a
    elif (0.6<=x<=1 and 0<=y<0.3) or (0.6<=y<=1 and 0<=x<0.3):
        return b
    elif (0.6<=x<=1 and 0.3<=y<0.6) or (0.6<=y<=1 and 0.3<=x<0.6):
        return c
    else:
        raise NotImplementedError
    '''
    
def exp_graphon(x, y, theta=3.0):
    return 2*(math.exp(theta*x*y)/(1+math.exp(theta*x*y))-0.5)
