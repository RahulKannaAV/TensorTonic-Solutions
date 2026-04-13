import numpy as np
from collections import Counter

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    prob_dict = Counter(y)
    entropy_val = 0.0
    for key in prob_dict:
        prob_dict[key] /= len(y)

    for c in range(len(prob_dict.keys())):
        prob = prob_dict[c]
        if(prob != 0.0):
            entropy_val -= (prob * np.log2(prob))

    return entropy_val