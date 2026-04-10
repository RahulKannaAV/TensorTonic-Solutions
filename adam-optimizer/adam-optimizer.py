import numpy as np
import math

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    new_param = []
    m_new = []
    v_new = []
    
    for i in range(len(param)):
        # First moment updation
        mt = (beta1 * m[i]) + ((1-beta1)*grad[i])

        # Second moment updation
        vt = (beta2 * v[i]) + ((1 - beta2) * (grad[i]**2))

        # Bias corrections
        mt_cap = mt / (1 - (beta1**t))
        vt_cap = vt / (1 - (beta2**t))

        # Parameter updation
        param_dash = param[i] - ((lr * mt_cap)/((math.sqrt(vt_cap) )+ eps))
        new_param.append(param_dash)
        m_new.append(mt)
        v_new.append(vt)

    new_param = np.array(new_param)
    m_new = np.array(m_new)
    v_new = np.array(v_new)

    return (new_param, m_new, v_new)