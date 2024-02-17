import numpy as np
import matplotlib.pyplot as plt
def P_steinhardt(h0, K, c, M): 
    return 1 - (h0 - K*c)/M

def P_new(h0, K, c, sigma):

    outer_exp = K/(K + sigma**2)

    if K > h0/(1 + c): 
        frac2 = ((K)        / (K*(1 + c) - h0))**((K*(1 + c) - h0) * outer_exp)
    elif K == h0/(1 + c): 
        frac2 = 1
    else: 
        return 0 
    try: 
        frac1 = ((sigma**2) / (h0 - K*c + sigma**2))**((h0- K*c + sigma**2)*outer_exp)
    except:     
        breakpoint()

    P = (frac1 * frac2 )**outer_exp
    
    return P 

M = 100
h0 = 100
c = 0.1 
ps = []
pn = []
ks = []
sigma = 10
K = 1000
for k in range(K): 
    ks.append(k)
    ps.append(P_steinhardt(h0, k, c, M))
    pn.append(P_new(h0, k, c, sigma))

plt.figure()
plt.plot(ks, ps, '.')
plt.plot(ks, pn, '.')
plt.legend(["old", "new"])
plt.show()