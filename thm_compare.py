import numpy as np 
import matplotlib.pyplot as plt

def Pu_Villes(x, B):
    if x < B:
        return 1 - x/B 
    else: 
        return 0

def Pu_Hoeff(K, x, sigma):
    if x <= K: 
        frac1 = (sigma**2) / (x + sigma**2)
        exp1 = (x + sigma**2)
        if x < K: 
            frac2 = (K) / (K - x)
            exp2 = (K - x)
        else: 
            frac2 = 1 
            exp2 = 1 

        if frac1**exp1*frac2**exp2 < 1 : 
            return (frac1**exp1*frac2**exp2)**(K/(K + sigma**2))
        else: 
            return 1 
    else: 
        return 0


def Pu_Freedman(x, sigma):
    try:
        frac1 = (sigma**2) / (x + sigma**2)
        exp1 = (x + sigma**2)
    except: 
        breakpoint()
    return frac1**exp1 * np.exp(x)



def Pu(K,x,sigma): 
    return Pu_Freedman( x, sigma)


B = 10
zs = []
ys = []
yHs = []
vals = []
K = 100
xs = np.linspace(0, B, 3000)

for x in xs:
    sigma = np.sqrt(np.exp(-1)*x)
    vals.append(sigma) 
    zs.append(Pu_Villes(x, B))
    ys.append([Pu(10, x, sigma/4), Pu(10, x, sigma/2), Pu(10, x, sigma), Pu(10, x, sigma*2), Pu(10, x, sigma*4)])
    yHs.append([Pu_Hoeff(10, x, sigma/4), Pu_Hoeff(10, x, sigma/2), Pu_Hoeff(10, x, sigma), Pu_Hoeff(10, x, sigma*2), Pu_Hoeff(10, x, sigma*4)])

ys = np.array(ys)
yHs = np.array(yHs)
vals = np.array(vals)

# Set matplotlib font to times 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12


fig = plt.figure()
fig.set_size_inches(6, 3)
plt.plot(vals, zs, 'r')
plt.plot(vals, ys[:,0]*0, 'k', linewidth=0.5, linestyle='--')
plt.plot(np.sqrt(np.exp(-1)*(B-1)) + vals*0, vals/np.max(vals), 'k', linewidth=0.5, linestyle='--')
plt.plot(vals, ys[:,0], color=(0.7,0.7,0.7), linestyle='--')
plt.plot(vals, ys[:,1], color=(0.7,0.7,0.7), linestyle='--')
plt.plot(vals, ys[:,2], 'b')
plt.plot(vals, ys[:,3], color=(0.7,0.7,0.7), linestyle='--')
plt.plot(vals, ys[:,4], color=(0.7,0.7,0.7), linestyle='--')
plt.legend(['Ville', 'Hoeffding, $\sigma^2 = \frac{1}{2}\sqrt{e^{-1}\lambda}$','Hoeffding, $\sigma^2 = \sqrt{e^{-1}\lambda}$', 'Hoeffding, $\sigma^2 = 2\sqrt{e^{-1}\lambda}$'])
plt.xlabel('$\lambda$')
plt.ylabel('Probability')
plt.show()
# plt.savefig('plots/hoeffding_ville_comparison.svg')