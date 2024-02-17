import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

# Define probability bounds
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
    

# Run Simulation 
n_steps = 400
n_sims = 1000
h0 = 10
alpha = 0.99
p = 1/6
sigma_step = np.sqrt(4/12)
n_plots = 5


# if UNIFORM: # These should all be less than or eqaul to sqrt(4/12)
#     sigma_step = np.sqrt(4/12)
# elif TRUNCNORM: 
#     sigma_step = np.sqrt(truncnorm.stats(-1,1)[1] + 0) 
# else: 
#     sigma_step = np.sqrt(p + (1-p)*(p/(1-p))**2)

def randomSample(n_sims, dist_type="uniform"):
    r = np.random.uniform(size=n_sims)
    if dist_type == "uniform": # uniform distribution 
        return r*2 - 1 
    elif dist_type == "trunc_norm": # truncated gausssian, variance = 0.29112509
        r = truncnorm.rvs(-1,1,size=n_sims)
    elif dist_type == "binary": # binary 
        r = -1 * (r < p) + (p/(1-p)) * (r>=p)
    return r 

def run_sim(dist_type = "uniform"):     

    ts = []
    trajs = np.zeros((n_sims, n_steps))
    x0 = np.ones(n_sims)*h0
    x_next = x0 
    for i in range(n_steps): 
        ts.append(i)
        r = randomSample(n_sims, dist_type)
        x_next = alpha*x_next + r 
        trajs[:,i] = x_next

    if False: 
        plt.figure()
        plt.plot(ts, trajs.T, 'k', alpha=0.1)
        plt.plot([ts[0], ts[-1]], [-1/(1-alpha), -1/(1-alpha)], 'r')
        plt.show("traj.png")

    return ts, trajs

dists = ["uniform", "trunc_norm", "binary"]
colors = ['pink', 'g', 'y']
if True: 
    cs = np.linspace(0, 1/(1-alpha), 1000)

    fig, axs = plt.subplots(1, n_plots, figsize=(100, n_plots))

    for d_idx, dist in enumerate(dists): 
        ts, trajs = run_sim(dist)

        Ks = np.linspace(1, n_steps, n_plots)
        for i, k in enumerate(Ks): 
            probs = []
            sum_quad_var = np.sum([alpha**(2*(k -i) ) for i in range(int(k))])
            sigma = sigma_step*np.sqrt(sum_quad_var) 
            hmin = alpha**k*h0 + sum([-alpha**i for i in range(int(k)) ]) 
            hmin = min(hmin, 0 )
            ps = []
            p_unsafe = []
            
            for c in cs: 
                lambduh = alpha**k * (h0 + c ) +  alpha**(k-1)*( 1- alpha ) 
                if dist == "binary": 
                    if -c < hmin: 
                        ps.append(0)
                    else: 
                        ps.append(Pu_Hoeff(k, lambduh, sigma))
                unsafe = np.sum(trajs[:,:int(k)] < -c, axis = 1)
                n_unsafe = np.sum(unsafe>0)
                p_unsafe.append(n_unsafe/n_sims)
            
            axs[i].plot(-np.array(cs), p_unsafe, color = colors[d_idx], alpha = 1, marker=".", markersize=7.5, linestyle="")
            if dist == dists[-1]:
                axs[i].plot([0, hmin, hmin, -cs[-1]],[1,1,0,0], 'r', linewidth=3)
                axs[i].plot(-np.array(cs), ps, 'b', linewidth=3)
                axs[i].set_ylim((-0.1, 1.1))
                axs[i].set_title(f'K={int(k)}')
                axs[i].set_xlabel('Level Set Value')
                axs[i].set_aspect(cs[-1])
                axs[i].invert_xaxis()
        # axs[i].set_box_aspect(1)

    axs[0].set_ylabel('Safety Probability Lower Bound')

    # plt.show()
    plt.show("temp.png")
    # plt.savefig("issf_compare.svg")

