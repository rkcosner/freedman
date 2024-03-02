import numpy as np
import matplotlib.pyplot as plt

# Probability Stuff
np.random.seed(0)
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


# Calculate HLIP Matrices
z = 0.62 # Robot nominal z-height
T_SSP = 0.33 # Time length of single support phase 
vel_x = 0.1 # Nominal velocity (?) 

T_DSP = 0.0 # Double support phase time length 
g = 9.81 

# Derived variables 
lambduh = np.sqrt(g/z) 

eA_ssp = np.array([ [-lambduh * (np.exp(lambduh * T_SSP) + np.exp(-lambduh * T_SSP)),    -np.exp(lambduh * T_SSP + np.exp(-lambduh * T_SSP))], 
                    [-lambduh**2 * (np.exp(lambduh * T_SSP) + np.exp(-lambduh * T_SSP)),  -lambduh * (np.exp(lambduh * T_SSP) + np.exp(-lambduh * T_SSP))]
                ]) / (-2 * lambduh)

A_step = np.array([ [1, T_DSP], 
                    [0, 1]])

B_step = np.array([ [-1], 
                    [0]])

A = eA_ssp @ A_step 
B = eA_ssp @ B_step 

step_length = 0.0176#vel_x * T_SSP 

sigma_1 = lambduh * np.tanh(T_SSP * lambduh / 2.0);

x_impact = np.zeros((2, 1))
x_impact[0,0] = vel_x * (T_SSP + T_DSP) / (2.0 + T_DSP * sigma_1)
x_impact[1,0] = sigma_1 * vel_x * (T_SSP + T_DSP) / (2.0 + T_DSP * sigma_1)

x_impact_next = A @ x_impact + B * step_length 





A_global = np.array([[  1, A[0,0], A[0,1]], 
                     [  0, A[0,0], A[0,1]], 
                     [  0, A[1,0], A[1,1]]
                    ])

B_global = np.array([   [B[0,0]], 
                        [B[0,0]], 
                        [B[1,0]]
                    ])

x_global = np.array([   [0], 
                        [x_impact[0,0]], 
                        [x_impact[1,0]]
                    ])

C2D = np.array([[1,0,0,0,0,0], 
                [0,0,0,1,0,0]])
A2D = np.block([[A_global, np.zeros((3,3))], 
                 [np.zeros((3,3)), A_global]])
B2D = np.block([[B_global, np.zeros((3,1))], [np.zeros((3,1)), B_global ]])
x2D = np.block([[x_global],[x_global]])

n_sims = 1000
n_to_plot = 100

# A2D = np.repeat(A2D[np.newaxis,:,:], repeats=n_sims, axis=0)
# B2D = np.repeat(B2D[np.newaxis,:,:], repeats=n_sims, axis=0)
x2D = np.repeat(x2D[np.newaxis,:,:], repeats=n_sims, axis=0)


class wall_barrier(): 
    def __init__(self, const, alpha, plane): 
        self.const = const
        self.alpha = alpha 
        self.c = plane 
    
    def val(self, x): 
        return self.const - self.c @ x
    
    def get_bits(self, x): 
        Lfh = self.const - self.c @ A2D @ x
        Lgh = - self.c @ B2D 
        ah = self.alpha * self.val(x)
        return Lfh, Lgh, ah

    def filter(self, x, u_nom): 
        Lfh, Lgh, ah = self.get_bits(x) 
        if Lfh + Lgh @ u_nom >= ah: 
            return u_nom
        else: 
            return u_nom + Lgh.T/(Lgh @ Lgh.T) *(-Lfh - Lgh @ u_nom + ah) 


class circle_barrier(): 
    def __init__(self, radius, alpha, center): 
        self.radius =radius
        self.alpha = alpha 
        self.center = center 
    
    def val(self, x): 
        return np.linalg.norm(x[:,[0,3],:] - self.center, axis = 1) - self.radius 

    def get_bits(self, x): 
        # Convexify 
        unit_vec = C2D @ x - self.center 
        norm = np.linalg.norm(unit_vec, axis=1)[:,0]
        unit_vec[:,0,0] /= norm
        unit_vec[:,1,0] /= norm

        # Calculate Barrier Bits for convexified barrier
        Lfh = unit_vec.transpose(0,2,1) @ (C2D @ A2D @ x - self.center) - self.radius
        Lgh = unit_vec.transpose(0,2,1) @ C2D @ B2D 
        ah = self.alpha * self.val(x)
        return Lfh, Lgh, ah

    def filter(self, x, u_nom): 
        Lfh, Lgh, ah = self.get_bits(x) 

        switch = ((Lfh + Lgh @ u_nom)[:,:,0] >= ah)[:,0]

        opt_u = np.zeros(u_nom.shape)
        opt_u[switch] = u_nom[switch]
        not_switch = (switch == False * ah[:,0])
        opt_u[not_switch] = u_nom[not_switch] + Lgh[not_switch].transpose(0,2,1)/(Lgh[not_switch] @ Lgh[not_switch].transpose(0,2,1)) @(-Lfh[not_switch] - Lgh[not_switch] @ u_nom[not_switch] + ah[not_switch,np.newaxis])

        return opt_u 

        
    def get_plot_points(self): 
        pts = []
        thetas = np.linspace(0,2*np.pi, 300)
        for theta in thetas:
            pts.append([np.cos(theta)*self.radius + self.center[0,0], np.sin(theta)*self.radius + self.center[1,0]])
        pts = np.array(pts)
        return pts 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

noise_maxes = [0.001, 0.01, 0.05, 0.1]
alphas = [0.9, 0.95, 0.999]
T = 25
n_steps = int(T/T_SSP) 
print("N steps = " + str(n_steps))
center = np.array([[1.5],[1]])
radius = 1 * np.sqrt(2) - 0.5


fig, axs = plt.subplots(2*len(alphas),len(noise_maxes))# gridspec_kw={'height_ratios':[4,1]},) 

for alpha_number, alpha in enumerate(alphas): 
    for sim_number, trunc_max in enumerate(noise_maxes):
        # trunc_max = 0.05
        v_dist_scale = 1
        cbf = circle_barrier(radius, alpha, center)
        u_nom = np.array([[step_length]])
        traj = np.zeros((n_sims, 2, n_steps+1))
        hs = np.zeros((n_sims, 1, n_steps+1))
        v_des = 0.1

        # Starting positoin 
        x2D[:, 0,0] = 0
        x2D[:, 3,0] = 0

        v_des = np.array([[v_des], [v_des]])
        v_des = np.repeat(v_des[np.newaxis,:,:], repeats=n_sims, axis=0 )


        from scipy.stats import truncnorm        
        D = np.array([[1,0,0,0], 
                    [1,0,0,0], 
                    [0,1,0,0], 
                    [0,0,1,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        delta = np.linalg.norm(np.eye(2)) * np.linalg.norm(np.array([1,1])) * trunc_max
        sigma = np.sqrt(trunc_max**2/3)#np.sqrt(truncnorm.stats(-trunc_max, trunc_max,moments="v", scale = trunc_scale))
        def step(x2D, step_length): 
            # v = truncnorm.rvs(-trunc_max, trunc_max,size=(n_sims, 6,1), scale = trunc_scale)
            v = np.random.uniform(-trunc_max,trunc_max,size=(n_sims,4,1))
            v[:,1,0] = np.random.uniform(-trunc_max*v_dist_scale,trunc_max*v_dist_scale,size=(n_sims))
            v[:,3,0] = np.random.uniform(-trunc_max*v_dist_scale,trunc_max*v_dist_scale,size=(n_sims))
            x_next = A2D @ x2D + B2D @ step_length  + D@v 
            return x_next

        # Simulate 

        traj[:,:,0] = x2D[:,[0,3],0]
        hs[:,:,0] = cbf.val(x2D)
        h0 = hs[0,0,0]
        violations = hs*0 
        ts = [0]
        ps = [0]
        for i in range(n_steps): 
            ell_vtracking = np.zeros((n_sims, 2,1))
            ell_vtracking[:,0,0] = (v_des[:, 0,0] - (A2D[:3,:3] @ x2D[:, :3])[:,2,0])/B2D[2,0]
            ell_vtracking[:,1,0] = (v_des[:, 1,0] - (A2D[3:,3:] @ x2D[:, 3:])[:,2,0])/B2D[5,1]
            u_nom = ell_vtracking

            u = cbf.filter(x2D, u_nom)
            x2D = step(x2D, u)
            traj[:,:,i+1] = x2D[:,[0,3],0]
            hs[:,:,i+1] = cbf.val(x2D)

            violations[:,:,i+1] =  (hs[:,:,i+1] < 0) * ( 1 - violations[:,:,i]) + violations[:,:,i] # if the last was safe and the current is unsafe and if there was ever a previous unsafe
            ts.append(T_SSP*(i+1))
            p_bound = Pu(i+1,alpha**(i+1) * h0 / delta,  sigma * np.sqrt(i+1) / delta )
            ps.append(p_bound)
            # print(cbf.val(x_global), x_global)

        violations = np.sum(violations, axis = 0 )




        # Plot Percents
        axs[1 + 2 * alpha_number, sim_number].plot(ts, ps, 'b')
        axs[1 + 2 * alpha_number, sim_number].plot(ts, violations[0,:]/n_sims, 'r')
        axs[1 + 2 * alpha_number, sim_number].set_xlabel("time [sec]")
        axs[1 + 2 * alpha_number, sim_number].set_ylabel("$P_u$")
        axs[1 + 2 * alpha_number, sim_number].set_ylim((-0.05,1.05))
        # axs[1 + 2 * alpha_number, sim_number].get_yaxis().set_visible(False)
        # axs[1 + 2 * alpha_number, sim_number].get_xaxis().set_visible(False)
        print(violations[0,-1])


        # Plot trajectory 
        pts = cbf.get_plot_points()
        axs[0 + 2 * alpha_number, sim_number].plot(pts[:,0], pts[:,1], 'r')
        axs[0 + 2 * alpha_number, sim_number].plot(np.swapaxes(traj[:n_to_plot,0,:],0,1), np.swapaxes(traj[:n_to_plot,1,:],0,1), color="b", alpha = 0.1 )
        axs[0 + 2 * alpha_number, sim_number].set_aspect("equal")
        axs[0 + 2 * alpha_number, sim_number].set_xlabel(r"$\mathbf{p}_x$")
        axs[0 + 2 * alpha_number, sim_number].set_ylabel(r"$\mathbf{p}_y$")
        axs[0 + 2 * alpha_number, sim_number].set_xlim((-1,3))
        axs[0 + 2 * alpha_number, sim_number].set_ylim((-1,3))
        # axs[0 + 2 * alpha_number, sim_number].get_yaxis().set_visible(False)
        # axs[0 + 2 * alpha_number, sim_number].get_xaxis().set_visible(False)


# plt.show()
plt.savefig("plots/hlip_plot.svg")