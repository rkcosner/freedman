import numpy as np
import matplotlib.pyplot as plt

# Calculate HLIP Matrices
z = 0.62 # Robot nominal z-height
T_SSP = 0.3 # Time length of single support phase 
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


class barrier(): 
    def __init__(self, const, alpha ): 
        self.const = const
        self.alpha = alpha 
        self.c = np.array([[1, 0, 0]])
    
    def val(self, x): 
        return self.const - self.c @ x
    
    def get_bits(self, x): 
        Lfh = 10 - self.c @ A_global @ x
        Lgh = - self.c @ B_global 
        ah = self.alpha * self.val(x)
        return Lfh, Lgh, ah

    def filter(self, x, u_nom): 
        Lfh, Lgh, ah = self.get_bits(x) 
        if Lfh + Lgh @ u_nom >= ah: 
            return u_nom
        else: 
            return u_nom + Lgh.T/(Lgh @ Lgh.T) *(-Lfh - Lgh @ u_nom + ah) 

def h(x): 
    c = np.array([[1,0,0]])
    return 10 - c @ x 


def step(x_global, step_length): 
    x_next = A_global @ x_global + B_global * step_length 
    return x_next



alpha = 0.5
wall_loc = 10 
cbf = barrier(wall_loc, alpha) 
u_nom = np.array([[step_length]])
traj = []
hs = []
for i in range(20): 
    u = cbf.filter(x_global, u_nom)
    x_global = step(x_global, u)
    traj.append(x_global[0,0])
    hs.append(cbf.val(x_global)[0,0])
    # print(cbf.val(x_global), x_global)

fig, axs = plt.subplots(2,1) 
axs[0].plot(hs) 
axs[1].plot(traj)
plt.show()

# % Configuration variables
# z = 0.62;
# T_SSP = 0.3;
# vel_x = 0.1;

# % Other constants
# T_DSP = 0.0;
# g = 9.81;

# % Derives variables
# lambda = sqrt(g / z);
# step_length = vel_x * T_SSP;
# eA_ssp = eAssp(T_SSP, lambda);
# A_step = [1, T_DSP; 0, 1];
# B_step = [-1; 0];

# % The step to step discrete dynamics
# A = eA_ssp * A_step;
# B = eA_ssp * B_step;

# % Test if the results makes sense
# sigma_1 = CalculateSigma1(lambda, T_SSP);

# x_impact = zeros(2, 1);
# x_impact(1) = vel_x * (T_SSP + T_DSP) / (2.0 + T_DSP * sigma_1);
# x_impact(2) = sigma_1 * vel_x * (T_SSP + T_DSP) / (2.0 + T_DSP * sigma_1);

# x_impact_next = A * x_impact + B * step_length;