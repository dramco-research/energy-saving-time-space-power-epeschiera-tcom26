import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams.update({'font.size':17})
params = {"xtick.direction": "in", "ytick.direction": "in"}
plt.rcParams.update(params)
plt.close('all')
import warnings
warnings.filterwarnings("ignore")

from utils import (PC_model, inv_eCDF_SNR5G, inv_eCDF_SNR4G,
                   solve_problem0, solve_problem1, solve_problem2, solve_problem3)

class colors:
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'

# Function to compute the (continuous) power consumption
def f(x):
    return x[1]/x[0]*P_0/M+x[1]*P_1/M+\
           x[1]/x[0]*gamma*((1/(x[1]*(x[1]-Ka)))*sigma2/beta@(np.exp(Rpr*x[0])-1))**alpha

# Function to compute the transmit power at any active antenna
def Pa_func(na,ma):
    return 1/(ma*(ma-Ka))*sigma2/beta@(np.exp(Rpr*N/na)-1)

config_vec = ['8T8R','64T64R']                          # Vector of BS configurations
K_vec = np.array([4,8])                               # Vector of numbers of users
kappa_prc_vec = [1,6,18]                                # Vector of kappa [%]
T_frame = 10e-3                                         # Frame duration [s]
alpha = 0.75                                            # PA consumption exponent

NF_dB = 9           					                # Noise factor [dB]
NT = 300            					                # Noise temperature [K]
kB = 1.380649e-23   					                # Boltzmann constant [W/K/Hz]
tol = 1e-8                                              # Tolerance of Newton's method
max_it = 1e3                                            # Maximum number of iterations of Newton's method

# Comment one of the following two lines to select enabled/disabled time-domain hardware power-saving modes
# delta_PA_dtx = 1; delta_TRX_idle = 1
delta_PA_dtx = 0.25; delta_TRX_idle = 0.5

N_iter = int(1e3)                                       # Number of iterations (channel realizations)
# To store the results
P_cons = np.zeros([2,8,len(kappa_prc_vec),N_iter])      # Power consumption of optimized (proposed) scheme
P_cons_AbW = np.zeros([2,8,len(kappa_prc_vec),N_iter])  # Power consumption of awake-but-whisper scheme
P_cons_RtS = np.zeros([2,8,len(kappa_prc_vec),N_iter])  # Power consumption of rush-to-sleep scheme
P_cons_RtM = np.zeros([2,8,len(kappa_prc_vec),N_iter])  # Power consumption of rush-to-mute scheme
P_cons_B27 = np.zeros([2,8,len(kappa_prc_vec),N_iter])  # Power consumption of benchmark [27] scheme
R_stats = np.zeros([2,8,len(kappa_prc_vec),N_iter])     # Collect rates
EE = np.zeros([2,8,len(kappa_prc_vec),N_iter])          # Energy efficiency of optimized (proposed) scheme
EE_AbW = np.zeros([2,8,len(kappa_prc_vec),N_iter])      # Energy efficiency of awake-but-whisper scheme
EE_RtS = np.zeros([2,8,len(kappa_prc_vec),N_iter])      # Energy efficiency of rush-to-sleep scheme
EE_RtM = np.zeros([2,8,len(kappa_prc_vec),N_iter])      # Energy efficiency of rush-to-mute scheme
EE_B27 = np.zeros([2,8,len(kappa_prc_vec),N_iter])      # Energy efficiency of benchmark [27] scheme


# Iterate over BS configuration
for cfg in range(2):
    config = config_vec[cfg]
    Ka_vec = np.arange(K_vec[cfg])+1
    K = K_vec[cfg]

    # Iterate over number of active users
    for ka in range(len(Ka_vec)):
        Ka = Ka_vec[ka]
        print('BS config:',cfg,', No. of users:',Ka)

        gamma, P_0, P_1, P_sleep, B, P_max, M, K, T, tau_CP, tau_DL, tau_sig  = \
            PC_model(config, delta_PA_dtx, delta_TRX_idle, Ka, alpha)
        gamma_pr, P_0_pr, P_1_pr, _, _, _, _, _, _, _, _, _ = \
            PC_model(config, delta_PA_dtx, delta_TRX_idle, Ka, 1)
        N = np.floor((tau_DL-tau_sig*tau_DL)*T_frame/T)     # Number of time slots (OFDM symbols)
        if config == '64T64R' or config == '32T32R':
            P_T = 20                                            # Total transmit power at BS [W]
        elif config == '8T8R':
            P_T = 32
        else:
            P_T = 160
        sigma2 = NT*kB*B*10**(NF_dB/10) 		                # Noise power [W]

        # Iterate over network load
        for i in range(len(kappa_prc_vec)):
            kappa_prc = kappa_prc_vec[i]

            n = 0
            while n < N_iter:
                if np.mod(n,100) == 0:
                    print('Load',i+1,', iter',n,'/',N_iter)

                if config == '8T8R' or config == '32T32R' or config == '64T64R':
                    SNR = inv_eCDF_SNR5G(Ka)                        # SNR vector
                else:
                    SNR = inv_eCDF_SNR4G(Ka)
                beta = (sigma2*SNR)/(P_T*(M-1))                     # Large-scale fading coefficient vector
                rho = P_max*beta/sigma2                             # Single-antenna equivalent SNR

                R_0 = np.random.uniform(0,1,Ka)                     # Baseline user rates [bit/subcarrier/OFDM symbol]
                R_0 = R_0/np.sum(R_0)

                def obj_kappamax(x):
                    return np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*rho**(-1)@(np.exp(x*R_0*np.log(2))-1)))
                constrs = [{'type': 'ineq', 'fun': lambda x: x}]
                # noinspection PyTypeChecker
                result = minimize(obj_kappamax, np.array(1), constraints=constrs,
                                  tol=1e-8, options={'disp': False, 'maxiter': 100})
                kappa_max = result.x[0]

                kappa = (kappa_prc/100)*kappa_max
                R = kappa*R_0
                Rpr = R*np.log(2)

                R_sum = B*(1-tau_CP)*tau_DL*(1-tau_sig)*np.sum(R)           # Sum rate [bit/s/Hz]

                Ma_min = int(np.ceil(Ka/2+1/2*np.sqrt(Ka**2+4*rho**(-1)@(np.exp(Rpr)-1))))
                Na_vec = np.arange(N) + 1
                tmp = 0
                for k in range(Ka):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*N/Na_vec)-1)
                Na_min = Na_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                Pa_min = Pa_func(N,M)
                if Pa_min > 1.01 * P_max:
                    print('Problem not feasible!!!')
                    continue

                # Solve the problem with Algorithm 1
                [x,y] = solve_problem0(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                if x >= 1 and y <= M and y >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    pass
                elif x < 1 and y <= M and y >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    x = 1
                    y = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                    if y > M and y >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                        y = M
                    elif y <= M and y < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                        y = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
                elif x < 1 and y > M and y >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    x1 = 1
                    y1 = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                    if y1 > M and y1 >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                        y1 = M
                    elif y1 <= M and y1 < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                        y1 = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta.T@(np.exp(Rpr*x1)-1))
                    tmp1 = f([1,y1])
                    y2 = M
                    x2 = solve_problem2(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                    if x2 < 1 and y2 >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1)):
                        x2 = 1
                    elif x2 >= 1 and y2 < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1)):
                        x_vec = np.linspace(1,1e3,int(1e5))
                        tmp = 0
                        for k in range(Ka):
                            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                        x2 = x_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                    tmp2 = f([x2,M])
                    if tmp1 <= tmp2:
                        x = 1
                        y = y1
                    else:
                        x = x2
                        y = M
                elif x < 1 and y <= M and y < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    x1 = 1
                    y1 = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                    if y1 > M and y1 >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                        y1 = M
                    elif y1 <= M and y1 < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                        y1 = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1))
                    tmp1 = f([x1,y1])
                    x2 = solve_problem3(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1,P_max)
                    y2 = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1))
                    if x2 < 1 and y2 <= M:
                        x2 = 1
                    elif x2 >= 1 and y2 > M:
                        x_vec = np.linspace(1,1e3,int(1e5))
                        tmp = 0
                        for k in range(Ka):
                            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                        x2 = x_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                        y2 = M
                    tmp2 = f([x2,y2])
                    if tmp1 <= tmp2:
                        x = 1
                        y = y1
                    else:
                        x = x2
                        y = y2
                elif x >= 1 and y > M and y >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    y = M
                    x = solve_problem2(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                    if x < 1 and y >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                        x = 1
                    elif x >= 1 and y < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                        x_vec = np.linspace(1,1e3,int(1e5))
                        tmp = 0
                        for k in range(Ka):
                            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                        x = x_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                elif x >= 1 and y <= M and y < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    x = solve_problem3(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1,P_max)
                    y = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
                    if x < 1 and (y <= M or np.isnan(y)):
                        x = 1
                        y = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
                    elif x >= 1 and y > M:
                        y = M
                        x_vec = np.linspace(1,1e3,int(1e5))
                        tmp = 0
                        for k in range(Ka):
                            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                        x = x_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                elif x >= 1 and y > M and y < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                    y1 = M
                    x1 = solve_problem2(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1)
                    if x1 < 1 and y1 >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                        x1 = 1
                    elif x1 >= 1 and y1 < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                        x_vec = np.linspace(1,1e3,int(1e5))
                        tmp = 0
                        for k in range(Ka):
                            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                        x1 = x_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                    tmp1 = f([x1,y1])
                    x2 = solve_problem3(tol,max_it,Rpr,sigma2/beta,gamma,M,Ka,alpha,P_0,P_1,P_max)
                    y2 = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1))
                    if x2 < 1 and y2 <= M:
                        x2 = 1
                    elif x2 >= 1 and y2 > M:
                        y2 = M
                        x_vec = np.linspace(1,1e3,int(1e5))
                        tmp = 0
                        for k in range(Ka):
                            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                        x2 = x_vec[np.argmin(np.abs(M-Ka/2-1/2*np.sqrt(Ka**2+4*tmp)))]
                    tmp2 = f([x2,y2])
                    if tmp1 <= tmp2:
                        x = x1
                        y = y1
                    else:
                        x = x2
                        y = y2

                if np.isnan(x) or np.isnan(y):
                    print('x or y is nan')
                    continue
                Na = int(np.round(N/x))
                Ma = np.max([Ka+1,int(np.round(y))])
                Pa = Pa_func(Na,Ma)

                # Check that transmit power constraint is satisfied
                while Pa > P_max:
                    if f([N/(Na+1),Ma]) < f([N/Na,Ma+1]) and Na < N:
                        Na = Na+1
                        Pa = Pa_func(Na,Ma)
                    else:
                        Ma = Ma+1
                        Pa = Pa_func(Na,Ma)

                # Compute solution of benchmark [27]
                x_B27 = 1
                y_B27 = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma_pr,M,Ka,1,P_0_pr,P_1_pr)
                if y_B27 > M and y_B27 >= Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x_B27)-1)):
                    y_B27 = M
                elif y_B27 <= M and y_B27 < Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x_B27)-1)):
                    y_B27 = Ka/2+1/2*np.sqrt(Ka**2+4/P_max*sigma2/beta@(np.exp(Rpr*x_B27)-1))
                Ma_B27 = np.max([Ka+1,int(np.round(y_B27))])
                while Pa_func(N,Ma_B27) > P_max:
                    Ma_B27 = Ma_B27+1
                P_cons_B27[cfg,ka,i,n] = f([1,Ma_B27])+P_sleep

                # Save results
                P_cons[cfg,ka,i,n] = f([N/Na,Ma])+P_sleep
                P_cons_AbW[cfg,ka,i,n] = f([1,M])+P_sleep
                P_cons_RtS[cfg,ka,i,n] = f([N/Na_min,M])+P_sleep
                P_cons_RtM[cfg,ka,i,n] = f([1,Ma_min])+P_sleep

                EE[cfg,ka,i,n] = R_sum/P_cons[cfg,ka,i,n]
                EE_AbW[cfg,ka,i,n] = R_sum/P_cons_AbW[cfg,ka,i,n]
                EE_RtS[cfg,ka,i,n] = R_sum/P_cons_RtS[cfg,ka,i,n]
                EE_RtM[cfg,ka,i,n] = R_sum/P_cons_RtM[cfg,ka,i,n]
                EE_B27[cfg,ka,i,n] = R_sum/P_cons_B27[cfg,ka,i,n]
                R_stats[cfg,ka,i,n] = R_sum

                n += 1


# Plot results (Pcons vs. K)
for ind_cfg in range(len(config_vec)):
    cfg = ind_cfg
    for kappa in range(len(kappa_prc_vec)):
        fig,axs = plt.subplots()
        axs.plot(np.arange(K_vec[cfg])+1,np.median(P_cons[cfg,0:K_vec[cfg],kappa,:],axis=1),
                 '-ob',ms=8,mfc='white',mew=2,lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(P_cons_RtM[cfg,0:K_vec[cfg],kappa,:],axis=1),
                 '--og',ms=8,mfc='white',mew=2,lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(P_cons_RtS[cfg,0:K_vec[cfg],kappa,:],axis=1),
                 '--o',ms=8,mfc='white',mew=2,color='brown',lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(P_cons_AbW[cfg,0:K_vec[cfg],kappa,:],axis=1),
                 '-.or',ms=8,mfc='white',mew=2,lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(P_cons_B27[cfg,0:K_vec[cfg],kappa,:],axis=1),
                 '-.oc',ms=8,mfc='white',mew=2,lw=2)
        axs.set_xticks(np.arange(K_vec[cfg])+1)
        axs.set_xticklabels(np.arange(K_vec[cfg])+1)
        axs.set_xlabel('No. of users, $K$')
        axs.set_ylabel('Power consumption [W]')
        axs.grid(linestyle=':')
        if delta_PA_dtx == 0.25:
            axs.set_title('Load '+str(kappa+1)+', Config. '+config_vec[cfg]+
                          ',\n Enabled time-domain HW power-saving modes',fontsize=13)
        else:
            axs.set_title('Load '+str(kappa+1)+', Config. '+config_vec[cfg]+
                          ',\n Disabled time-domain HW power-saving modes',fontsize=13)
        plt.tight_layout()
        plt.show()

# Plot results (EE vs. K)
for ind_cfg in range(len(config_vec)):
    cfg = ind_cfg
    for kappa in range(len(kappa_prc_vec)):
        fig,axs = plt.subplots()
        axs.plot(np.arange(K_vec[cfg])+1,np.median(EE[cfg,0:K_vec[cfg],kappa,:],axis=1)*1e-6,
                 '-ob',ms=9,mfc='white',mew=2,lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(EE_RtM[cfg,0:K_vec[cfg],kappa,:],axis=1)*1e-6,
                 '--og',ms=8,mfc='white',mew=2,lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(EE_RtS[cfg,0:K_vec[cfg],kappa,:],axis=1)*1e-6,
                 '--o',ms=8,mfc='white',mew=2,color='brown',lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(EE_AbW[cfg,0:K_vec[cfg],kappa,:],axis=1)*1e-6,
                 '-.or',ms=8,mfc='white',mew=2,lw=2)
        axs.plot(np.arange(K_vec[cfg])+1,np.median(EE_B27[cfg,0:K_vec[cfg],kappa,:],axis=1)*1e-6,
                 '-.oc',ms=8,mfc='white',mew=2,lw=2)
        axs.set_xlabel('No. of users, $K$')
        axs.set_ylabel('Energy efficiency [Mbit/J]')
        axs.grid(linestyle=':')
        if delta_PA_dtx == 0.25:
            axs.set_title('Load '+str(kappa+1)+', Config. '+config_vec[cfg]+
                          ',\n Enabled time-domain HW power-saving modes',fontsize=13)
        else:
            axs.set_title('Load '+str(kappa+1)+', Config. '+config_vec[cfg]+
                          ',\n Disabled time-domain HW power-saving modes',fontsize=13)
        plt.tight_layout()
        plt.show()
