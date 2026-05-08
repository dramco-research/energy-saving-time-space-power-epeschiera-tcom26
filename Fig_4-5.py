import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams.update({'font.size':13})
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
def f(x,alpha=0.75):
    return x[1]/x[0]*P_0/M+x[1]*P_1/M+\
           x[1]/x[0]*gamma*((1/(x[1]*(x[1]-K)))*sigma2/beta@(np.exp(Rpr*x[0])-1))**alpha

# Function to compute the (continuous) power consumption, used for evaluation
def f_eval(x,alpha=0.75):
    return x[1]/x[0]*P_0/M+x[1]*P_1/M+\
           x[1]/x[0]*gamma*((1/(x[1]*(x[1]-K)))*sigma2/beta@(np.exp(Rpr*x[0])-1))**alpha

# Function to compute the transmit power at any active antenna
def Pa_func(na,ma):
    return 1/(ma*(ma-K))*sigma2/beta@(np.exp(Rpr*N/na)-1)

# config = '64T64R'                                 # BS configuration
# config = '4T4R'
config = '8T8R'
K = 4                                               # Number of active users
T_frame = 10e-3                                     # Frame duration [s]
alpha = 0.75                                        # PA consumption exponent

# Comment one of the following two lines to select enabled/disabled time-domain hardware power-saving modes
# delta_PA_dtx = 1; delta_TRX_idle = 1
delta_PA_dtx = 0.25; delta_TRX_idle = 0.5

kappa_prc_vec = [1,6,18]                            # Vector of kappa [%]

# Initialize parameters of power consumption model
gamma, P_0, P_1, P_sleep, B, P_max, M, K_max, T, tau_CP, tau_DL, tau_sig = \
    PC_model(config, delta_PA_dtx, delta_TRX_idle, K, 0.75)
gamma_pr, P_0_pr, P_1_pr, _, _, _, _, _, _, _, _, _ = \
    PC_model(config, delta_PA_dtx, delta_TRX_idle, K, 1)
N = np.floor((tau_DL-tau_sig*tau_DL)*T_frame/T)     # Number of time slots (OFDM symbols)
if config == '64T64R' or config == '32T32R':
    P_T = 20                                        # Total transmit power at BS [W]
elif config == '8T8R':
    P_T = 32
else:
    P_T = 160
NF_dB = 9           					            # Noise factor [dB]
NT = 300            					            # Noise temperature [K]
kB = 1.380649e-23   					            # Boltzmann constant [W/K/Hz]
sigma2 = NT*kB*B*10**(NF_dB/10) 		            # Noise power [W]
tol = 1e-8                                          # Tolerance of Newton's method
max_it = 1e3                                        # Maximum number of iterations of Newton's method

plot_GT = 1   # to plot the ground truth (optimal solution)

N_Iter = int(1e3)                                   # Number of iterations (channel realization)
# To store the results
P_cons = np.zeros([len(kappa_prc_vec),N_Iter])      # Optimized (proposed) scheme
P_cons_AbW = np.zeros([len(kappa_prc_vec),N_Iter])  # Awake-but-whisper scheme
P_cons_RtS = np.zeros([len(kappa_prc_vec),N_Iter])  # Rush-to-sleep scheme
P_cons_RtM = np.zeros([len(kappa_prc_vec),N_Iter])  # Rush-to-mute scheme
P_cons_B27 = np.zeros([len(kappa_prc_vec),N_Iter])  # Benchmark [27] scheme
P_cons_opt = np.zeros([len(kappa_prc_vec),N_Iter])  # Optimal scheme

R_stats = np.zeros([len(kappa_prc_vec),N_Iter*K])   # Collect rates


# Iterate over network load
for i in range(len(kappa_prc_vec)):
    kappa_prc = kappa_prc_vec[i]

    n = 0
    while n < N_Iter:
        if np.mod(n,100) == 0:
            print('Load',i+1,', iter',n,'/',N_Iter)

        if config == '8T8R' or config == '32T32R' or config == '64T64R':
            SNR = inv_eCDF_SNR5G(K)                         # SNR vector
        else:
            SNR = inv_eCDF_SNR4G(K)
        beta = (sigma2*SNR)/(P_T*(M-1))                     # Large-scale fading coefficient vector
        rho = P_max*beta/sigma2                             # Single-antenna equivalent SNR

        R_0 = np.random.uniform(0,1,K)                      # Baseline user rates [bit/subcarrier/OFDM symbol]
        R_0 = R_0/np.sum(R_0)

        def obj_kappamax(x):
            return np.abs(M-K/2-1/2*np.sqrt(K**2+4*rho**(-1)@(np.exp(x*R_0*np.log(2))-1)))
        constrs = [{'type': 'ineq', 'fun': lambda x: x}]
        # noinspection PyTypeChecker
        result = minimize(obj_kappamax, np.array(1), constraints=constrs,
                          tol=1e-8, options={'disp': False, 'maxiter': 100})
        kappa_max = result.x[0]                             # Maximum rate scaling

        kappa = (kappa_prc/100)*kappa_max
        R = kappa*R_0
        R_stats[i,n*K:(n+1)*K] = R
        Rpr = R*np.log(2)

        Ma_min = int(np.ceil(K/2+1/2*np.sqrt(K**2+4*rho**(-1)@(np.exp(Rpr)-1))))
        Na_vec = np.arange(N) + 1
        tmp = 0
        for k in range(K):
            tmp += rho[k]**(-1)*(np.exp(Rpr[k]*N/Na_vec)-1)
        Na_min = Na_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
        Pa_min = Pa_func(N,M)
        if Pa_min > 1.01 * P_max:
            print('Problem not feasible!')
            continue

        # Solve the problem with Algorithm 1
        [x,y] = solve_problem0(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
        if x >= 1 and y <= M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            pass
        elif x < 1 and y <= M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            x = 1
            y = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
            if y > M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                y = M
            elif y <= M and y < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                y = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
        elif x < 1 and y > M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            x1 = 1
            y1 = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
            if y1 > M and y1 >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                y1 = M
            elif y1 <= M and y1 < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                y1 = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta.T@(np.exp(Rpr*x1)-1))
            tmp1 = f([1,y1])
            y2 = M
            x2 = solve_problem2(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
            if x2 < 1 and y2 >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1)):
                x2 = 1
            elif x2 >= 1 and y2 < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1)):
                x_vec = np.linspace(1,1e3,int(1e5))
                tmp = 0
                for k in range(K):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                x2 = x_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
            tmp2 = f([x2,M])
            if tmp1 <= tmp2:
                x = 1
                y = y1
            else:
                x = x2
                y = M
        elif x < 1 and y <= M and y < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            x1 = 1
            y1 = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
            if y1 > M and y1 >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                y1 = M
            elif y1 <= M and y1 < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                y1 = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1))
            tmp1 = f([x1,y1])
            x2 = solve_problem3(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1,P_max)
            y2 = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1))
            if x2 < 1 and y2 <= M:
                x2 = 1
            elif x2 >= 1 and y2 > M:
                x_vec = np.linspace(1,1e3,int(1e5))
                tmp = 0
                for k in range(K):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                x2 = x_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
                y2 = M
            tmp2 = f([x2,y2])
            if tmp1 <= tmp2:
                x = 1
                y = y1
            else:
                x = x2
                y = y2
        elif x >= 1 and y > M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            y = M
            x = solve_problem2(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
            if x < 1 and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                x = 1
            elif x >= 1 and y < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
                x_vec = np.linspace(1,1e3,int(1e5))
                tmp = 0
                for k in range(K):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                x = x_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
        elif x >= 1 and y <= M and y < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            x = solve_problem3(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1,P_max)
            y = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
            if x < 1 and (y <= M or np.isnan(y)):
                x = 1
                y = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
            elif x >= 1 and y > M:
                y = M
                x_vec = np.linspace(1,1e3,int(1e5))
                tmp = 0
                for k in range(K):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                x = x_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
        elif x >= 1 and y > M and y < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            y1 = M
            x1 = solve_problem2(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
            if x1 < 1 and y1 >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                x1 = 1
            elif x1 >= 1 and y1 < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x1)-1)):
                x_vec = np.linspace(1,1e3,int(1e5))
                tmp = 0
                for k in range(K):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                x1 = x_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
            tmp1 = f([x1,y1])
            x2 = solve_problem3(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1,P_max)
            y2 = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x2)-1))
            if x2 < 1 and y2 <= M:
                x2 = 1
            elif x2 >= 1 and y2 > M:
                y2 = M
                x_vec = np.linspace(1,1e3,int(1e5))
                tmp = 0
                for k in range(K):
                    tmp += rho[k]**(-1)*(np.exp(Rpr[k]*x_vec)-1)
                x2 = x_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
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
        Ma = np.max([K+1,int(np.round(y))])
        Pa = Pa_func(Na,Ma)

        # Check that transmit power constraint is satisfied
        while Pa > P_max:
            if f([N/(Na+1),Ma]) < f([N/Na,Ma+1]) and Na < N:
                Na = Na+1
                Pa = Pa_func(Na,Ma)
            else:
                Ma = Ma+1
                Pa = Pa_func(Na,Ma)

        if plot_GT == 1:
            f_opt = np.inf
            Ma_opt = 0
            Na_opt = 0
            Na_vec = np.arange(Na_min,N+1)
            Ma_vec = np.arange(Ma_min,M+1)
            for na in range(len(Na_vec)):
                for ma in range(len(Ma_vec)):
                    Naa = Na_vec[na]
                    Maa = Ma_vec[ma]
                    Paa = 1/(Maa*(Maa-K))*np.sum(sigma2/beta*(2**(R*N/Naa)-1))
                    if Paa <= P_max and f([N/Naa,Maa]) < f_opt:
                        Na_opt = Naa
                        Ma_opt = Maa
                        f_opt = f([N/Naa,Maa])
            Ma_opt = np.max([K+1,Ma])
            P_cons_opt[i,n] = f([N/Na_opt,Ma_opt])+P_sleep

        # Compute solution of benchmark [27]
        x_B27 = 1
        y_B27 = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma_pr,M,K,1,P_0_pr,P_1_pr)
        if y_B27 > M and y_B27 >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x_B27)-1)):
            y_B27 = M
        elif y_B27 <= M and y_B27 < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x_B27)-1)):
            y_B27 = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x_B27)-1))
        Ma_B27 = np.max([K+1,int(np.round(y_B27))])
        while Pa_func(N,Ma_B27) > P_max:
                Ma_B27 = Ma_B27+1
        Ma_B27 = np.max([K+1,int(np.round(y_B27))])
        P_cons_B27[i,n] = f_eval([1,Ma_B27])+P_sleep

        # Save results
        P_cons[i,n] = f_eval([N/Na,Ma])+P_sleep
        P_cons_AbW[i,n] = f_eval([1,M])+P_sleep
        P_cons_RtS[i,n] = f_eval([N/Na_min,M])+P_sleep
        P_cons_RtM[i,n] = f_eval([1,Ma_min])+P_sleep

        n += 1


# Plot results (CDF of power consumption for 3 BS configurations and 3 network loads)
sub = 1     # subsampling factor
N_Iter = N_Iter/sub
fig, axs = plt.subplots(3,1,gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 1},constrained_layout=True)
plt.subplots_adjust(top=0.82)
axs[0].plot(np.sort(P_cons[0,::sub]),np.arange(N_Iter)/N_Iter, '-b', lw=2.5, label='Optimized',clip_on=False)
if plot_GT == 1:
    axs[0].plot(np.sort(P_cons_opt)[0,::165],(np.arange(N_Iter)/N_Iter)[::166], 'xk', alpha=1, lw=2.5, label='Optimal',clip_on=False)
axs[0].plot(np.sort(P_cons_AbW[0,::sub]),np.arange(N_Iter)/N_Iter, '--r', lw=2.5, label='Awake-but-whisper',clip_on=False)
axs[0].plot(np.sort(P_cons_RtS[0,::sub]),np.arange(N_Iter)/N_Iter, ':', lw=2.5, color='cyan', label='Rush-to-sleep',clip_on=False)
axs[0].plot(np.sort(P_cons_RtM[0,::sub]),np.arange(N_Iter)/N_Iter, '-.g', lw=2.5, label='Rush-to-mute',clip_on=False)
axs[0].plot(np.sort(P_cons_B27[0,::sub]),np.arange(N_Iter)/N_Iter, '-.', lw=2.5, color='brown', label='Benchmark [27]',clip_on=False)
axs[0].xaxis.set_label_coords(0.5,-0.35)
axs[0].set_xlabel('$P_\\mathrm{cons}$ [W]')
axs[0].set_ylabel('CDF')
axs[0].set_xlim([np.min(P_cons)-0.02*np.min(P_cons),np.max([P_cons,P_cons_AbW,P_cons_RtS,P_cons_RtM])+
                 0.02*np.max([P_cons,P_cons_AbW,P_cons_RtS,P_cons_RtM])])
axs[0].set_ylim([0,1])
axs[0].grid(linestyle=':')
axs[0].set_title('Low load ($\\kappa/\\kappa_\\mathrm{max} =$ '+str(kappa_prc_vec[0]/100)+')',fontsize=13)
axs[1].plot(np.sort(P_cons[1,::sub]),np.arange(N_Iter)/N_Iter, '-b', lw=2.5, label='Optimized',clip_on=False)
if plot_GT == 1:
    axs[1].plot(np.sort(P_cons_opt)[1,::165],(np.arange(N_Iter)/N_Iter)[::166], 'xk', alpha=1, lw=2.5, label='Optimal',clip_on=False)
axs[1].plot(np.sort(P_cons_AbW[1,::sub]),np.arange(N_Iter)/N_Iter, '--r', lw=2.5, label='Awake-but-whisper',clip_on=False)
axs[1].plot(np.sort(P_cons_RtS[1,::sub]),np.arange(N_Iter)/N_Iter, ':', lw=2.5, color='cyan', label='Rush-to-sleep',clip_on=False)
axs[1].plot(np.sort(P_cons_RtM[1,::sub]),np.arange(N_Iter)/N_Iter, '-.g', lw=2.5, label='Rush-to-mute',clip_on=False)
axs[1].plot(np.sort(P_cons_B27[1,::sub]),np.arange(N_Iter)/N_Iter, '-.', lw=2.5, color='brown', label='Benchmark [27]',clip_on=False)
axs[1].xaxis.set_label_coords(0.5,-0.35)
axs[1].set_xlabel('$P_\\mathrm{cons}$ [W]')
axs[1].set_ylabel('CDF')
axs[1].set_xlim([np.min(P_cons)-0.02*np.min(P_cons),np.max([P_cons,P_cons_AbW,P_cons_RtS,P_cons_RtM])+
                 0.02*np.max([P_cons,P_cons_AbW,P_cons_RtS,P_cons_RtM])])
axs[1].set_ylim([0,1])
axs[1].grid(linestyle=':')
axs[1].set_title('Medium load ($\\kappa/\\kappa_\\mathrm{max} =$ '+str(kappa_prc_vec[1]/100)+')',fontsize=13)
axs[2].plot(np.sort(P_cons[2,::sub]),np.arange(N_Iter)/N_Iter, '-b', lw=2.5, label='Optimized',clip_on=False)
if plot_GT == 1:
    axs[2].plot(np.sort(P_cons_opt)[2,::165],(np.arange(N_Iter)/N_Iter)[::166], 'xk', alpha=1, lw=2.5, label='Optimal',clip_on=False)
axs[2].plot(np.sort(P_cons_AbW[2,::sub]),np.arange(N_Iter)/N_Iter, '--r', lw=2.5, label='Awake-but-whisper',clip_on=False)
axs[2].plot(np.sort(P_cons_RtS[2,::sub]),np.arange(N_Iter)/N_Iter, ':', lw=2.5, color='cyan', label='Rush-to-sleep',clip_on=False)
axs[2].plot(np.sort(P_cons_RtM[2,::sub]),np.arange(N_Iter)/N_Iter, '-.g', lw=2.5, label='Rush-to-mute',clip_on=False)
axs[2].plot(np.sort(P_cons_B27[2,::sub]),np.arange(N_Iter)/N_Iter, '-.', lw=2.5, color='brown', label='Benchmark [27]',clip_on=False)
axs[2].xaxis.set_label_coords(0.5,-0.35)
axs[2].set_xlabel('$P_\\mathrm{cons}$ [W]')
axs[2].set_ylabel('CDF')
axs[2].set_xlim([np.min(P_cons)-0.02*np.min(P_cons),np.max([P_cons,P_cons_AbW,P_cons_RtS,P_cons_RtM])+
                 0.02*np.max([P_cons,P_cons_AbW,P_cons_RtS,P_cons_RtM])])
axs[2].set_ylim([0,1])
axs[2].grid(linestyle=':')
axs[2].set_title('High load ($\\kappa/\\kappa_\\mathrm{max} =$ '+str(kappa_prc_vec[2]/100)+')',fontsize=13)
print('Config: '+config+', delta_PA_dtx = '+str(np.round(delta_PA_dtx,2))+
          ', delta_TRX_idle = '+str(np.round(delta_TRX_idle,2))+
          ', N = '+str(int(N)))
if plot_GT == 1:
    fig.legend(
        handles=[
            plt.Line2D([0], [0], lw=2.5, linestyle='None', marker='x', color='black', label='Optimal'),
            plt.Line2D([0], [0], lw=2.5, linestyle='-', color='blue', label='Optimized'),
            plt.Line2D([0], [0], lw=2.5, linestyle='--', color='red', label='Awake-but-whisper'),
            plt.Line2D([0], [0], lw=2.5, linestyle=':', color='cyan', label='Rush-to-sleep'),
            plt.Line2D([0], [0], lw=2.5, linestyle='-.', color='green', label='Rush-to-mute'),
            plt.Line2D([0], [0], lw=2.5, linestyle='-.', color='brown', label='Benchmark [27]'),
        ],
        loc='upper center',bbox_to_anchor=(0.5, 1.01),ncols=3,frameon=False,columnspacing=0.6,fontsize=8
    )
else:
    fig.legend(
        handles=[
            plt.Line2D([0], [0], lw=2.5, linestyle='-', color='blue', label='Optimized'),
            plt.Line2D([0], [0], lw=2.5, linestyle='--', color='red', label='Awake-but-whisper'),
            plt.Line2D([0], [0], lw=2.5, linestyle=':', color='cyan', label='Rush-to-sleep'),
            plt.Line2D([0], [0], lw=2.5, linestyle='-.', color='green', label='Rush-to-mute'),
            plt.Line2D([0], [0], lw=2.5, linestyle='-.', color='brown', label='Benchmark [27]'),
        ],
        loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, columnspacing=0.6, fontsize=8
    )
plt.tight_layout()
plt.show()

print('Median R_k_deliv at load 1:',np.round(B*(1-tau_CP)*tau_DL*(1-tau_sig)*np.median(R_stats[0,:])*1e-6,1),
      'Mbit/s')
print('Median R_k_deliv at load 2:',np.round(B*(1-tau_CP)*tau_DL*(1-tau_sig)*np.median(R_stats[1,:])*1e-6,2),
      'Mbit/s')
print('Median R_k_deliv at load 3:',np.round(B*(1-tau_CP)*tau_DL*(1-tau_sig)*np.median(R_stats[2,:])*1e-6,1),
      'Mbit/s')
