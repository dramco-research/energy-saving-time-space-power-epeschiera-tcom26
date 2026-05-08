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
                   solve_problem0, solve_problem1, solve_problem2, solve_problem3, plot_sol)

class colors:
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    BOLD = '\033[1m'

# Function to compute the (continuous) power consumption
def f(x):
    return x[1]/x[0]*P_0/M + x[1]*P_1/M +\
           x[1]/x[0]*gamma*((1/(x[1]*(x[1]-K)))*sigma2/beta@(np.exp(Rpr*x[0])-1))**alpha

# Function to compute the transmit power at any active antenna
def Pa_function(na,ma):
    return 1/(ma*(ma-K))*sigma2/beta@(np.exp(Rpr*N/na)-1)

plot_flag = 0   # to plot the solution and constraints in the 2D space

config = '64T64R'                                   # BS configuration
# config = '2T2R'
# config = '4T4R'
# config = '8T8R'
# config = '32T32R'
K = 8                                               # Number of active users
T_frame = 10e-3                                     # Frame duration [s]
alpha = 0.75                                        # PA consumption exponent

# Comment one of the following two lines to select enabled/disabled time-domain hardware power-saving modes
# delta_PA_dtx = 1; delta_TRX_idle = 1
delta_PA_dtx = 0.25; delta_TRX_idle = 0.5

# Initialize parameters of power consumption model
gamma, P_0, P_1, P_sleep, B, P_max, M, K_max, T, tau_CP, tau_DL, tau_sig = \
    PC_model(config, delta_PA_dtx, delta_TRX_idle, K, alpha)
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

N_points = 20                                       # Number of points of the plots
P_cons = np.zeros(N_points+1)                       # Optimized (proposed) scheme
P_cons_AbW = np.zeros(N_points+1)                   # Awake-but-whisper scheme
P_cons_RtS = np.zeros(N_points+1)                   # Rush-to-sleep scheme
P_cons_RtM = np.zeros(N_points+1)                   # Rush-to-mute scheme
Ma_opt_vec = np.zeros(N_points+1)                   # Vector of optimized number of active antennas
Na_opt_vec = np.zeros(N_points+1)                   # Vector of optimized number of active time slots
R_sum = np.zeros(N_points+1)                        # Sum rate [bit/s/Hz]

if config == '8T8R' or config == '32T32R' or config == '64T64R':
    SNR = inv_eCDF_SNR5G(K)                         # SNR vector
else:
    SNR = inv_eCDF_SNR4G(K)
beta = (sigma2*SNR)/(P_T*(M-1))                     # Large-scale fading coefficient vector
rho = P_max*beta/sigma2                             # Single-antenna equivalent SNR
if np.sum(rho < 2/K) > 0:
    print(colors.RED+'Domain not convex'+colors.ENDC)
    print()
R_0 = np.random.uniform(0,1,K)                     # Baseline user rates [bit/subcarrier/OFDM symbol]
R_0 = R_0/np.sum(R_0)

def obj_kappamax(x):
    return np.abs(M-K/2-1/2*np.sqrt(K**2+4*rho**(-1)@(np.exp(x*R_0*np.log(2))-1)))
constrs = [{'type': 'ineq', 'fun': lambda x: x}]
# noinspection PyTypeChecker
result = minimize(obj_kappamax, np.array(1), constraints=constrs,
                  tol=1e-8, options={'disp': False, 'maxiter': 100})
kappa_max = result.x[0]                             # Maximum rate scaling
print('kappa_max =',kappa_max)


# Iterate over network load (kappa/kappa_max)
kappa_vec = np.zeros(N_points+1)
kappa_vec[1:N_points+1] = np.linspace(1e-6,1,N_points)*kappa_max
for n in range(N_points-1):
    kappa = kappa_vec[n+1]
    print(colors.YELLOW+'kappa/kappa_max = '+f'{kappa/kappa_max:.2f}'+colors.ENDC)
    R = kappa*R_0
    Rpr = R*np.log(2)

    Ma_min = int(np.ceil(K/2+1/2*np.sqrt(K**2+4*rho**(-1)@(np.exp(Rpr)-1))))
    Na_vec = np.arange(N)+1
    tmp = 0
    for k in range(K):
        tmp += rho[k]**(-1)*(np.exp(Rpr[k]*N/Na_vec)-1)
    Na_min = Na_vec[np.argmin(np.abs(M-K/2-1/2*np.sqrt(K**2+4*tmp)))]
    Pa_min = Pa_function(N,M)
    if Pa_min > 1.0001*P_max:
        print(colors.RED,'Problem not feasible!',colors.ENDC)
        break

    # Solve the problem with Algorithm 1
    [x,y] = solve_problem0(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
    if x >= 1 and y <= M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
        print('Proposed solver (unconstrained)')
    elif x < 1 and y <= M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
        x = 1
        y = solve_problem1(tol,max_it,Rpr,sigma2/beta,gamma,M,K,alpha,P_0,P_1)
        if y > M and y >= K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            y = M
        elif y <= M and y < K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1)):
            y = K/2+1/2*np.sqrt(K**2+4/P_max*sigma2/beta@(np.exp(Rpr*x)-1))
        print('Proposed solver (subcase R_1)')
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
        print('Proposed solver (subcase R_2)')
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
        print('Proposed solver (subcase R_3)')
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
        print('Proposed solver (subcase R_4)')
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
        print('Proposed solver (subcase R_5)')
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
        print('Proposed solver (subcase R_6)')

    if np.isnan(x) or np.isnan(y):
        continue
    Na = np.max([1,int(np.round(N/x))])
    Ma = np.max([K+1,int(np.round(y))])
    Pa = Pa_function(Na,Ma)

    print('Na \t\t=',colors.BLUE,f'{int(Na):03d}',colors.ENDC,'\t\t|\t\tN \t\t= ',N,
          '\nMa \t\t=',colors.GREEN,f'{int(Ma):02d}',colors.ENDC,'\t\t\t|\t\tM \t\t= ',M,
          '\nPa \t\t= ',f'{Pa:.2f}','\t\t|\t\tPmax \t= ',f'{P_max:.2f}',
          '\nP_cons \t= ',f'{f([N/Na,Ma])+P_sleep:.2f}')

    Na_opt_vec[n+1] = Na
    Ma_opt_vec[n+1] = Ma

    # Check that transmit power constraint is satisfied
    while Pa > P_max:
        if f([N/(Na+1),Ma]) < f([N/Na,Ma+1]) and Na < N:
            Na = Na+1
            Pa = Pa_function(Na,Ma)
        else:
            Ma = Ma+1
            Pa = Pa_function(Na,Ma)

    if plot_flag == 1:
        plot_sol(Na,Ma,x,y,M,K,rho,Rpr,N,P_max,beta,sigma2,Ma_min,kappa,kappa_max,R,config)

    P_cons[n+1] = f([N/Na,Ma])+P_sleep
    P_cons_AbW[n+1] = f([1,M])+P_sleep
    P_cons_RtS[n+1] = f([N/Na_min,M])+P_sleep
    P_cons_RtM[n+1] = f([1,Ma_min])+P_sleep

    # Brute-force approach as sanity check
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
    Ma_opt = np.max([K+1,Ma_opt])
    if np.isnan(Na_opt) or np.isnan(Ma_opt):
        continue
    Pa_opt = 1/(Ma_opt*(Ma_opt-K))*np.sum(sigma2/beta*(2**(R*N/Na_opt)-1))
    print('Brute force')
    print('Na \t\t=',colors.BLUE,f'{int(Na_opt):03d}',colors.ENDC,'\t\t|\t\tN \t\t= ',N,
          '\nMa \t\t=',colors.GREEN,f'{int(Ma_opt):02d}',colors.ENDC,'\t\t\t|\t\tM \t\t= ',M,
          '\nPa \t\t= ',f'{Pa_opt:.2f}','\t\t|\t\tPmax \t= ',f'{P_max:.2f}',
          '\nP_cons \t= ',f'{f([N/Na_opt,Ma_opt])+P_sleep:.2f}')

    print()

P_cons[0] = P_cons_AbW[0] = P_cons_RtS[0] = P_cons_RtM[0] = P_sleep
kappa = kappa_max
R = kappa*R_0
Rpr = R*np.log(2)
P_cons[-1] = P_cons_AbW[-1] = P_cons_RtS[-1] = P_cons_RtM[-1] = f([1,M])+P_sleep
Na_opt_vec[0] = 0
Ma_opt_vec[0] = 0
Na_opt_vec[-1] = N
Ma_opt_vec[-1] = M

# Plot results (power consumption vs. network load)
plt.figure()
plt.plot(kappa_vec/kappa_max,P_cons, '-b', lw=2.5, label='Optimized')
plt.plot(kappa_vec/kappa_max,P_cons_AbW, '--r', lw=2.5, label='Awake-but-whisper')
plt.plot(kappa_vec/kappa_max,P_cons_RtS, ':c', lw=2.5, label='Rush-to-sleep')
plt.plot(kappa_vec/kappa_max,P_cons_RtM, '-.g', lw=2.5, label='Rush-to-mute')
plt.xlabel('Network load, $\\kappa/\\kappa_\\mathrm{max}$')
plt.ylabel('$P_\\mathrm{cons}$ [W]')
# plt.xlim([0,1])
plt.grid(linestyle=':')
plt.legend(fontsize=13)
plt.title('Config: '+config+', $K =$ '+str(K)+', $\\delta_\\mathrm{PA}^\\mathrm{dtx} =$ '+
          str(np.round(delta_PA_dtx,2))+', $\\delta_\\mathrm{TRX}^\\mathrm{idle} =$ '+
          str(np.round(delta_TRX_idle,2))+', $N =$ '+str(int(N)),fontsize=13)
plt.tight_layout()
plt.show()

# Plot results (ratio of active resources vs. network load)
plt.figure()
plt.plot(kappa_vec/kappa_max,Na_opt_vec/N*100, '-b', lw=2, label='$N_\\mathrm{a}/N$')
plt.plot(kappa_vec/kappa_max,Ma_opt_vec/M*100, '--r', lw=2, label='$M_\\mathrm{a}/M$')
plt.xlabel('Network load, $\\kappa/\\kappa_\\mathrm{max}$')
plt.ylabel('Ratio of active resources [%]')
plt.grid(linestyle=':')
plt.legend(fontsize=13)
plt.title('Config: '+config+', $K =$ '+str(K)+', $\\delta_\\mathrm{PA}^\\mathrm{dtx} =$ '+
          str(np.round(delta_PA_dtx,2))+', $\\delta_\\mathrm{TRX}^\\mathrm{idle} =$ '+
          str(np.round(delta_TRX_idle,2))+', $N =$ '+str(int(N)),fontsize=13)
plt.tight_layout()
plt.show()
