import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function that initializes the power consumption model
def PC_model(config,delta_PA_dtx,delta_TRX_idle,K,alpha):

    def P_PA(p):
        return params_PA["xi"]*(P_max/eta_PA_max)+\
               (1-params_PA["xi"])*(P_max**(1-params_PA["alpha"])*p**params_PA["alpha"]/eta_PA_max)

    # Numerical estimates of model parameters
    # Ref
    params_Ref = {
        "fstar": 1e9,
        "Bstar": 1e6,
        "Pstar": 1
    }
    # PA
    params_PA = {
        "xi": 0.12,
        "alpha": alpha,
        "eta_PA_tech": 0.38,
        "a": 1e-4,
        "b": 1,
        "delta_PA_idle": 0.25,
        "delta_PA_dtx": delta_PA_dtx,
        "delta_PA_sleep": 0
    }
    # AFE
    params_AFE = {
        "alpha_11": 0,
        "alpha_12": 0.25,
        "delta_misc_sleep": 0.75,
        "P_TRX_ref": 0.18,
        "alpha_21": 0.5,
        "alpha_22": 0.5,
        "delta_TRX_idle": delta_TRX_idle,
        "delta_TRX_sleep": 0.5
    }
    if config == "32T32R" or config == "64T64R":
        params_AFE["P_misc_ref"] = 17
    else:
        params_AFE["P_misc_ref"] = 22
    # DBB
    params_DBB = {
        "P_link_ref": 90,
        "P_phy_ref": 0.07,
        "alpha_3": 0.5,
        "alpha_4": 1,
        "delta_phy_idle": 0.75,
        "delta_phy_sleep": 0.75
    }
    # PSC
    params_PSC = {
        "eta_supply": 0.9,
        "eta_cool_PA": 0.9,
        "eta_cool_AFE": 0.9,
        "eta_cool_DBB": 0.95
    }
    # Protocol
    if config == "8T8R" or config == "32T32R" or config == "64T64R":
        params_Protocol = {
            "tau_DL": 0.75,
            "tau_UL": 0.25,
            "tau_sig": 1/14,
            "zeta_sig": 1/12,
            "T": 35.6e-6,
            "Delta_f": 30e3,
            "tau_CP": 1-1/(35.6e-6*30e3)
        }
    else:
        params_Protocol = {
            "tau_DL": 1,
            "tau_UL": 1,
            "tau_sig": 2/7,
            "zeta_sig": 2/6,
            "T": 71.4e-6,
            "Delta_f": 15e3,
            "tau_CP": 1-1/(71.4e-6*15e3)
        }
    # BS configuration parameters
    if config == "micro-4T4R":
        f = 1.8e9
        B = 20e6
        P_max = 5
        M = 4
        K_max = 2
    elif config == "multi-2T2R":
        f = 0.8e9
        B = 20e6
        P_max = 80
        M = 2
        K_max = 1
    elif config == "2T2R":
        f = 1.8e9
        B = 20e6
        P_max = 80
        M = 2
        K_max = 1
    elif config == "4T4R":
        f = 1.8e9
        B = 20e6
        P_max = 40
        M = 4
        K_max = 2
    elif config == "8T8R":
        f = 3.5e9
        B = 100e6
        P_max = 40
        M = 8
        K_max = 4
    elif config == "32T32R":
        f = 3.5e9
        B = 100e6
        P_max = 6.25
        M = 32
        K_max = 8
    else: # config == "64T64R":
        f = 3.5e9
        B = 100e6
        P_max = 3.125
        M = 64
        K_max = 8

    eta_PA_max = params_PA["eta_PA_tech"]/(1+params_PA["a"]*(P_max/params_Ref["Pstar"])*
                 ((f/params_Ref["fstar"])**2)+params_PA["b"]*(params_Ref["Pstar"]/P_max))
    P_misc = params_AFE["P_misc_ref"]*((B/params_Ref["Bstar"])**params_AFE["alpha_11"])*\
             (((M*P_max)/params_Ref["Pstar"])**params_AFE["alpha_12"])
    P_TRX = params_AFE["P_TRX_ref"]*((B/params_Ref["Bstar"])**params_AFE["alpha_21"])*\
            ((P_max/params_Ref["Pstar"])**params_AFE["alpha_22"])

    gamma = (1/(params_PSC["eta_supply"]*params_PSC["eta_cool_PA"]))*params_Protocol["tau_DL"]\
        *(1-params_Protocol["tau_sig"])*(P_PA(1)-P_PA(0))
    P_0 = (M*(
        (1/(params_PSC["eta_supply"]*params_PSC["eta_cool_PA"]))*params_Protocol["tau_DL"]*
            (1-params_Protocol["tau_sig"])*P_PA(0)*(
        (1-params_PA["delta_PA_dtx"]))))
    P_1 = (M*(
        (1/(params_PSC["eta_supply"]*params_PSC["eta_cool_PA"]))*(
            params_Protocol["tau_sig"]*params_Protocol["tau_DL"]*P_PA(params_Protocol["zeta_sig"]*P_max)+
            (1-params_Protocol["tau_sig"])*params_Protocol["tau_DL"]*P_PA(0)*params_PA["delta_PA_dtx"]+
            (1-params_Protocol["tau_DL"])*P_PA(0)*params_PA["delta_PA_idle"]-
            P_PA(0)*params_PA["delta_PA_sleep"])+
        (1/(params_PSC["eta_supply"]*params_PSC["eta_cool_AFE"]))*(
            P_TRX*(params_Protocol["tau_DL"]+params_Protocol["tau_UL"]+
            (2-params_Protocol["tau_DL"]-params_Protocol["tau_UL"])*params_AFE["delta_TRX_idle"]-
            2*params_AFE["delta_TRX_sleep"]))))
    P_sleep = ((1/(params_PSC["eta_supply"]*params_PSC["eta_cool_PA"]))*(
            M*params_PA["xi"]*(P_max/eta_PA_max)*params_PA["delta_PA_sleep"])+
        (1/(params_PSC["eta_supply"]*params_PSC["eta_cool_AFE"]))*(
            P_misc+2*M*P_TRX*params_AFE["delta_TRX_sleep"])+
        (1/(params_PSC["eta_supply"]*params_PSC["eta_cool_DBB"]))*(
            params_DBB["P_link_ref"]+params_DBB["P_phy_ref"]*((B/params_Ref["Bstar"])**params_DBB["alpha_4"])*
            (K*(params_Protocol["tau_DL"]+params_Protocol["tau_UL"]+
            (2-params_Protocol["tau_DL"]-params_Protocol["tau_UL"])*params_DBB["delta_phy_idle"])+
             2*(K_max-K)*params_DBB["delta_phy_sleep"])))

    return (gamma, P_0, P_1, P_sleep, B, P_max, M, K_max,
            params_Protocol["T"], params_Protocol["tau_CP"], params_Protocol["tau_DL"], params_Protocol["tau_sig"])

# Load SNR and its CDF (for 5G BSs)
SNR5G = np.load('Data/SNR5G.npy')
eCDF_SNR5G = np.load('Data/eCDF_SNR5G.npy')
# Function that samples the CDF of 5G SNR
def inv_eCDF_SNR5G(K):
    u = np.random.uniform(0,1,K)
    SNR_dB = np.zeros(K)
    for i in range(K):
        pos = np.argmin(np.abs(u[i]-eCDF_SNR5G))
        pos2 = pos+int(np.sign(u[i]-eCDF_SNR5G[pos]))
        dist = np.abs(eCDF_SNR5G[pos]-eCDF_SNR5G[pos2])
        SNR_dB[i] = (((dist-np.abs(u[i]-eCDF_SNR5G[pos]))*SNR5G[pos]+
                      (dist-np.abs(u[i]-eCDF_SNR5G[pos2]))*SNR5G[pos2])/dist)
    return 10**(SNR_dB/10)
# Load SNR and its CDF (for 4G BSs)
SNR4G = np.load('Data/SNR4G.npy')
# Function that samples the CDF of 5G SNR
eCDF_SNR4G = np.load('Data/eCDF_SNR4G.npy')
def inv_eCDF_SNR4G(K):
    u = np.random.uniform(0,1,K)
    SNR_dB = np.zeros(K)
    for i in range(K):
        pos = np.argmin(np.abs(u[i]-eCDF_SNR4G))
        pos2 = pos+int(np.sign(u[i]-eCDF_SNR4G[pos]))
        dist = np.abs(eCDF_SNR4G[pos]-eCDF_SNR4G[pos2])
        SNR_dB[i] = (((dist-np.abs(u[i]-eCDF_SNR4G[pos]))*SNR4G[pos]+
                      (dist-np.abs(u[i]-eCDF_SNR4G[pos2]))*SNR4G[pos2])/dist)
    return 10**(SNR_dB/10)

# Function that computes the gradient of function f(x,y)
def f_grad(x,y,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    D_Rpr = np.diag(Rpr)
    phi = zeta@(np.exp(Rpr*x)-1)
    phipr = zeta@D_Rpr@np.exp(Rpr*x)
    return [-y/x**2*P_0/M+(y*gamma)/((y*(y-K))**alpha)*(alpha*phi**(alpha-1)*phipr*x-phi**alpha)/(x**2),
            P_0/x/M+gamma/x*phi**alpha*(1/(y**alpha*(y-K)**(1+alpha)))*(y*(1-2*alpha)-K*(1-alpha))+P_1/M]
# Function that computes the Hessian of function f(x,y)
def f_hess(x,y,Rpr,zeta,gamma,M,K,alpha,P_0):
    D_Rpr = np.diag(Rpr)
    phi = zeta@(np.exp(Rpr*x)-1)
    phipr = zeta@D_Rpr@np.exp(Rpr*x)
    phiprpr = zeta@D_Rpr**2@np.exp(Rpr*x)
    return [[2*y/x**3*P_0/M+(y*gamma)/((y*(y-K))**alpha)*(1/x**3)*(x*(alpha*(alpha-1)*
            phi**(alpha-2)*phipr**2*x+alpha*phi**(alpha-1)*phiprpr*x)-
            2*(alpha*phi**(alpha-1)*phipr*x-phi**alpha)),
            -P_0/x**2/M+gamma*1/(y**alpha*(y-K)**(1+alpha))*(y*(1-2*alpha)-K*(1-alpha))*
            (alpha*phi**(alpha-1)*phipr*x-phi**alpha)/(x**2)],
            [-P_0/x**2/M+gamma*1/(y**alpha*(y-K)**(1+alpha))*(y*(1-2*alpha)-K*(1-alpha))*
            (alpha*phi**(alpha-1)*phipr*x-phi**alpha)/(x**2),
            gamma/x*phi**alpha*(alpha/(y**(1+alpha)*(y-K)**(2+alpha)))*
             (-K**2*(1-alpha)+4*K*y*(1-alpha)+2*y**2*(2*alpha-1))]]
# Function that computes the function f_1(y)
def f1(y,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    Ptilde = zeta@(np.exp(Rpr)-1)
    return P_0/M*y+gamma*(Ptilde**alpha)*(y**(1-alpha))/((y-K)**alpha)+P_1/M*y
# Function that computes the gradient of function f_1(y)
def f1_grad(y,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    Ptilde = zeta@(np.exp(Rpr)-1)
    return P_0/M+gamma*(Ptilde**alpha)*(1/(y**alpha*(y-K)**(1+alpha)))*(y*(1-2*alpha)-K*(1-alpha))+P_1/M
# Function that computes the Hessian of function f_1(y)
def f1_hess(y,Rpr,zeta,gamma,K,alpha):
    Ptilde = zeta@(np.exp(Rpr)-1)
    return (gamma*(Ptilde**alpha)*(alpha/(y**(1+alpha)*(y-K)**(2+alpha)))*
            (-K**2*(1-alpha)+4*K*y*(1-alpha)+2*y**2*(2*alpha-1)))
# Function that computes the gradient of function f_2(x)
def f2(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    phi = zeta@(np.exp(Rpr*x)-1)
    return P_0/x+(M**(1-alpha))/((M-K)**alpha)*gamma*(phi**alpha)/x+P_1/x
# Function that computes the gradient of function f_2(x)
def f2_grad(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    D_Rpr = np.diag(Rpr)
    phi = zeta@(np.exp(Rpr*x)-1)
    phipr = zeta@D_Rpr@np.exp(Rpr*x)
    return -P_0/x**2+(M**(1-alpha))/((M-K)**alpha)*gamma*(alpha*phi**(alpha-1)*phipr*x-phi**alpha)/(x**2)-P_1/x**2
# Function that computes the Hessian of function f_2(x)
def f2_hess(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    D_Rpr = np.diag(Rpr)
    phi = zeta@(np.exp(Rpr*x)-1)
    phipr = zeta@D_Rpr@np.exp(Rpr*x)
    phiprpr = zeta@D_Rpr**2@np.exp(Rpr*x)
    return (2*P_0/x**3+(M**(1-alpha))/((M-K)**alpha)*gamma*(1/x**3)*(x*(alpha*(alpha-1)*
            phi**(alpha-2)*phipr**2*x+alpha*phi**(alpha-1)*(phiprpr*x+phipr)-
            alpha*phi**(alpha-1)*phipr)-2*(alpha*phi**(alpha-1)*phipr*x-phi**alpha))+2*P_1/x**3)
# Function that computes the function f_3(x)
def f3(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max):
    phi = zeta@(np.exp(Rpr*x)-1)
    l = K**2+4/P_max*phi
    delta = P_0/M+gamma*P_max**alpha
    return (K/2+1/2*np.sqrt(l))*(delta/x+P_1/M)
# Function that computes the gradient of function f_3(x)
def f3_grad(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max):
    D_Rpr = np.diag(Rpr)
    phi = zeta@(np.exp(Rpr*x)-1)
    phipr = zeta@D_Rpr@np.exp(Rpr*x)
    l = K**2+4/P_max*phi
    delta = P_0/M+gamma*P_max**alpha
    return -1/x**2*(K*delta)+2/(P_max*np.sqrt(l))*phipr*(delta/x+P_1/M)-delta/x**2*np.sqrt(l)
# Function that computes the Hessian of function f_3(x)
def f3_hess(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max):
    D_Rpr = np.diag(Rpr)
    phi = zeta@(np.exp(Rpr*x)-1)
    phipr = zeta@D_Rpr@np.exp(Rpr*x)
    phiprpr = zeta@D_Rpr**2@np.exp(Rpr*x)
    l = K**2+4/P_max*phi
    delta = P_0/M+gamma*P_max**alpha
    return (2/x**3*(K*delta)+2*(np.sqrt(l)*phiprpr-2*l**(-1/2)*phipr**2/P_max)/(P_max*l)*(delta/x+P_1/M)-
            2*phipr/(P_max*np.sqrt(l))*(delta/x**2)-delta*(2*l**(-1/2)*x**2*phipr/P_max-2*np.sqrt(l)*x)/(x**4))

# Function that solves problem 0
def solve_problem0(tol,max_it,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    x_0 = 1.001
    y_0 = K+0.001
    x_curr = np.copy(x_0)
    y_curr = np.copy(y_0)
    x = 0
    y = 0
    lambda2 = np.inf
    it = 0
    while lambda2/2 > tol and it < max_it:
        t = 1
        deltax = -(la.inv(f_hess(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0))@
                   f_grad(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1))[0]
        deltay = -(la.inv(f_hess(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0))@
                   f_grad(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1))[1]
        x = x_curr + t*deltax
        y = y_curr + t*deltay
        if y > M*100:
            y = M+0.1
            break
        if x < 0:
            x = 0.00001
            break
        [x_curr,y_curr] = [np.copy(x),np.copy(y)]
        lambda2 = f_grad(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1)@\
                  la.inv(f_hess(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0))@\
                  f_grad(x_curr,y_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1)
        it += 1
    return x, y
# Function that solves problem 1
def solve_problem1(tol,max_it,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    y_0 = K+0.001
    y_curr = np.copy(y_0)
    y = 0
    lambda2 = np.inf
    it = 0
    while lambda2/2 > tol and it < max_it:
        t = 1
        deltax = -((f1_hess(y_curr,Rpr,zeta,gamma,K,alpha))**(-1)*
                   f1_grad(y_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1))
        y = y_curr + t*deltax
        y_curr = np.copy(y)
        lambda2 = f1_grad(y_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1)**2*\
                  f1_hess(y_curr,Rpr,zeta,gamma,K,alpha)**(-1)
        it += 1
    return y
# Function that solves problem 2
def solve_problem2(tol,max_it,Rpr,zeta,gamma,M,K,alpha,P_0,P_1):
    x_0 = 1.001
    x_curr = np.copy(x_0)
    x = 0
    lambda2 = np.inf
    it = 0
    while lambda2/2 > tol and it < max_it:
        t = 1
        deltax = -((f2_hess(x_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1))**(-1)*
                   f2_grad(x_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1))
        x = x_curr + t*deltax
        x_curr = np.copy(x)
        if x_curr < 0:
            break
        lambda2 = f2_grad(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1)**2*\
                  f2_hess(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1)**(-1)
        it += 1
    return x
def solve_problem3(tol,max_it,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max):
    x_0 = 1.001
    x_curr = np.copy(x_0)
    x = 0
    lambda2 = np.inf
    it = 0
    while lambda2/2 > tol and it < max_it:
        t = 1
        deltax = -((f3_hess(x_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max))**(-1)*
                   f3_grad(x_curr,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max))
        x = x_curr + t*deltax
        x_curr = np.copy(x)
        if x_curr < 0:
            break
        lambda2 = f3_grad(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max)**2*\
                  f3_hess(x,Rpr,zeta,gamma,M,K,alpha,P_0,P_1,P_max)**(-1)
        it += 1
    return x

# Function that plots the 2D solution
def plot_sol(Na,Ma,Na_opt,Ma_opt,x_unc,y_unc,x,y,M,K,rho,Rpr,N,P_max,beta,sigma2,Ma_min,kappa,kappa_max,R,config):
    def obj_xmax(x):
        return np.abs(M-K/2-1/2*np.sqrt(K**2+4*rho**(-1)@(np.exp(Rpr*x)-1)))
    constrs = [{'type': 'ineq', 'fun': lambda x: x-1}]
    # noinspection PyTypeChecker
    result = minimize(obj_xmax, np.array([N-1]), constraints=constrs, tol=1e-8, options={'disp': False, 'maxiter': 100})
    xmax = result.x[0]
    xax = np.linspace(1,xmax,int(1e4))
    tmp = 0
    for k in range(K):
        tmp += 4*(2**(R[k]*xax)-1)/(K**2*P_max*beta[k]/sigma2)
    PAPC = K/2+K/2*np.sqrt(1+tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(y=M,color='k',alpha=1,linestyle=':',label=None)
    ax.axvline(x=1,color='b',alpha=1,linestyle='--',label=None)
    ax.plot(xax,PAPC,'-g')
    ax.plot(N/Na,Ma,'ok',ms=7,mew=1.5,mfc='k',lw=2)
    ax.plot(N/Na_opt,Ma_opt,'sm',ms=6,mew=1.5,mfc='m',lw=2)
    ax.plot(x_unc,y_unc,'xy',ms=7,mew=1.5,mfc='y',lw=2)
    y_top = M
    y_bottom = PAPC
    condition = (xax>=1) & (y_bottom<=M)
    ax.fill_between(xax,y_bottom,y_top,where=condition,color='gray',alpha=0.25)
    xpos = np.argmin(np.abs(M-PAPC))
    xmax = xax[xpos]
    ax.set_xlim([np.min([1-0.1*xmax,x]),np.max([1.1*xmax,x])])
    ax.set_ylim([np.min([Ma_min-(M-Ma_min)*0.1,y]),M+(M-Ma_min)*0.1])
    ax.set_xlabel('$x=N/N_\\mathrm{a}$')
    ax.set_ylabel('$y=M_\\mathrm{a}$')
    plt.title('Config: '+config+', $N =$ '+str(N)+', $K =$ '+str(K)+', $\\kappa/\\kappa_\\mathrm{max} =$ '+
              str(np.round(kappa/kappa_max,2)),fontsize=16)
    plt.tight_layout()
    plt.show()
