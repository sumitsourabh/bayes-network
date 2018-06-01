import numpy as np
import pandas as pd
from scipy.stats import mvn, norm, gaussian_kde
from scipy.optimize import fsolve
from matplotlib import pyplot as plt, ticker as ticker
from scipy import integrate
import os

# Portfolio = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\PortfolioA.csv', sep=',',index_col = 0)

def simulation(score, algorithm, *arg):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model'
    if arg:
        algorithm = algorithm + str(arg[0])
    directory = score + '_' + algorithm + '_' + '200' + '_' + '100'
    file = 'PortfolioA_' + score +'_' + algorithm + '.csv'

    Portfolio = pd.read_csv(os.path.join(path,directory,file), sep=',', index_col = 0)
    # Read the correlation matrix from .csv
    Omega = pd.read_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis\Omega.csv', index_col = 0,header = 0)

    m = len(Portfolio)    # number of counterparties in the portfolio
    N = 1000000          # number of simulations
    p = Omega.shape[0]    # number of systematic factors

    Beta = Portfolio[[col for col in list(Portfolio) if col.startswith('beta')]].values
    gamma = Portfolio['gamma'].values
    PD = Portfolio['PD'].values
    EAD = Portfolio['EAD'].values
    LGD = Portfolio['LGD'].values

    # Analytical Expected Loss
    EL_an = np.sum(PD*EAD*LGD)

    # Calibrate default thresholds with PDs
    d = norm.ppf(PD)
    # perform a Cholesky factorisation
    L = np.linalg.cholesky(Omega)

    np.random.seed(10)
    # generate independent normals
    Z = np.random.standard_normal((p, N))

    # convert independent unit normals to correlated
    F = np.dot(L, Z)

    # idiosyncratic loading s.t. the returns are standard normal
    id_load = np.diagonal(np.sqrt(1-np.dot(np.dot(Beta,Omega),Beta.T)))
    epsilon = np.random.standard_normal((N, m))
    # Put everything together to get the returns
    X = np.dot(Beta,F) + (id_load*epsilon).T

    # Calculate UL with no contagion
    # construct default indicator
    I = (((X.T-d)<0))
    L = (EAD*LGD*I).T
    # print(np.mean(L,axis=1))
    Loss=np.sum(L,axis=0)

    # Calculate UL with contagion

    SOV_ID = Portfolio['SOV_ID'].values
    SOV_LINK = Portfolio['SOV_LINK'].values

    Dsd = np.zeros(m)
    Dnsd = np.zeros(m)

    # With contagion
    for i in range(0,m):
        if SOV_ID[i] != 0:
            Dsd[i] = d[i]
            Dnsd[i] = d[i]
        else:
            sov_ind = np.nonzero(SOV_ID == SOV_LINK[i])[0][0]
            PDs = PD[sov_ind]
            corr = np.dot(np.dot((Beta[i]).T,Omega),(Beta[sov_ind]))
            
            Fsd = lambda x: mvn.mvndst([-100, -100], [x, norm.ppf(PDs)],[0,0],corr)[1] / PDs - gamma[i]
            Dsd[i] = fsolve(Fsd, norm.ppf(gamma[i])) # is thera a better initial guess?
            Fnsd = lambda x: mvn.mvndst([-100, norm.ppf(PDs)], [x, 100],[0,1],corr)[1] - PD[i] + gamma[i]*PDs
            Dnsd[i] = fsolve(Fnsd, norm.ppf(PD[i])) # is there a better initial guess?
            if Dsd[i]< d[i] or PD[i]<PD[sov_ind]:
                Dsd[i] = d[i]
                Dnsd[i] = d[i]
        


    # Thresholds
    D = np.array([Dnsd]*N).T

    X_sov = X[np.nonzero(SOV_ID !=0)[0]]
    D_sov = D[np.nonzero(SOV_ID !=0)[0]]

    I_sov = (((X_sov-D_sov)<0))

    for i in range(0,N):
        for j in range(0,m):
            if SOV_ID[j]==0 and I_sov[SOV_LINK[j]-1,i] == 1:
                D[j,i] = Dsd[j]

    # construct default indicator
    I_c = ((X-D)<0).T

    L = (EAD*LGD*I_c).T
    # print(np.mean(L,axis=1))
    Loss_c=np.sum(L,axis=0)

    EL = np.mean(Loss)
    # Arithmetic mean of Loss
    EL_c = np.mean(Loss_c)

    # UL_98 = np.percentile(Loss, 98)
    UL_99 = np.percentile(Loss, 99)
    UL_995 = np.percentile(Loss, 99.5)
    UL_999 = np.percentile(Loss, 99.9)
    UL_9999 = np.percentile(Loss, 99.99)

    # UL_98_c = np.percentile(Loss_c, 98)
    UL_99_c = np.percentile(Loss_c, 99)
    UL_995_c = np.percentile(Loss_c, 99.5)
    UL_999_c = np.percentile(Loss_c, 99.9)
    UL_9999_c = np.percentile(Loss_c, 99.99)


    UL = np.array([ UL_99, UL_995, UL_999, UL_9999])
    UL_c = np.array([ UL_99_c, UL_995_c, UL_999_c, UL_9999_c])

    return UL, UL_c

def plot_result(UL, UL_c, *arg):
    ind = np.arange(len(UL))
    width = 0.5

    p1 = plt.bar(ind, UL, width = width)
    p2 = plt.bar(ind, np.maximum((UL_c-UL),0), bottom = UL, width = width)

    plt.ylabel('Loss (millions)')
    plt.xlabel('Quantile')
    plt.xticks(ind, ('99%','99.5%', '99.9%','99.99%'))

    plt.legend((p1[0],p2[0]),('Loss', 'Contagion add-on'))

    if arg:
        path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital'
        figure_name = '_'.join([str(a) for a in arg]) + '.png'
        plt.savefig(os.path.join(path,figure_name))

    plt.show()

def main():
    score = 'k2'
    algorithm = 'tabu-1-20'
    UL, UL_c = simulation(score, algorithm)
    print("UL:   ",UL)
    print("UL_c: ",UL_c)

    # print("EL_an: ",EL_an)
    # print("EL: ",EL)
    # print("EL_c: ",EL_c)

    plot_result(UL, UL_c, score, algorithm)

if __name__ == '__main__':
    main()
