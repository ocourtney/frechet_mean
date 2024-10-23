import numpy as np
import math
import networkx as nx
import netcomp as nc
import scipy.linalg as la
import scipy.sparse.linalg as spla


class GraphEnsemble:
    def __init__(self, n, N, dP):
        '''
        Input:
        - n : Integer (number of nodes)
        - N : Integer (number of graphs/sample size)
        - dP : either List or Integer (when experimenting on synthetic SBM data input dP = [community densities p and 
                cross-community density q], when experimenting on real data input dP = number of communities k)
        '''
        self.n = n
        self.N = N
        self.dP = dP
        self.q = math.log(self.n/2)/(self.n/2)
        if type(dP) == int:
            self.k = dP
        else:
            self.k = len(dP)
        self.eigs = np.empty((self.N, self.k))
        self.graphs = []
    
    def kSBM(self, a_q = 1): #returns one SBM realization

        b = np.empty(self.k, dtype = int) #create array for community sizes

        for i in range(self.k):
            b[i] = self.n/self.k

        q = a_q*self.q
        Q = np.full((self.k,self.k), q)
        P = np.tril(Q,-1) + np.diag(self.dP) + np.triu(Q,1)

        g = nx.stochastic_block_model(b, P, seed=0)
        G = nx.to_numpy_array(g) #converts to numpy adjacency matrix
        return G
    
    def fill_graphs_SBM(self):
        self.k = len(self.dP)
        for _ in range(self.N):
            self.graphs.append(self.kSBM())
    
    def fill_graphs_data(self, data):
            self.graphs = data
        
    def fill_eigs(self):
        for i in range(self.N):
            self.eigs[i,:] = spla.eigs(self.graphs[i], k = self.k, which = 'LR', return_eigenvectors=False)

    
#END CLASS DEFINITION

''' returns a block matrix of weighted entries from dP, a.k.a the expected adjacency matrix '''
def A_weighted(dP, q, n):
    
    rows = cols = int(n/len(dP))
    result = np.full((n, n), q)
    
    for i in range(len(dP)):
        result[i * rows:(i + 1) * rows, i * cols:(i + 1) * cols] = np.full((rows,cols),dP[i])
   
    np.fill_diagonal(result, 0)
    return result

''' returns the sum of the spectral distances between one possible mean graph (given by dP), 
and each graph in the ensemble of size N (given by an array of their eigenvalues).  '''
def frechet_func(dP, sample):
    
    eigs_hat = np.empty(sample.k)
    norm = 0

    A_hat = A_weighted(dP, sample.q, sample.n)
    eigs_hat = la.eigh(A_hat, eigvals_only=True, subset_by_index=[sample.n-sample.k, sample.n-1])
    #eigs_hat = (1/math.sqrt(sample.n))*spla.eigs(A_hat, k = sample.k, which = 'LR', return_eigenvectors=False)
    for i in range(sample.N):
        norm += np.linalg.norm((eigs_hat - sample.eigs[i,:])/math.sqrt(sample.n), ord = 2)
    
    return norm/sample.N

''' returns the estimated gradient via finite differences'''
def finite_diff(descent_func, dP, sample, dx = 0.0001):
    
    grad = np.zeros(len(dP))
    pnew = dP.copy()
    
    for i in range(len(dP)):
        pnew[i] = dP[i] + dx
        pplus = descent_func(pnew, sample)
        pnew[i] = dP[i] - dx
        pminus = descent_func(pnew, sample)
        
        grad[i] = pplus - pminus
        pnew[i] = dP[i]
    
    return (1/(2*dx))*grad


''' returns the local minimum via gradient descent'''
def grad_descent(descent_func, start, gradient_func = finite_diff, sample = None, dx = 0.0001, learn_rate=0.00005, nmax=50, tol=1e-4):
    
    val = start.copy()
    val_history = []
    grad_history = []

    for i in range(nmax):
        gradient = np.array(gradient_func(descent_func, val, sample, dx))
        
        val_history.append([val.copy(), descent_func(val, sample)])
        grad_history.append(gradient)
        
        val_old = val.copy()
        val = val_old - learn_rate * gradient
        
        if abs((val_history[i][1]-descent_func(val, sample))/val_history[i][1]) <= tol: #break based on relative change in frechet function
            break
            
    return val, val_history, grad_history


''' checks graph connectedness by reviewing the largest eigenvalues of the normalized adjacency 
matrix DAD (if eigenvalue 1 has multiplicity one, there is one connected commponent)'''
def check_connect(adj):
    dvec = np.sum(adj, axis=1)
    D = [1/np.sqrt(d) if d != 0 else 0 for d in dvec]
    D = np.diag(D)
    #print(D)
    Nadj = np.matmul(np.matmul(D, adj), D)
    alleigs = spla.eigs(Nadj, k = 9, which = 'LR', return_eigenvectors=False)
    
    count = 0
    for i in range(len(alleigs)):
        if (alleigs[i] < 1.0001) & (alleigs[i] > 0.9991):
            count +=1
    
    return alleigs, count