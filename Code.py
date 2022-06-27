"""needed Packages"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import networkx as nx
import scipy
import pickle
import pprint
import tikzplotlib
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from IPython.display import clear_output
from math import comb

"""specify plot parameters"""
plt.rcParams.update({
         'legend.fontsize' : 'x-large',
         'figure.figsize'  : (6, 4),
         'axes.labelsize'  : '20',
         'axes.titlesize'  :'20',
         'legend.fontsize' : 18,
         'legend.fontsize' : 18,
         'xtick.labelsize' :'20',
         'ytick.labelsize' :'20',
         'mathtext.fontset':'stix',
         'font.family'     :'STIXGeneral',
         'lines.markersize': '7'
         })



"""Main Classes"""
class Simulator(dict):
    def __init__(self, GH = False, GO= False, init = True):
        
        #================================
        self.p_er = 0.3    #edge removal probability
        self.p_t = 0.3     #treatment probability 
        self.n = 50        #number of nodes
        self.h = 50
        self.o = 50
        
        #=================================
        # True Hidden network
        if True:#GNP
            self.p_e = 0.4  #edge probability
            self.GH = nx.gnp_random_graph(self.n, self.p_e, seed=None, directed=False)
            self.AH = nx.adjacency_matrix(self.GH)

        if False:#SBM
            sizes   = [20, 20, 20]
            edge_prob   = np.array([[0.05, 0.1, 0.1], [0.1, 0.35, 0.1], [0.1, 0.1, 0.40]])
            self.GH = nx.stochastic_block_model(sizes, edge_prob, seed=None)
        
        #=================================
        #Treatment
        self.W = np.array([np.random.random() < self.p_t for i in range(self.n)])
        
        #=================================
        #Covariates
        self.C = np.random.randint(-10,10,self.n) 
        
        #=================================
        # True Observed network
        self.GO = self.sample_network(self.GH, 0, self.p_er)
        self.AO = nx.adjacency_matrix(self.GO)
        
        #=================================
        # Potential Hidden and Observed Networks
        if True: 
            self.GhatH =          [self.sample_network(self.GO, (self.p_e/(1-self.p_e))*(self.p_er), 0) for i        in range(self.h)]
            self.GhatO =          [[self.sample_network(GhatH_i, 0,  self.p_er) for i in range(self.o)] for GhatH_i in self.GhatH]
        if False:
            self.GhatH =          [self.sample_network_SBM(self.GO, sizes, (edge_prob/(1-edge_prob))*(edge_prob)) for i        in range(self.h)]
            self.GhatO =          [[self.sample_network(GhatH_i, 0,  self.p_er) for i in range(self.o)] for GhatH_i in self.GhatH]
            
    def linreg(self, X,Y):
        beta = LinearRegression(fit_intercept = False).fit(X.reshape(-1, 1),Y.reshape(-1, 1)).coef_[0]
        return beta
    
    def estimate(self, E):
        """estimate target function from given network"""
        #extract features from network and prepare data for estimatio
        X = np.transpose(np.array([self.W, self.Z, E]))
        Y = self.Y
        #estimate
        fit = LinearRegression().fit(X,Y)
        print(fit.intercept_,fit.coef_)
        return fit.coef_[2]
    
    def sample_network_SBM(self,G, sizes, edge_prob): #tbd.
        """
        edge_prob = prob. of adding   missing  edge
        """
        Gs = G.copy()
        Gs.add_edges_from(nx.stochastic_block_model(sizes, edge_prob, seed=None).edges())
        return Gs   
    
    def sample_network(self,G,p, q):
        """
        p = prob. of adding   missing  edge
        q = prob. of deleting existing edge
        """
        Gs = G.copy()
        for e in nx.non_edges(G): #add eges at random
            if np.random.random() < p:
                Gs.add_edge(*e)
        for e in nx.edges(G): #remove eges at random
            if np.random.random() < q:
                Gs.remove_edge(*e)
        return Gs
    
    def exposure_abs(self, G):
        """absolute number of treated neighbors in neighborhood"""
        A = nx.adjacency_matrix(G)
        num_treated = np.array(A.todense().dot(self.W))[0]
        return num_treated
    
    def exposure_frac(self, G):
        """estimate effective exposure that is experianced by each node given a certain network"""
        degrees = [val for (node, val) in G.degree()]
        A = nx.adjacency_matrix(G)
        num_treated = np.array(A.todense().dot(self.W))[0]
        treated_frac=  num_treated / degrees
        return treated_frac  
    
    def remove_edges_uniform(G, alpha):
        edges_to_remove = sample(G.edges, ceil(G.number_of_edges() * alpha))
        new_g = G.copy()
        new_g.remove_edges_from(edges_to_remove)
        return new_g

    def remove_nodes_uniform(G, alpha):
        nodes_to_keep = sample(G.nodes, floor(G.number_of_nodes() * (1 - alpha)))

        return G.subgraph(nodes_to_keep)

    def remove_edges_pro_deg(G, alpha):
        new_g = G.copy()

        deg = nx.degree(G)
        degsum = np.array([deg[x] + deg[y] for x, y in G.edges()])

        np.random.seed(int.from_bytes(os.urandom(4), byteorder='big'))
        edge_ids = np.random.choice(G.number_of_edges(), size=ceil(G.number_of_edges() * alpha),
                                    replace=False, p=degsum / degsum.sum())

        edges_to_remove = np.array(G.edges)[edge_ids, :]

        new_g.remove_edges_from(edges_to_remove)
        return new_g

    def add_edges_random(G, alpha):
        edges_to_add = ceil(G.number_of_edges() * alpha)
        new_G = G.copy()
        N = new_G.number_of_nodes()
        edgelist = set([tuple(sorted(e)) for e in new_G.edges()])
        
        cnt = 0
        new_edges = set()

        while cnt < edges_to_add:
            new_edge = tuple(sorted([randint(0, N - 1), randint(0, N - 1)]))
            if new_edge[0] == new_edge[1]:
                continue
            elif not (new_edge in edgelist or new_edge in new_edges):
                new_edges.add(new_edge)
                cnt += 1

        new_G.add_edges_from(new_edges)
        return new_G
 


class Chin_MV(Simulator):
    def __init__(self, intercept = True, direct = True, indirect = True, covariates = False, GH = False, GO= False, init = True):
        """
        Description Here.
        """
        #================================
        #Initiate Parent class
        Simulator.__init__(self)
        
        #================================
        self.exposure   = self.exposure_frac
        self.intercept  = intercept
        self.direct     = direct
        self.indirect   = indirect
        self.covariates = covariates
        
        #=================================        
        #true exposure from network
        self.E = self.exposure(self.GH) 
        
        #=================================        
        #Causal Function
        self.t0 = 2
        self.tW = 8
        self.tE = 4
        self.tC = 0
        self.Y = self.t0 + self.tW*self.W + self.tC*self.C + self.tE*self.E #+ 0.0*np.random.normal(size =self.n)
        
        #=================================
        #true theta vector
        self.theta_true = []
        if intercept:
            self.theta_true.append(self.t0)        
        if direct:
            self.theta_true.append(self.tW)        
        if indirect:
            self.theta_true.append(self.tE)        
        if covariates:
            self.theta_true.append(self.tC)
        self.par_num = len(self.theta_true)

        #=================================        
        #true exposure from observed network
        self.EO = self.exposure(self.GO)
        
        #estimate
        EO        = np.expand_dims(self.EO, axis = -1)
        W         = np.expand_dims(self.W , axis = -1)
        C         = np.expand_dims(self.C , axis = -1)
        intercept = np.ones(EO.shape)
        
        XO = EO
        if self.direct: 
            XO = np.append(W, XO        , axis = -1)
        if self.intercept: 
            XO = np.append(intercept, XO, axis = -1)        
        if self.covariates:
            XO = np.append(XO,C         , axis = -1)  
        
        self.XO = XO 
        
        self.theta0 = sm.OLS(self.Y,self.XO).fit().params
        
        
        #=================================
        # Sampled Potential Hidden an Potential Observed Networks
        self.EhatH = np.array([ self.exposure(GhatH_i )         for GhatH_i  in                               self.GhatH])
        self.EhatO = np.array([[self.exposure(GhatO_ij)         for GhatO_ij in GhatO_i]       for GhatO_i in self.GhatO])
        return 
    
    def run(self, steps):
        #parameters for shorter notation:
        o = self.o
        h = self.h
        n = self.n 
        par_num = self.par_num
        
        results = {}
        results["True"]= np.round(self.theta_true,2)
        
        #======================
        # Naiv
        theta0 = np.array(self.theta0)
        results["Naiv estimator:"]=np.round(theta0,2)
        
        #====================
        #linear features
        EhatH     = np.expand_dims([np.repeat([EhatH_i], o, axis=0) for EhatH_i in self.EhatH]  , axis = -1)
        EhatO     = np.expand_dims(                                                self.EhatO   , axis = -1)
        W         = np.expand_dims( np.repeat([np.repeat([self.W], o , axis=0)],h , axis=0)     , axis = -1)
        C         = np.expand_dims( np.repeat([np.repeat([self.C], o , axis=0)],h , axis=0)     , axis = -1)
        intercept = np.ones(EhatH.shape)

        
        G = sm.OLS(EhatH.flatten(), EhatO.flatten()).fit().params[0]
        print(G)
        #F = np.append(np.ones(self.EO.shape),self.W, axis =-1)
        #F = np.append(F,self.EO * G, axis = -1).reshape(self.n,par_num, order="F")
        #print("seperate E fit",sm.OLS(self.Y, F).fit().params)
        
        
        
        #combine features
        XH = EhatH
        XO = EhatO
        if self.direct: 
            XH = np.append(W, XH        , axis = -1)
            XO = np.append(W, XO        , axis = -1)
        if self.intercept: 
            XH = np.append(intercept, XH, axis = -1)
            XO = np.append(intercept, XO, axis = -1)        
        if self.covariates:
            XH = np.append(XH,C         , axis = -1)
            XO = np.append(XO,C         , axis = -1)
            
       #====================
        # One obs per Oij
        XH = XH.reshape(h*o, n, par_num)
        XO = XO.reshape(h*o, n, par_num)
        X = np.array([np.dot(inv(np.dot(xo.T, xo)),np.dot(xo.T, xh)) for xo, xh in np.stack((XO,XH), axis = 1)]).reshape(o*h*par_num, par_num, order="F")
        b =  np.repeat([theta0], o*h, axis = 0).reshape(o*h*par_num,1) 
        beta_obs = sm.OLS(b,X).fit().params 
        results["Analytical Estimator with 1 obs per Oij"]=np.round(beta_obs,2)
        
        #====================
        # One obs per Hi
        XH = XH.reshape(h, o*n, par_num)
        XO = XO.reshape(h, o*n, par_num)
        X = np.array([np.dot(inv(np.dot(xo.T, xo)),np.dot(xo.T, xh)) for xo, xh in np.stack((XO,XH), axis= 1)]).reshape(h*par_num, par_num, order="F")
        b =  np.repeat([theta0], h, axis = 0).reshape(h*par_num,1) 
        beta_hidden = sm.OLS(b,X).fit().params 
        results["Analytical Estimator with 1 obs per Hi"]=np.round(beta_hidden,2)
        def fun(theta):
            _ = X*theta
            return np.sum((np.array(_ - b).flatten())**2)
        #print(scipy.optimize.minimize(fun, theta0))
        
        #====================
        # One obs in total
        XH = XH.reshape(h*o*n, par_num)
        XO = XO.reshape(h*o*n, par_num)
        X = np.array(np.dot(inv(np.dot(XO.T, XO)),np.dot(XO.T, XH))) #.reshape(h*par_num,par_num, order="F")
        b =  theta0
        beta_global = sm.OLS(b,X).fit().params
        #print("ana", np.dot(inv(X),theta0))
        #print("ana LS", np.dot(inv(np.dot(X.T, X)),np.dot(X.T, b)))
        results["Analytical Estimator with 1 obs total"]=np.round(beta_global,2)        
        return results
   
    def run_UV(self, steps):
        results = {}
        
        beta0 =  self.linreg(self.EO,self.Y)[0]
        results["Naiv estimator:"]=np.round(beta0,2)
        
        EhatH = np.array([np.repeat([EhatH_i], self.o, axis=0) for EhatH_i in self.EhatH])
        EhatO = self.EhatO
        
        #====================
        # One obs per Oij
        EH = EhatH.reshape(self.h*self.o,self.n,1)
        EO = EhatO.reshape(self.h*self.o,self.n,1)
        E = np.array([np.dot(inv(np.dot(eo.T, eo)),np.dot(eo.T, eh)) for eo, eh in zip(EO,EH)]).reshape(-1, 1)
        E_ = np.array([np.dot(inv(np.dot(eo.T, eo)),np.dot(eo.T, eh)) for eo, eh in zip(EO,EH)]).reshape(-1, 1)
        b =  np.array(len(E)*[beta0])
        beta_obs = LinearRegression(fit_intercept = False).fit(E,b).coef_[0]
        results["Analytical Estimator with 1 obs per Oij"]=np.round(beta_obs,2)

        #====================
        # One obs per Hi
        EH = EhatH.reshape(self.h,self.o*self.n,1)
        EO = EhatO.reshape(self.h,self.o*self.n,1)
        E = np.array([np.dot(inv(np.dot(eo.T, eo)),np.dot(eo.T, eh)) for eo, eh in zip(EO,EH)]).reshape(-1, 1)
        b =  np.array(len(E)*[beta0])
        beta_hidden = LinearRegression(fit_intercept = False).fit(E,b).coef_[0]
        results["Analytical Estimator with 1 obs per Hi"]=np.round(beta_hidden,2)


        #====================
        # One obs in total
        EH = EhatH.reshape(self.h*self.o*self.n,1)
        EO = EhatO.reshape(self.h*self.o*self.n,1)
        E_global = np.dot(inv(np.dot(EO.T, EO)),np.dot(EO.T, EH))
        beta_global = beta0 / E_global
        results["Analytical Estimator with 1 obs total"]=np.round(beta_global,2)

        #====================
        #MoM
        beta = beta0
        beta_list = [beta]
        
        for i in range(steps):
            Y = beta * self.EhatH     
            delta_bar = np.mean([[self.linreg(self.EhatO[h][o],Y[h]) for o in range(self.o)] for h in range(self.h)])
            beta_new = beta - 0.5*(delta_bar - beta0)
            beta_list.append(beta_new)
            if abs((beta_new - beta) / beta) < 0.001 and (i> 10):
                beta = beta_new
                break
            beta = beta_new
        results["Iterative Estimator"]=np.round(beta,2)
        
        #====================
        #Confidence interval
        CL_u = beta
        CL_u_list = [CL_u]
        
        for i in range(steps):
            Y = CL_u * self.EhatH     
            delta_bar = np.percentile([[self.linreg(self.EhatO[h][o],Y[h]) for o in range(self.o)] for h in range(self.h)],2.5)
            CL_u_new = CL_u - 0.5*(delta_bar - beta0)
            CL_u_list.append(CL_u_new)
            if abs((CL_u_new - CL_u) / CL_u) < 0.001 and (i> 10):
                CL_u = CL_u_new
                break
            CL_u = CL_u_new
        #print(CL_u)

        
        CL_l = beta + (beta- CL_u)
        CL_l_list = [CL_l]
        for i in range(steps):
            Y = CL_l * self.EhatH     
            delta_bar = np.percentile([[self.linreg(self.EhatO[h][o],Y[h]) for o in range(self.o)] for h in range(self.h)],97.5)
            CL_l_new = CL_l - 0.5*(delta_bar - beta0)
            CL_l_list.append(CL_l_new)
            if abs((CL_l_new - CL_l) / CL_l) < 0.001 and (i> 10):
                CL_l = CL_l_new
                break
            CL_l = CL_l_new
        #print(CL_l)
        results["Iterative Confidence Interval"]=np.round((CL_l, CL_u),2)

                
        #====================
        if True:
            #Plot
            fig, ax = plt.subplots(1, 1)

            ax.axhline(beta_obs, color = "m", label ="Pot.Obs")
            ax.axhline(beta_hidden, color = "darkgrey", label ="Pot.Hidden")
            ax.axhline(beta_global, color = "grey", label = "Obs")
            ax.axhline(self.tE, color= "red", label = "true value")
            ax.axhline(beta0, color= "green", label = "naiv")
            ax.plot(beta_list, color ="black", linestyle = "solid", label = "Iterative")
            ax.plot(CL_u_list, color ="black", linestyle = "dashed", label = "Upper CI")
            ax.plot(CL_l_list, color ="black", linestyle = "dashed", label = "Lower CI")
            ax.set_xlabel(r'$num \ iterations$')
            ax.set_ylabel(r'$\theta$')
            ax.legend()
        
        return results

    def linreg(self, X,Y):
        beta = LinearRegression(fit_intercept = False).fit(X.reshape(-1, 1),Y.reshape(-1, 1)).coef_[0]
        return beta


class AronowSamii(Simulator):
    def __init__(self, GH = False, GO= False, init = True):
        Simulator.__init__(self)
        
        #================================
        
        self.exposure = self.exposure_AS

        #true exposure from network
        self.E = self.exposure(self.GH)         
        self.theta_true = [10,7,5,1]
        self.Y = self.pred_Y_AS(self.theta_true, self.E)
            
        
        #=================================
        #true exposure / exposure prob from observed network
        
        self.EO = self.exposure(self.GO)
        self.PO = self.prob_AS(0.5, self.GO)
        self.theta0 = self.estimate_AS(self.EO, self.PO, self.Y)
        
        #=================================
        # Sampled Potential Hidden an Potential Observed Networks

        self.EhatH = np.array([ self.exposure(GhatH_i)  for GhatH_i  in self.GhatH])
        self.PH    = np.array([ self.prob_AS(0.5,GhatH_i)   for GhatH_i  in self.GhatH]) 
        self.EhatO = np.array([[self.exposure(GhatO_ij) for GhatO_ij in GhatO_i] for GhatO_i in self.GhatO])
        self.PO    = np.array([[self.prob_AS(0.5,GhatO_ij)  for GhatO_ij in GhatO_i] for GhatO_i in self.GhatO])

        

    def run_AS(self, steps):
        
        #====================
        #MoM
        
        theta0 = self.theta0
        theta  = theta0
        theta_list = [theta]
        
        for i in range(steps):
            Y = np.array([self.pred_Y_AS(theta, EhatH_i)  for EhatH_i  in self.EhatH])
            delta_bar = np.mean([[self.estimate_AS(self.EhatO[h][o], self.PO[h][o], Y[h]) for o in range(self.o)] for h in range(self.h)], axis = (0,1))
            theta_new = theta - 0.5*(delta_bar - theta0)
            theta_list.append(theta_new)
            if False:#abs((theta_new - theta) / theta) < 0.001 and (i> 10):
                theta = theta_new
                break
            theta = theta_new
        self.theta_list = theta_list
            
        
        #====================
        #Confidence interval
        
        CL_u = theta
        CL_u_list = [CL_u]
        
        for i in range(steps):
            Y = np.array([self.pred_Y_AS(CL_u, EhatH_i)  for EhatH_i  in self.EhatH])
            delta_bar = np.percentile([[self.estimate_AS(self.EhatO[h][o], self.PO[h][o], Y[h]) for o in range(self.o)] for h in range(self.h)], 2.5, axis = (0,1))
            CL_u_new = CL_u - 0.5*(delta_bar - theta0)
            CL_u_list.append(CL_u_new)
            if np.linalg.norm((CL_u_new - CL_u), ord=1) < 0.001:
                CL_u = CL_u_new
                break
            CL_u = CL_u_new
            
        self.CL_u_list = CL_u_list
        
        CL_l = theta + (theta- CL_u)
        CL_l_list = [CL_l]
        for i in range(steps):
            Y = np.array([self.pred_Y_AS(CL_l, EhatH_i)  for EhatH_i  in self.EhatH])
            delta_bar = np.percentile([[self.estimate_AS(self.EhatO[h][o], self.PO[h][o], Y[h]) for o in range(self.o)] for h in range(self.h)], 97.5, axis = (0,1))
            CL_l_new = CL_l - 0.5*(delta_bar - theta0)
            CL_l_list.append(CL_l_new)
            if np.linalg.norm((CL_l_new - CL_l), ord=1) < 0.001:
                CL_l = CL_l_new
                break
            CL_l = CL_l_new
        self.CL_l_list = CL_l_list
   
            
        #====================
        #plot results
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.set_title(r'$c_{11}$')
        ax2.set_title(r'$c_{10}$')
        ax3.set_title(r'$c_{01}$')
        ax4.set_title(r'$c_{00}$')
        
        #estimate
        theta_list = list(zip(*theta_list))
        ax1.plot(theta_list[0], color = "m")
        ax2.plot(theta_list[1], color = "m")
        ax3.plot(theta_list[2], color = "m")
        ax4.plot(theta_list[3], color = "m")
        
        #lower Confidence Interval Bound
        CL_l_list = list(zip(*CL_l_list))
        ax1.plot(CL_l_list[0], color = "m")
        ax2.plot(CL_l_list[1], color = "m")
        ax3.plot(CL_l_list[2], color = "m")
        ax4.plot(CL_l_list[3], color = "m")
        
        #upper Confidence Interval Bound
        CL_u_list = list(zip(*CL_u_list))
        ax1.plot(CL_u_list[0], color = "m")
        ax2.plot(CL_u_list[1], color = "m")
        ax3.plot(CL_u_list[2], color = "m")
        ax4.plot(CL_u_list[3], color = "m")
        
        #true
        ax1.axhline(self.theta_true[0], color = "r")
        ax2.axhline(self.theta_true[1], color = "r")
        ax3.axhline(self.theta_true[2], color = "r")
        ax4.axhline(self.theta_true[3], color = "r")        
        
        #naiv
        ax1.axhline(theta0[0], color = "g")
        ax2.axhline(theta0[1], color = "g")
        ax3.axhline(theta0[2], color = "g")
        ax4.axhline(theta0[3], color = "g")
        
        return 
    
    def exposure_AS(self, G):
        """calculates exposure state of exposure model from Aronow and Samii"""
        
        A = nx.adjacency_matrix(G)
        num_treated = np.array(A.todense().dot(self.X))
        
        #exposure states
        c11 = (self.X == 1) & (num_treated >= 1)
        c10 = (self.X == 1) & (num_treated == 0)
        c01 = (self.X == 0) & (num_treated >= 1)
        c00 = (self.X == 0) & (num_treated == 0)
        
        return *c11, *c10, *c01, *c00
    
    def prob_AS(self,p,G):
        """calculates exposure probabilities of each exposure for each node"""
        
        degrees = [val for (node, val) in G.degree()]
        
        #exposure probabilities
        p_ie11  = [p    *(1-(1-p)**di)for di in degrees]
        p_ie10  = [p    *   (1-p)**di for di in degrees]
        p_ie01  = [(1-p)*(1-(1-p)**di)for di in degrees]
        p_ie00  = [(1-p)*   (1-p)**di for di in degrees]
        
        return p_ie11, p_ie10, p_ie01, p_ie00

    def estimate_AS_(self, E, P, Y):
        """estimator from Aronow and Samii"""
        
        #exposure probability
        p_ie11, p_ie10, p_ie01, p_ie00 =  P
        
        #esposure received
        c11, c10, c01, c00 = E
        
        #estimates
        y11= np.nansum(c11* Y/ (p_ie11)) / self.n
        y10= np.nansum(c10* Y/ (p_ie10)) / self.n
        y01= np.nansum(c01* Y/ (p_ie01)) / self.n
        y00= np.nansum(c00* Y/ (p_ie00)) / self.n
        
        return y11, y10, y01, y00
    def estimate_AS(self, E, P, Y):
        """estimator from Aronow and Samii"""
        
        #exposure probability
        p_ie11, p_ie10, p_ie01, p_ie00 =  P
        
        #esposure received
        c11, c10, c01, c00 = E
        
        #estimates
        y11= np.nansum(c11* Y/ (p_ie11)) / self.n
        y10= np.nansum(c10* Y/ (p_ie10)) / self.n
        y01= np.nansum(c01* Y/ (p_ie01)) / self.n
        y00= np.nansum(c00* Y/ (p_ie00)) / self.n
        _11= np.nansum(c11/ (p_ie11)) / self.n
        _10= np.nansum(c10/ (p_ie10)) / self.n
        _01= np.nansum(c01/ (p_ie01)) / self.n
        _00= np.nansum(c00/ (p_ie00)) / self.n
        
        return y11 /_11 , y10 /_10, y01 /_01, y00 / _00
    
    def prob_AS_m(self,p,G):
        p  = 0.5 # treatment probability
        
        degrees = [val for (node, val) in G.degree()]
        
        #exposure probabilities
        p_ie11  = [p    *sum([comb(di,x) * p**x * (1-p)**(di-x) for x in range(1,di+1)]) for di in degrees]
        p_ie10  = [p    * (1-p)**di for di in degrees]
        p_ie01  = [(1-p)*sum([comb(di,x) * p**x * (1-p)**(di-x) for x in range(1,di+1)]) for di in degrees]
        p_ie00  = [(1-p)* (1-p)**di for di in degrees]
        
        #p_e = zip(p_ie11, p_ie10, p_ie01, p_ie00)
        return p_ie11, p_ie10, p_ie01, p_ie00
    
    def pred_Y_AS(self, theta, E):
        E = np.array(list(zip(*E)))
        Y = E.dot(theta) 
        return Y 



"""
        #c11, c10, c01, c00 = np.split(self.exposure_AS(G),[1,2,3])

    def compare_centrality_dicts_correlation(d1, d2, scipy_correlation=kendalltau):
        if set(d1) != set(d2):
            nodes = sorted(set(d1).intersection(set(d2)))
        else:
            nodes = sorted(d1)

        v1 = np.round([d1[x] for x in nodes], 12)
        v2 = np.round([d2[x] for x in nodes], 12)

        return scipy_correlation(v1, v2).correlation


    def robustness_calculator_builder(centrality_measure, comparison_function=compare_centrality_dicts_correlation):
        @wraps(centrality_measure)
        def f(g0, g1):
            return compare_centrality_dicts_correlation(centrality_measure(g0), centrality_measure(g1))
        return f


    def estimate_robustness(measured_network, error_mechanism, robustness_calculator, iterations=50, return_values=False):
        measured_robustness = np.array([robustness_calculator(measured_network, error_mechanism(measured_network))
                                        for _ in range(iterations)])
        vals = measured_robustness if return_values else None
        return namedtuple("robustness_estimate", "mean, sd values")(measured_robustness.mean(),
                                                                    measured_robustness.std(), vals)
"""