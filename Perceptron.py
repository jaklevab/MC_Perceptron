# Imports
import numpy as np
import random as rd
from tqdm import tqdm
import pandas as pd
import scipy.stats as st
import functools

#TODO: Add Breaking Condition

class PerceptronMCMC():

    def __init__(self,n_dim,alpha,beta,w_real=None):
        self.n_dim=int(n_dim)
        self.alpha=alpha
        self.beta=beta
        self.nb_samples=int(n_dim*alpha)
        self.X=np.random.normal(0,1,(self.n_dim,self.nb_samples))
        w0=st.bernoulli.rvs(0.5, size=n_dim).reshape(1,-1)
        w0[w0==0]=-1
        self.w0=w0
        self.w=w0
        if w_real is None:
            w_real=st.bernoulli.rvs(0.5, size=n_dim).reshape(1,-1)
            w_real[w_real==0]=-1
        self.w_real=w_real
        self.y=np.sign(np.dot(self.w_real,self.X))
    

    def reset_weights(self,to_initial=False):
        if to_initial:
            self.w=self.w0
        else:
            inter=st.bernoulli.rvs(0.5, size=self.n_dim).reshape(1,-1)
            inter[inter==0]=-1
            self.w=inter

    def conf_energy(self,w):
        w=w.reshape((1,self.n_dim))
        X=self.X.reshape((self.n_dim,self.nb_samples))
        pred=np.sign(np.dot(w,X))
        y=self.y.reshape((1,self.nb_samples))
        return 0.5*np.sum((y-pred)**2,axis=1)

    def accept_step(self):
        w_new=self.w.copy()
        w_new[0,rd.choice(range(self.n_dim))]*=-1
        diff_energy=self.conf_energy(w_new)-self.conf_energy(self.w)
        acc_proba=min(1,np.exp(-self.beta*(diff_energy)))
        if rd.random()<acc_proba:
            self.w=w_new

    def simulation(self,max_iter,thresh_energy=1e-5):
        W=[];E=[]
        for it in (range(max_iter)):
            energy=self.conf_energy(self.w)
            if energy<thresh_energy:
                W.append(self.w);E.append(energy)
            else:
                W.append(self.w);E.append(energy) 
                self.accept_step()
        return(W,E)

    def avg_simul(self,max_iter=1000,nb_simulations=100):
        All_W=[];All_E=[]
        for simu in (range(nb_simulations)):
            W_res,E_res=self.simulation(max_iter)
            All_W.append(W_res);All_E.append(E_res);
            self.reset_weights()
        All_W=np.array(All_W).reshape((nb_simulations,max_iter,self.n_dim))
        All_E=np.array(All_E).reshape((nb_simulations,max_iter))
        return All_W,All_E
    
    
class PerceptronSA(PerceptronMCMC):
    
    def __init__(self,n_dim,alpha,beta,w_real=None):
        super().__init__(n_dim,alpha,beta,w_real)
        self.beta0=beta
       
    def cooling_brute(self,it,n):
        self.beta=self.beta0*((it+1)//n)
        
    def expon_cooling(self,rate):
        self.beta=self.beta*rate
    
    def simulation(self,max_iter,beta_max=5):
        W=[];E=[];betas=[]
        rate=(beta_max/(self.beta0))**(1.0/max_iter)
        for it in (range(max_iter)):
            energy=self.conf_energy(self.w)
            W.append(self.w);E.append(energy);betas.append(self.beta)
            self.accept_step()
            self.cooling_brute(it,20)
            #self.expon_cooling(rate)
        return(W,E)
    
    def avg_simul(self,max_iter=1000,nb_simulations=100):
        All_W=[];All_E=[]
        for simu in (range(nb_simulations)):
            self.beta=self.beta0
            W_res,E_res=self.simulation(max_iter)
            All_W.append(W_res);All_E.append(E_res);
            self.reset_weights()
        All_W=np.array(All_W).reshape((nb_simulations,max_iter,self.n_dim))
        All_E=np.array(All_E).reshape((nb_simulations,max_iter))
        return All_W,All_E



