from __future__ import division
from numpy import *
from pylab import *

# A small number: 10 times the smallest possible number on this computer for a
# real number.
epsilon = 10.0*np.finfo(float).eps

# Temperature of neuron
T = 6.3
#T = 18.5

## Functions Normal definition

alpha_n = np.vectorize(lambda v,T: 3**((T-6.3)/10)*0.01*(v + 55 + epsilon)/(1-np.exp(-(v + 55 + epsilon)/10)))
beta_n  = np.vectorize(lambda v,T: 3**((T-6.3)/10)*0.125*np.exp(-(v+65)/80))
n_inf   = np.vectorize(lambda v,T: alpha_n(v,T)/(alpha_n(v,T) + beta_n(v,T)))

# Na channel (activating)
alpha_m = np.vectorize(lambda v,T: 3**((T-6.3)/10)*0.1*(v + 40 + epsilon)/(1-np.exp(-(v + 40 + epsilon)/10)))
beta_m  = np.vectorize(lambda v,T: 3**((T-6.3)/10)*4*np.exp(-(v+65)/18))
m_inf   = np.vectorize(lambda v,T: alpha_m(v,T)/(alpha_m(v,T) + beta_m(v,T)))

# Na channel (inactivating)
alpha_h = np.vectorize(lambda v,T: 3**((T-6.3)/10)*0.07*np.exp(-(v+65)/20))
beta_h  = np.vectorize(lambda v,T: 3**((T-6.3)/10)*1/(1+np.exp(-(v + 35)/10)))
h_inf   = np.vectorize(lambda v,T: alpha_h(v,T)/(alpha_h(v,T) + beta_h(v,T)))


"""
# original
# K channel
alpha_n = vectorize(lambda v: 0.01*(-v + 10)/(exp((-v + 10)/10) - 1) if v != 10 else 0.1)
beta_n  = lambda v: 0.125*exp(-v/80)
n_inf   = lambda v: alpha_n(v)/(alpha_n(v) + beta_n(v))

# Na channel (activating)
alpha_m = vectorize(lambda v: 0.1*(-v + 25)/(exp((-v + 25)/10) - 1) if v != 25 else 1)
beta_m  = lambda v: 4*exp(-v/18)
m_inf   = lambda v: alpha_m(v)/(alpha_m(v) + beta_m(v))

# Na channel (inactivating)
alpha_h = lambda v: 0.07*exp(-v/20)
beta_h  = lambda v: 1/(exp((-v + 30)/10) + 1)
h_inf   = lambda v: alpha_h(v)/(alpha_h(v) + beta_h(v))
"""
## setup parameters and state variables
Tim     = 40    # ms
dt    = 0.025 # ms
time  = arange(0,Tim+dt,dt)

## HH Parameters
V_rest  = -65      # mV
Cm      = 1      # uF/cm2
gbar_Na = 1.0*120    # mS/cm2
#gbar_Na = 60    # mS/cm2
gbar_K  = 36     # mS/cm2
gbar_l  = 0.3    # mS/cm2
E_Na    = 50.0    # mV
E_K     = -77.0    # mV
E_l     = -54.4 # mV
# orignal parameter values
#gbar_K  = 36     # mS/cm2
#gbar_l  = 0.3    # mS/cm2
#E_Na    = 115    # mV
#E_K     = -12    # mV
#E_l     = 10.613 # mV

S       = 7                # number of compartments
elec    = 3                # compartment index of stimulating electrode
RA      = .1               # specific intracellular resistivity (kOhm*cm2)
r       = 2e-4             # compartment radius (cm) 0.2 um
l       = 0.00001          # compartment length (cm) 0.1 um 
Ra      = (RA*l)/(pi*r**2) # intracellular resistance (kOhm*cm)

Vm      = zeros([S,len(time)])  # mV
Vm[:,0] = V_rest
dV      = zeros(S)              # mV
m       = ones(S)*m_inf(V_rest,T)
h       = ones(S)*h_inf(V_rest,T)
n       = ones(S)*n_inf(V_rest,T)

## Stimulus
I = zeros(len(time))
for i, t in enumerate(time):
#  if 5 <= t <= 30: I[i] = 10 # uA/cm2
  if 5 <= t <= 10: I[i] = 100.0 # uA/cm2

# HH channel currents
def hh(Vm, g_Na, g_K, g_l):
  return (g_Na * (Vm - E_Na)
        + g_K  * (Vm - E_K )
        + g_l  * (Vm - E_l ))

## connection matrix
Sc = zeros([S,S])
for i in range(S):
  if i == 0:
      Sc[i,0:2]     = [1,-1]
  elif i == S-1:
      Sc[i,i-1:S]   = [-1,1]
  else:
      Sc[i,i-1:i+2] = [-1,2,-1]

## simulate model
for i in range(1,len(time)):
  g_Na = gbar_Na*(m**3)*h
  g_K  = gbar_K*(n**4)
  g_l  = gbar_l

  m += dt*(alpha_m(Vm[:,i-1],T)*(1 - m) - beta_m(Vm[:,i-1],T)*m)
  h += dt*(alpha_h(Vm[:,i-1],T)*(1 - h) - beta_h(Vm[:,i-1],T)*h)
  n += dt*(alpha_n(Vm[:,i-1],T)*(1 - n) - beta_n(Vm[:,i-1],T)*n)

  dV = -hh(Vm[:,i-1], g_Na, g_K, g_l) - Sc.dot(Vm[:,i-1]) / Ra
  dV[elec] += I[i-1]

  Vm[:,i] = Vm[:,i-1] + dt * dV / Cm

## plot membrane potential traces
ylabel_set = False
for i,tr in enumerate(Vm):
  ax = subplot(S,1,i+1)
  ax.plot(time, tr)
  if i == 0:
    ax.set_title('Hodgkin-Huxley Active Compartment Example')
  if i == floor(S/2) and not ylabel_set:
    ax.set_ylabel('Membrane Potential (mV)')
    ylabel_set == True
  if i != S-1:
    ax.set_xticklabels([])
  ax.set_yticks([-20, 50, 120])
xlabel('Time (msec)')
show()
