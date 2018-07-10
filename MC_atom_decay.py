from __future__ import division
import numpy
import random
import matplotlib.pyplot as pyplot
import scipy.integrate


t_half_rad = 20.8 # half-life Ra225 (days)
t_half_act = 10.0 # half-life Ac225 (days)
N0 = 250 # Initial no of Ra225
t1 = 100 # end time for simulation (days)
n_timepoints = 50 #number of timepoints to solve to
tau_rad = t_half_rad / numpy.log(2)
tau_act = t_half_act / numpy.log(2)

def analytic(N0, timebase):
    '''This function solves the analytic number of Ra225'''
    n_analytic = N0 * numpy.exp(-timebase / tau_rad)

    return n_analytic

def simulate_monte_carlo(N0, t1, n_timepoints):
    '''This function simulates the decay of Ra225 using the Monte Carlo technique'''
    dt = t1/ n_timepoints
    
    # Probability of decay of Ra within dt
    p_decay_rad = 1 - numpy.exp(-dt/ tau_rad)
    count_radium = numpy.zeros((n_timepoints,))

     # Probability of decay of Ac within dt
    p_decay_act = 1 - numpy.exp(-dt/ tau_act)
    count_actinium = numpy.zeros((n_timepoints,))
    
    atoms = numpy.ones((N0,))

    for idx_time in range(n_timepoints):
        count_radium[idx_time] = (atoms == 1).sum()
        count_actinium[idx_time] = (atoms == 2).sum()
        
        for idx_atom in range(N0):
            if atoms[idx_atom] == 1:
                if random.random() <= p_decay_rad:
                    atoms[idx_atom] = 2
                # decayed radium (1) becomes actinium (2)
            if atoms[idx_atom] == 2:
                if random.random() <= p_decay_act:
                    atoms[idx_atom] = 3
                # decayed actinium (2) becomes Francium (3)
                
    return (count_radium, count_actinium)

def f((N_rad, N_act),t):
    dN_rad = -N_rad/ tau_rad
    dN_act = N_rad/ tau_rad - N_act/ tau_act
    
    return numpy.array((dN_rad, dN_act))

    
dt = t1/n_timepoints
timebase = numpy.arange(0, t1, dt)
n_analytic = analytic(N0, timebase)
m_carlo = simulate_monte_carlo(N0, t1, n_timepoints)
m_carlo_rad = m_carlo[0]
m_carlo_act = m_carlo[1]
scipy_rad = scipy.integrate.odeint(f, (N0, 0), timebase)[:,0]
scipy_act = scipy.integrate.odeint(f, (N0, 0), timebase)[:,1]



pyplot.figure()
pyplot.plot(timebase, n_analytic, label = 'analytic $^{225}$Ra', color = 'red')
pyplot.plot(timebase, m_carlo_rad, label = 'Monte $^{225}$Ra', color = 'blue')
pyplot.plot(timebase, m_carlo_act, label = 'Monte $^{225}$Ac', color = 'green')
pyplot.plot(timebase, scipy_rad, label = 'Scipy $^{225}$Ra', color = 'black', linestyle='--')
pyplot.plot(timebase, scipy_act, label = 'Scipy $^{225}$Ac', color = 'grey', linestyle='--')
pyplot.xlabel('Time /days')
pyplot.ylabel('Number of atoms')
pyplot.title('Simulation of decay chain of $^{225}$Ra > $^{225}$Ac > $^{221}$Fr')
pyplot.legend(loc= 'upper right')
pyplot.tight_layout()

pyplot.show()
    
    

