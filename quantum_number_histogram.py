# William Lippincott
# this code will plot all the histogram plots that I currently have onto
# one singluar plot in a 2x2 setup of subplots

from sparkx.loader.JetscapeLoader import JetscapeLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sparkx.Filter import keep_mesons
from sparkx.Particle import Particle
import warnings
from typing import Tuple, List, Union, Optional
from sparkx.Filter import particle_status
from matplotlib.gridspec import GridSpec

dataset = [
    ("/home/billylipp/jetscape-docker/JETSCAPE/test_final_state_hadrons_Mlevelmax4_PdecaysOFF_Q05_vir08_final.dat")]

loader = JetscapeLoader(dataset[0])
particles, _, _, _ = loader.load()

filtered_particles1 = keep_mesons(particles)
# comment out whichever particle status you dont want
filtered_particles2 = particle_status(filtered_particles1, 811)
#filtered_particles2 = particle_status(filtered_particles1, 821)

num_events = len(particles)
all_energies = [p.E for evt in particles for p in evt]
E_min = min(all_energies)
E_max = max(all_energies)

num_bins = 20
bins = np.linspace(E_min, E_max, num_bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths  = np.diff(bins)

# set up the graph
fig = plt.figure(figsize=(6.5, 5.0))
grid_spacing = GridSpec(2, 2)
upper_left = fig.add_subplot(grid_spacing[0, 0])  
lower_left = fig.add_subplot(grid_spacing[1, 0])   
upper_right = fig.add_subplot(grid_spacing[0, 1])  
lower_right = fig.add_subplot(grid_spacing[1, 1]) 

J_counts = np.zeros(6, dtype=int)

for event in filtered_particles2:
    for p in event:
        if abs(p.pdg) % 10 == 8:
            J = 5
        else:
            J = int(((abs(p.pdg) % 10) - 1) / 2)
        if 0 <= J <= 5:
            J_counts[J] += 1

# calculate the percentage of total particles with that J value
percentages = (J_counts / J_counts.sum()) * 100

################################################################################
# S and L quantum numbers plotting section
# S = total spin angular momentum
# J = total angular momentum
# L = total orbital angular momentum
L_counts = np.zeros(7, dtype=int) # preallocate array for L = 0, 1, 2, 3, 4, 5, 6
S_counts = np.zeros(2, dtype=int) # preallocate array for S = 0, 1

# now move onto the energy spectrogram plotting preallocation
# create a dictionary for the energies for the L quantum number and for the
# energies for the S quantum number
E_for_L = {l: [] for l in range(7)} # L = 0, 1, 2, 3, 4, 5, 6
E_for_S = {s: [] for s in range(2)} # S = 0, 1
for event in filtered_particles2:
    for p in event:
        if abs(p.pdg) % 10 == 8:
            J = 5
        else:
            J = int(((abs(p.pdg) % 10) - 1) / 2) 
        seven_number_pid = f"{abs(p.pdg):07d}" # always seven characters long
        n_L   = int(seven_number_pid[-5])   
        if J == 0:            
            if   n_L == 0:  
                L,S = 0,0
            elif n_L == 1:  
                L,S = 1,1
            else:           
                continue
        else:                           
            if   n_L == 0:  
                L,S = J-1,1
            elif n_L == 1:  
                L,S = J,  0
            elif n_L == 2:  
                L,S = J,  1
            elif n_L == 3:  
                L,S = J+1,1
            else:           
                continue
        L_counts[L] += 1
        S_counts[S] += 1
        # this adds each particles energy to the bin "L" which is seen from
        # the syntax E_L[L] and then the p.E gets added to the bin from the 
        E_for_L[L].append(p.E)
        E_for_S[S].append(p.E)

# calculate the percentages of the L quantum number
percentages_L = ((L_counts / L_counts.sum()) * 100) 
percentages_S = ((S_counts / S_counts.sum()) * 100) 

###############################################################################
# now move onto the energy spectrogram plotting preallocation
# create a dictionary for the energies for the L quantum number and for the
# energies for the S quantum number
E_for_n_r = {r: [] for r in range(3)}

# preallocate a zeros array with 3 values for 0, 1, and 2 (the values that the radial quantum number takes on)
n_r_counts = np.zeros(3, dtype=int)

for event in filtered_particles2:
    for p in event:
        # make the particle id always 7 characters long so I can index the 2 number
        seven_number_pid = f"{abs(p.pdg):07d}"
        n_r = int(seven_number_pid[-6])
        # add up the number of particles for each radially excited number 
        n_r_counts[n_r] += 1
        # add the energy of each particle to each radially energy number
        E_for_n_r[n_r].append(p.E)

# calculate the percentages of the radial energies
percentages_n_r = ((n_r_counts / n_r_counts.sum()) * 100)

###############################################################################
# plotting
# J plot
upper_left.bar(range(6), percentages)
upper_left.set_ylabel('Percentage of total Particles')
upper_left.set_xlabel(r'$J$')
upper_left.set_title(r"Quantum Number $J$")

# L plot
upper_right.bar(range(7), percentages_L)
upper_right.set_ylabel('Percentage of total Particles')
upper_right.set_xlabel(r'$L$')
upper_right.set_title(r"Quantum Number $L$")

# S plot
lower_right.bar(range(2), percentages_S)
lower_right.set_ylabel('Percentage of total Particles')
lower_right.set_xticks(range(2), [0, 1])
lower_right.set_xlabel(r'$S$')
lower_right.set_title(r"Quantum Number $S$")

# radial excitation plot
lower_left.bar(range(3), percentages_n_r)
#plt.xticks(range(3), [str(n) for n in range(3)])
lower_left.set_xticks(range(3), [0, 1, 2])
lower_left.set_ylabel('Percentage of total Particles')
lower_left.set_xlabel("Radial Excitation number")
lower_left.set_title("Radial Excitation")

fig.suptitle("Histograms for Quantum Numbers - only Recombination")
fig.tight_layout()
fig.savefig('histogram_group_plot.png', dpi=600)
plt.close(fig)