# William Lippincott
# this code plot a sclaed energy spectrogram of energy for only Upsilon mesons and also
# makes a ratio plot for the ratio of the energy spectrum of charged Upsilon
# for mlevelmax4 / mlevelmax 0. the code also does this same thing for just 
# charged Upsilon mesons formed from recombination. it plots all 4 plots on the same
# plot. 
from sparkx.loader.JetscapeLoader import JetscapeLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sparkx.Filter import particle_species
from sparkx.Filter import particle_status
from matplotlib.gridspec import GridSpec

dataset1 = [
    ("/home/billylipp/jetscape-docker/JETSCAPE/test_final_state_hadrons_Mlevelmax0_PdecaysON_FINAL_WDoff.dat")]
dataset2 = [
    ("/home/billylipp/jetscape-docker/JETSCAPE/test_final_state_hadrons_Mlevelmax4_PdecaysON_FINAL_WDoff.dat")]

# number of energy bins
num_bins = 20
E_cm = 91.2 / 2
# filter for only the particle ids in the following line
Upsilon_meson_particle_id = (553, -553) # B_c, B_c-

E_min, E_max = np.inf, -np.inf

loader0 = JetscapeLoader(dataset1[0])
loader4 = JetscapeLoader(dataset2[0])
particles0, _, _, _ = loader0.load()
particles4, _, _, _ = loader4.load()

filtered_particles_ALL_0 = particle_species(particles0, Upsilon_meson_particle_id)
filtered_particles_ALL_4 = particle_species(particles4, Upsilon_meson_particle_id)

filtered_particles0 = particle_status(filtered_particles_ALL_0, 811)
filtered_particles4 = particle_status(filtered_particles_ALL_4, 811)
num_events = len(particles0)

# collect energies for each event
energies0 = [p.E for evt in filtered_particles0[:num_events] if len(evt) > 0 for p in evt]
energies4 = [p.E for evt in filtered_particles4[:num_events] if len(evt) > 0 for p in evt]
energies0_ALL = [p.E for evt in filtered_particles_ALL_4[:num_events] if len(evt) > 0 for p in evt]
energies4_ALL = [p.E for evt in filtered_particles_ALL_0[:num_events] if len(evt) > 0 for p in evt]
x_e_0 = np.divide(energies0, E_cm)
x_e_4 = np.divide(energies4, E_cm)
x_e_0_ALL = np.divide(energies0_ALL, E_cm)
x_e_4_ALL = np.divide(energies4_ALL, E_cm)
# filter for only greater than 0 so log x-axis doesnt blow up
x_e_0 = x_e_0[x_e_0  > 0]
x_e_4 = x_e_4[x_e_4  > 0]
x_e_0_ALL = x_e_0_ALL[x_e_0_ALL > 0]
x_e_4_ALL = x_e_4_ALL[x_e_4_ALL > 0]

x_E_min0 = min(E_min, np.min(x_e_0))
x_E_max0 = max(E_max, np.max(x_e_0))
x_E_min4 = min(E_min, np.min(x_e_4))
x_E_max4 = max(E_max, np.max(x_e_4))

x_E_min0_ALL = min(E_min, np.min(x_e_0_ALL))
x_E_max0_ALL = max(E_max, np.max(x_e_0_ALL))
x_E_min4_ALL = min(E_min, np.min(x_e_4_ALL))
x_E_max4_ALL = max(E_max, np.max(x_e_4_ALL))

x_E_min_total = min(x_E_min0_ALL, x_E_min4_ALL, x_E_min0, x_E_min4)
x_E_min_total = max(x_E_min_total, 1e-4)
x_E_max_total = max(x_E_max0_ALL, x_E_max4_ALL, x_E_max0, x_E_max4)

bins = np.geomspace(x_E_min_total, x_E_max_total, num_bins + 1)
bin_centers = np.sqrt(bins[:-1] * bins[1:])
bin_widths = np.diff(bins)

fig = plt.figure(figsize=(6.5, 5.0))
grid_spacing = GridSpec(2, 2, height_ratios=[3, 1], hspace=0)
upper_left = fig.add_subplot(grid_spacing[0, 0])  
lower_left = fig.add_subplot(grid_spacing[1, 0], sharex=upper_left)   
upper_right = fig.add_subplot(grid_spacing[0, 1])  
lower_right = fig.add_subplot(grid_spacing[1, 1], sharex=upper_right) 

values0 = np.array([p.E for evt in filtered_particles0 for p in evt])
values4 = np.array([p.E for evt in filtered_particles4 for p in evt])
values0_ALL = np.array([p.E for evt in filtered_particles_ALL_0 for p in evt])
values4_ALL = np.array([p.E for evt in filtered_particles_ALL_4 for p in evt])
x_e_0_values = np.divide(values0, E_cm)
x_e_4_values = np.divide(values4, E_cm)
x_e_0_ALL_values = np.divide(values0_ALL, E_cm)
x_e_4_ALL_values = np.divide(values4_ALL, E_cm)

counts0, _  = np.histogram(x_e_0_values, bins=bins) # raw counts
counts4, _  = np.histogram(x_e_4_values, bins=bins) # raw counts
counts0_ALL, _  = np.histogram(x_e_0_ALL_values, bins=bins) # raw counts
counts4_ALL, _  = np.histogram(x_e_4_ALL_values, bins=bins) # raw counts

density0 = counts0 / (num_events * bin_widths) # 1/N_events * dN/dE
density4 = counts4 / (num_events * bin_widths) # 1/N_events * dN/dE
density0_ALL = counts0_ALL / (num_events * bin_widths) # 1/N_events * dN/dE
density4_ALL = counts4_ALL / (num_events * bin_widths) # 1/N_events * dN/dE

abs_err_dens0 = np.sqrt(counts0) / (num_events * bin_widths) # Poisson errors
abs_err_dens4 = np.sqrt(counts4) / (num_events * bin_widths) # Poisson errors
abs_err_dens0_ALL = np.sqrt(counts0_ALL) / (num_events * bin_widths) # Poisson errors
abs_err_dens4_ALL = np.sqrt(counts4_ALL) / (num_events * bin_widths) # Poisson errors
x_error = 0.5 * bin_widths

ratio = np.full_like(density0, np.nan, dtype=float)
ratio_ALL = np.full_like(density0_ALL, np.nan, dtype=float)
good_mask = (density0 > 0) & (density4 > 0)
good_mask_ALL = (density0_ALL > 0) & (density4_ALL > 0)

np.divide(density4, density0, out=ratio, where=good_mask)
np.divide(density4_ALL, density0_ALL, out=ratio_ALL, where=good_mask_ALL)

ratio_err = np.zeros_like(ratio)
ratio_err[good_mask] = ratio[good_mask] * np.sqrt(
    (abs_err_dens0[good_mask] / density0[good_mask])**2 +
    (abs_err_dens4[good_mask] / density4[good_mask])**2)
ratio_err_ALL = np.zeros_like(ratio)
ratio_err_ALL[good_mask_ALL] = ratio_ALL[good_mask_ALL] * np.sqrt(
    (abs_err_dens0_ALL[good_mask_ALL] / density0_ALL[good_mask_ALL])**2 +
    (abs_err_dens4_ALL[good_mask_ALL] / density4_ALL[good_mask_ALL])**2)

################################################################
# plotting
# lower right - ratio for just recom
lower_right.errorbar(bin_centers, ratio, yerr=ratio_err, xerr=x_error,
                fmt='o', ms='5',
                label="Mlevelmax4 / Mlevelmax0")
lower_right.fill_between(bin_centers, ratio - ratio_err, ratio + ratio_err,
                   color='C2', alpha=0.2)
#for y_line in (0, 1, 2):
 #   lower_right.axhline(y_line, ls="--", color="gray")

# lower left - ratio for both recom and fragm
lower_left.errorbar(bin_centers, ratio_ALL, yerr=ratio_err_ALL, xerr=x_error,
                fmt='o', ms='5',
                label="Mlevelmax4 / Mlevelmax0")
lower_left.fill_between(bin_centers, ratio_ALL - ratio_err_ALL, ratio_ALL + ratio_err_ALL,
                   color='C2', alpha=0.2)
#for y_line in (0, 1, 2):
  #  lower_left.axhline(y_line, ls="--", color="gray")

# upper right - energy sprectrum for just recom
upper_right.errorbar(bin_centers, density0, yerr=abs_err_dens0, xerr=x_error,
                fmt='o', ms='5',
                label="Mlevelmax0")
upper_right.fill_between(bin_centers, density0 - abs_err_dens0, density0 + abs_err_dens0,
                   color='C0', alpha=0.2)
upper_right.errorbar(bin_centers, density4, yerr=abs_err_dens4, xerr=x_error,
               fmt='o', ms='5',
               label="Mlevelmax4")
upper_right.fill_between(bin_centers, density4 - abs_err_dens4, density4 + abs_err_dens4,
                   color='C1', alpha=0.2)

# upper left - energy sprectrum for both recom and fragm
upper_left.errorbar(bin_centers, density0_ALL, yerr=abs_err_dens0_ALL, xerr=x_error,
                fmt='o', ms='5',
                label="Mlevelmax0")
upper_left.fill_between(bin_centers, density0_ALL - abs_err_dens0_ALL, density0_ALL + abs_err_dens0_ALL,
                   color='C0', alpha=0.2)
upper_left.errorbar(bin_centers, density4_ALL, yerr=abs_err_dens4_ALL, xerr=x_error,
               fmt='o', ms='5',
               label="Mlevelmax4")
upper_left.fill_between(bin_centers, density4_ALL - abs_err_dens4_ALL, density4_ALL + abs_err_dens4_ALL,
                   color='C1', alpha=0.2)

lower_left.legend(frameon=False, fontsize=8)
lower_right.legend(frameon=False, fontsize=8)

lower_left.set_xlabel(r'$x_{E}$')
upper_left.set_xlabel(r'$x_{E}$')
lower_right.set_xlabel(r'$x_{E}$')
upper_right.set_xlabel(r'$x_{E}$')
lower_left.set_xscale('log')
upper_left.set_xscale('log')
lower_right.set_xscale('log')
upper_right.set_xscale('log')

upper_left.set_ylabel(r'$1/N_\mathrm{events}\,dN/dx_E$')
upper_left.set_yscale('log')
upper_left.legend(frameon=False, fontsize=8)
upper_right.set_ylabel(r'$1/N_\mathrm{events}\,dN/dx_E$')
upper_right.set_yscale('log')
upper_right.legend(frameon=False, fontsize=8)

for axes in (upper_left, lower_left, upper_right, lower_right):
    axes.tick_params(direction="in", which="both", top=True, right=True)

# make the tick marks on the density graph not appear
plt.setp(upper_left.get_xticklabels(), visible=False)
plt.setp(upper_right.get_xticklabels(), visible=False)

upper_left.set_xlim(x_E_min_total, x_E_max_total)
lower_left.set_xlim(x_E_min_total, x_E_max_total)
upper_right.set_xlim(x_E_min_total, x_E_max_total)
lower_right.set_xlim(x_E_min_total, x_E_max_total)
upper_right.set_title("Only Recombination")
upper_left.set_title("Fragmentation and Recombination")

fig.suptitle(r'$x_{E}$ spectrum and Ratio Plot for $\Upsilon(1\mathrm{S})$ mesons')
fig.tight_layout()
fig.savefig(f"x_e_spectrum_ratio_Upsilon_mesons_multi_graph.png", dpi=600)
plt.close(fig)