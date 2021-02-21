# IMPORT LIBRARIES

import	os
import	argparse
import	datetime
import	time
from	time					import sleep
import	sys	
from	sys						import stdout
import	numpy 					as np
import	matplotlib.pyplot		as plt
from	matplotlib				import rc
from	matplotlib.ticker		import FormatStrFormatter, MultipleLocator, FuncFormatter
from	matplotlib.gridspec		import GridSpec
import	matplotlib.ticker		as tck
import	matplotlib.colors		as colors
import	matplotlib.patches		as patches
from	timeit					import default_timer as timer
from	astropy.modeling 		import models, fitting
from	scipy.optimize			import curve_fit
from	scipy.integrate 		import quad, ode

#######################################################################################################
#######################################################################################################

# LATEX: ON, MNRAS template

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif" : ["Times New Roman"],
	"font.size"  : 10})	

#######################################################################################################
#######################################################################################################

# CMDLINE OPTIONS, ERROR MSG

if len(sys.argv) > 2:
	sys.exit('ARGV ERROR, TRY:   python   Analyse_C2.py'+'\n'+'ARGV ERROR, TRY:   python   Analyse_C2.py   savesnaps=Y/N'+'\n')

if len(sys.argv) == 1:
	save = 'savesnaps=N'
elif len(sys.argv) == 2:
	save = sys.argv[1]

#######################################################################################################
#######################################################################################################

# FUNCTIONS

# The tree inverse CDFs for the homogeneous sphere:

def R_P(P,a):
	return a * np.cbrt(P)

def Ph_P(P):
	return 2 * np.pi * P

def Th_P(P):
	return np.arccos(1. - 2. * P)

# The tree PDFs for the homogeneous sphere:

def pdf_r(r,a):
	return 3. * r**2 / (a**3)

def pdf_hern_r(r,a):
	return 2. * a * r / ((a + r)**3)

def pdf_ph(ph):
	return 0.5 / np.pi * (1. + 0. * ph)

def pdf_th(th):
	return 0.5 * np.sin(th)

def pdf_plum(r,a):
	return 3 * r**2 / a**3 / (1 + r**2 / a**2)**(5/2)

# The circular velocity of the Plummer sphere, and derivative formulae

def v_circ(r):
	return r * np.sqrt( rho * 4. * np.pi / 3. )

def v_case_1(r):
	return 0.05 * v_circ(r)

def v_case_2(r):
	return 0.05 * a * v_circ(a) / r

def pdf_v_1(v):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. )
	return 3. / ( (k * a * V0)**3 ) * v**2 

def pdf_v_2(v):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. ) * (a**2) * V0 * R0
	return 3. * (k**3) / (a**3 * R0**3) / (v**4)

def pdf_l_1(l):
	k = 0.05 * np.sqrt( rho * 4. * np.pi / 3. ) * V0 / R0
	return 1.5 / (a**3 * R0**3) / (k**(3./2.)) * l**(1./2.) 

# A plotting primer for PDFs over histograms

def histo_pdf_plotter(ax, x_lim, x_step, x_bins, func, npar, x_min=0.):
	x = np.linspace(x_min, x_lim, 10000)
	if npar == 1:
		f_x = func(x, a)
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		f_x = func(x)
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)):
		x = []
		f_x = []
		e_x = []
		x_mid = x_bins[j] + x_step / 2
		for i in range(9):
			x_temp = x_bins[j] + x_step / 2 + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, a)
			elif npar == 0:
				f_temp = func(x_mid)
			e_temp = np.sqrt(f_temp * I) / I
			# ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		# ax.plot(x, f_x + e_x, color='black', lw=0.9)
		# ax.plot(x, f_x - e_x, color='black', lw=0.9)

def histo_pdf_plotter_log(ax, x_min, x_max, x_bins, func, npar):
	log_min = np.log10(x_min)
	log_max = np.log10(x_max)
	x = np.logspace(log_min, log_max, 1000)
	if npar == 1:
		f_x = func(x, aa) # * N
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		f_x = func(x) # * N
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)-1):
		x = []
		f_x = []
		e_x = []
		x_mid = (x_bins[j+1] + x_bins[j]) / 2.
		x_step = x_bins[j+1] - x_bins[j]
		for i in range(9):
			x_temp = x_mid + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, aa) #* N
			elif npar == 0:
				f_temp = func(x_mid) # * N
			e_temp = np.sqrt(f_temp * I) / I # np.sqrt(f_temp) # 
			# ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		# ax.plot(x, f_x + e_x, color='black', lw=0.9)
		# ax.plot(x, f_x - e_x, color='black', lw=0.9)

def histo_pdf_norm_log(ax, x_min, x_max, x_bins, func, npar):
	log_min = np.log10(x_min)
	log_max = np.log10(x_max)
	x = np.logspace(log_min, log_max, 1000)
	if npar == 1:
		norm , fuffa = quad(func, x_max, x_max, aa)
		f_x = func(x, a) # * N / norm
		ax.plot(x, f_x, color='black', lw=0.9)
	elif npar == 0:
		norm , fuffa = quad(func, x_max, x_max)
		f_x = func(x) / norm
		ax.plot(x, f_x, color='black', lw=0.9)
	for j in range(len(x_bins)-1):
		x = []
		f_x = []
		e_x = []
		x_mid = (x_bins[j+1] + x_bins[j]) / 2.
		x_step = x_bins[j+1] - x_bins[j]
		for i in range(9):
			x_temp = x_mid + x_step / 15 * (i-4)
			if npar == 1:
				f_temp = func(x_mid, a) #* N
			elif npar == 0:
				f_temp = func(x_mid) # * N
			e_temp = np.sqrt(f_temp * I) / I # np.sqrt(f_temp) # 
			# ax.vlines(x_mid, ymin=f_temp-e_temp, ymax=f_temp+e_temp, color='black', lw=0.9)
			x.append(x_temp)
			f_x.append(f_temp)
			e_x.append(e_temp)
		x = np.array(x)
		f_x = np.array(f_x)
		e_x = np.array(e_x)
		# ax.plot(x, f_x + e_x, color='black', lw=0.9)
		# ax.plot(x, f_x - e_x, color='black', lw=0.9)


#######################################################################################################
#######################################################################################################

time_prog_start = timer()

print()

plotfile = 'OUT_OCT_Exam_C2_10000.txt'

# Set up to obtain the accurate conversion factors from internal units to physical units:

G_cgs = 6.67430e-8			# cm^3 g^-1 s^-2
pc_cgs = 3.08567758e18		# cm
Msun_cgs = 1.98855e33 		# g
Myr_cgs = 31557600. * 1e6 	# s

# Conversion factors from internal units to the chosen physical units:

G0 = 1.
R0 = 5.													# pc
M0 = 1e4												# Msun
V0 = np.sqrt( G_cgs * (Msun_cgs * M0) / (pc_cgs * R0) )	# cm/s
T0 = (pc_cgs * R0) / V0									# s
T0 = T0 / Myr_cgs										# Myr
V0 = V0 / 1e5 											# km/s

a      = 1. * R0
lim_3d = 2. * R0

#######################################################################################################
#######################################################################################################

# OPEN SIMULATION OUTPUT FILE AND SAVE DATA INTO ARRAYS 

file = open(plotfile, "r")
print('Extracting data from: {:}'.format(plotfile))
firstrow = (file.readline()).strip("\n")
NL = 1
for line in file:
	NL += 1
file.close()

I = int(firstrow)				# N_particles
L = 3+4*I						# N_lines for each timestep
NT = int(NL / L)				# N_timesteps	
X = np.zeros((I,4,NT))			# Empty array for the positions at each t
V = np.zeros((I,4,NT))			# Empty array for the velocities at each t
P = np.zeros((I,NT))			# Empty array for the potentials at each t
K = np.zeros((I,NT))			# Empty array for the kinetic energies at each t
E = np.zeros((I,NT))			# Empty array for the energies at each t
M = np.zeros((I,NT))			# Empty array for the masses at each t (should be const)
N = np.zeros(NT)				# Empty array for the N_particles at each t (should be const)
T = np.zeros(NT)				# Empty array for the times t

file = open(plotfile, "r")		# Read data!
i = 0
t = 0
for line in file:
	linje = line.strip("\n")
	j = i % L
	if j == 0:
		N[t] = float(linje)
	elif j == 2:
		T[t] = float(linje) * T0
	elif j >= 3 and j < (3+I): 
		m = j-3
		M[m,t] = float(linje) * M0
	elif j >= (3+I) and j < (3+2*I):
		m = j - (3+I)
		b = linje.split()
		for k in range(len(b)):
			X[m,k+1,t] = float(b[k]) * R0
	elif j >= (3+2*I) and j < (3+3*I):
		m = j - (3+2*I)
		b = linje.split()
		for k in range(len(b)):
			V[m,k+1,t] = float(b[k]) * V0
	elif j >= (3+3*I) and j < (3+4*I):
		m = j - (3+3*I)
		P[m,t] = float(linje) * M0 / R0
		if (j+1) == (3+4*I):
			t += 1
			if t == NT:
				break
	i += 1
file.close()

print()
print("Number of bodies: {:d}".format(I))
print()
print('Conversion factors to physical units:')
print('1 r_IU = {:} pc'.format(R0))
print('1 m_IU = {:} M_sun'.format(M0))
print('1 v_IU = {:} km/s'.format(V0))
print('1 t_IU = {:} Myr'.format(T0))

M_tot = 0.
for i in range(I):
	M_tot += M[i,0]

rho_iu = 1. / ( 4. * np.pi / 3.) # / * 1.**3

# Find collapse time and compare it to the theoretical expectation

T_collapse = np.sqrt( 3. * np.pi / 32. / rho_iu ) * T0
t_collapse_m = np.argmax(T > T_collapse)
t_collapse_M = np.argmin(T < T_collapse)
t_collapse = 0.
if np.abs(T[t_collapse_m] - T_collapse) < np.abs(T[t_collapse_M] - T_collapse):
	t_collapse = t_collapse_m
else:
	t_collapse = t_collapse_M

T_collapse_sim = T[t_collapse]

print("DT/T = {:.4e}".format((T_collapse-T_collapse_sim)/T_collapse))

rho = M_tot / ( 4. * np.pi / 3. * a**3.)

print()
print('M_tot  = {:.0f} Msun'.format(M_tot))
print('m_i    = {:.0f}     Msun'.format(M[37,0]))

time_prog_load = timer()

print()
print("Data loading time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_load - time_prog_start)))
print()

#######################################################################################################
#######################################################################################################

# Let us fill the 0-th component (at any t) of each particle's position and velocity with its module

P_tot = np.zeros((NT))
K_tot = np.zeros((NT))
E_tot = np.zeros((NT))

for t in range(NT):
	for i in range(I):
		X[i,0,t] = np.sqrt(X[i,1,t]**2 + X[i,2,t]**2 + X[i,3,t]**2)
		V[i,0,t] = np.sqrt(V[i,1,t]**2 + V[i,2,t]**2 + V[i,3,t]**2)
		P[i,t] = P[i,t] * G_cgs * Msun_cgs / pc_cgs
		K[i,t] = 0.5 * (V[i,0,t])**2 * 1e10
		E[i,t] = P[i,t] + K[i,t]
		P_tot[t] += 0.5 * P[i,t]
		K_tot[t] += K[i,t]
		E_tot[t] += 0.5 * P[i,t] + K[i,t]

# find some useful limits for the plots - obsolete (?)

t_max = np.max(T)
r_max = 0.
v_max = 0.
for i in range(I):
	for t in range(NT):
		r_max = np.amax(np.array([ np.amax( X[i,0,t] ) , r_max ]))
		v_max = np.amax(np.array([ np.amax( V[i,0,t] ) , v_max ]))
r_max = 1.1 * r_max
v_max = 1.1 * v_max

#######################################################################################################
#######################################################################################################

# PLOT LAGRANGIAN RADII AS FUNTIONS OF TIME - TOT SYSTEM R.F.

fig_LR , ax_LR = plt.subplots(figsize=(5.5,5.5))
factor_tr = t_max / lim_3d
ax_LR.set_aspect(factor_tr)
ax_LR.grid(linestyle=':', which='both')
ax_LR.set_xlim(0, t_max)
ax_LR.set_ylim(0, lim_3d)
ax_LR.set_title('Lagrangian radii as functions of time',fontsize=10)
ax_LR.set_xlabel(r'$t\;$[Myr]')
ax_LR.set_ylabel(r'$r\;$[pc]') # , rotation='horizontal', horizontalalignment='right'

RL = np.zeros((9,NT))
for k in range(9):
	for t in range(NT):
		C = np.copy(X[:,0,t])
		C = np.sort(C)
		RL[k,t] = C[int(np.ceil(I/10*(k+1))-1)]
	ax_LR.plot(T , RL[k,:] , linestyle='' , marker='o' , markersize=0.75, label='{:d}0'.format(k+1) + r'$\%\; M_{tot}$') 
ax_LR.vlines(T_collapse, ymin=0, ymax=lim_3d, linestyle='--', color='black', label=r'$T_{coll} = \sqrt{\frac{3\pi}{32 G \rho_{0}}}$')

ax_LR.legend(frameon=True, bbox_to_anchor=(1.01,1), title=r'$\begin{array}{rcl} \;\;N \!\!&\!\! = \!\!&\!\! 10^{4} \\ M_{tot} \!\!&\!\! = \!\!&\!\! 10^{4} \, M_{\odot} \\ \;\;a & = & 5 \; \mathrm{pc} \end{array}$'+'\n',fontsize=10)

fig_LR.tight_layout()

fig_LR.savefig("C2_Results_PNG/Lagrangian_Radii_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_LR.savefig("C2_Results_EPS/Lagrangian_Radii_{:s}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Lagrangian radii, RF 0')
print()

##################################################################################################

# PLOT ENERGIES (TOT, KIN, POT) AS FUNCTIONS OF TIME

fig_E , ax_E = plt.subplots(figsize=(5.5,5.5))

ax_E.grid(linestyle=':')
ax_E.set_xlim(0, t_max)
# ax_E.set_ylim(-1e17,5e17)
# ax_E.set_aspect(t_max / 6e17)
ax_E.set_title('Total energies as functions of time\n',fontsize=10)
ax_E.set_xlabel(r'$t\;$[Myr]')
ax_E.set_ylabel(r'$E\;$[erg/g]') # , rotation='horizontal', horizontalalignment='right'

ax_E.plot(T, E_tot, linestyle=':',  color='black', markersize=1, label=r'$E_{tot}$')
ax_E.plot(T, P_tot, linestyle='-.', color='black', markersize=1, label=r'$E_{pot}$')
ax_E.plot(T, K_tot, linestyle='--', color='black', markersize=1, label=r'$E_{kin}$')
# ax_E.plot(T, 2*K_tot + P_tot, linestyle=(0, (3, 5, 1, 5, 1, 5)) , color='black' , markersize=1, label=r'$2E_{kin}+E_{pot}$')
ax_E.legend(frameon=True, bbox_to_anchor=(1.01,1))
fig_E.tight_layout()

fig_E.savefig("C2_Results_PNG/Energy_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_E.savefig("C2_Results_EPS/Energy_{:s}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Energy, RF 0')

##################################################################################################

# EVOLUTION OF THE POTENTIAL AS A FUNCTION OF THE RADIAL COORD. - TOT SYSTEM R.F.

fig_Phi = plt.figure(figsize=(5.5,5.5))
fig_Phi.suptitle('Evolution of the potential as a function of radius',fontsize=10)
gs = GridSpec(2, 2, figure=fig_Phi)
ax_phi = []
ax_phi.append(fig_Phi.add_subplot(gs[0,0]))
ax_phi.append(fig_Phi.add_subplot(gs[0,1]))
ax_phi.append(fig_Phi.add_subplot(gs[1,0]))
ax_phi.append(fig_Phi.add_subplot(gs[1,1]))

tt = [0, int(np.floor(NT/4)), int(np.floor(NT/3)), int(np.floor(NT/2))]
for i in range(4):
	ttt = tt[i]
	ax_phi[i].grid(linestyle=':', which='both')
	ax_phi[i].set_xlim(0,a)
	# ax_phi[i].set_ylim(-4e12,0)
	# ax_phi[i].set_aspect(t_max / 4e12)
	ax_phi[i].set_title('\nPotential at $t$ = {:.3f} Myr\n'.format(T[ttt]))
	ax_phi[i].set_xlabel(r'$r\;$[pc]')
	ax_phi[i].set_ylabel(r'$\Phi\;$[erg/g]')
	ax_phi[i].scatter(X[:,0,ttt], P[:,ttt], color='lightgrey', s=0.5, label=r'$\Phi(r)\,:\;simulation$')
	ax_phi[i].set_ylim(None,0)

fig_Phi.tight_layout()

fig_Phi.savefig("C2_Results_PNG/Potential_t_sample_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_Phi.savefig("C2_Results_EPS/Potential_t_sample_{:s}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Potential profile, RF 0')
print()

##################################################################################################
##################################################################################################

# PLOT ANGULAR MOMENTA AS FUNCTIONS OF TIME (MEAN, TOT, AT LAG_RADII) - TOT SYSTEM R.F.

fig_l , ax_l = plt.subplots(figsize=(5.5,5.5))

# Compute total angular momentum and mean angular momentum module

l = np.zeros((I,4,NT))
l_tot = np.zeros((4,NT))
l_mean = np.zeros(NT)
for t in range(NT):
	for i in range(I):
		l[i,1,t] = X[i,2,t] * V[i,3,t] - X[i,3,t] * V[i,2,t]
		l[i,2,t] = X[i,3,t] * V[i,1,t] - X[i,1,t] * V[i,3,t]
		l[i,3,t] = X[i,1,t] * V[i,2,t] - X[i,2,t] * V[i,1,t]
		l[i,0,t] = np.sqrt(l[i,1,t]**2 + l[i,2,t]**2 + l[i,3,t]**2)
		l_tot[1,t] += l[i,1,t] 
		l_tot[2,t] += l[i,2,t]
		l_tot[3,t] += l[i,3,t]
		l_mean[t]  += l[i,0,t] / I
	l_tot[0,t] = np.sqrt(l_tot[1,t]**2 + l_tot[2,t]**2 + l_tot[3,t]**2) / I

# Compute mean angular momentum at the lagrangian radii - OBSOLETE (it varies too much for a decent plot)

l_averaged = np.zeros((9,4,NT))
for k in range(9):
	for t in range(NT):
		C = np.copy(X[0:I-1,0,t])
		D1 = np.copy(l[0:I-1,1,t])
		D2 = np.copy(l[0:I-1,2,t])
		D3 = np.copy(l[0:I-1,3,t])
		sort_index = np.argsort(C)
		D1 = D1[sort_index]
		D2 = D2[sort_index]
		D3 = D3[sort_index]
		for j in range(200):
			i_new = int(np.ceil(I/10*(k+1))-100+j)
			l_averaged[k,1,t] += D1[i_new] / 200 
			l_averaged[k,2,t] += D1[i_new] / 200 
			l_averaged[k,3,t] += D1[i_new] / 200 
		l_averaged[k,0,t] = np.sqrt( l_averaged[k,1,t]**2 + l_averaged[k,2,t]**2 + l_averaged[k,3,t]**2 )

ax_l.grid(linestyle=':',which='both')
ax_l.set_xlim(0, t_max)
ax_l.set_ylim(1e-1,2e1)
ax_l.set_title('Average angular momentum as a function of time\n',fontsize=10)
ax_l.set_xlabel(r'$t\;$[Myr]')
ax_l.set_ylabel(r'$l\;$[pc km/s]')
ax_l.set_yscale('log')

'''
for k in [0,2,4,6,8]: # range(9):
	ax_l.plot(T, l_averaged[k,0,:], ls=':', label=r'$\langle l(r=R_{Lag}^{' + '{:d}'.format(int(10*(k+1))) + r'\%})\rangle$')
'''

ax_l.plot(T, l_tot[0,:], color='black', label=r'$|\,\vec{l}_{tot} \,| = \frac{1}{N} \sum_i{\vec{l}_i}\,|$')
ax_l.plot(T, l_mean, color='black' , ls='--', label=r'$\langle \,|\,\vec{l}\,|\, \rangle  = \frac{1}{N} \sum_i{|\vec{l}_i\,|}$')
ax_l.vlines(T_collapse, ymin=1e-1, ymax=2e1, linestyle=':', color='black', label=r'$T_{coll} = \sqrt{\frac{3\pi}{32 G \rho_{0}}}$')

ax_l.legend(frameon=True, loc=2)
fig_l.tight_layout()

fig_l.savefig("C2_Results_PNG/Angular_Momentum_{:}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_l.savefig("C2_Results_EPS/Angular_Momentum_{:}.eps".format(plotfile), bbox_inches='tight')

print()
print('Fig saved: Angular momentum, RF 0')
print()

##################################################################################################
##################################################################################################

# CHANGE REF. FRAME TO THAT OF THE C.M OF THE REMNANT, AT ALL TIMES (DO IT X10)

time_prog_CM_1 = timer()

_X_cm = np.zeros((4,NT))
_V_cm = np.zeros((4,NT))

for rep in range(10):

	X_cm = np.zeros((4,NT))
	V_cm = np.zeros((4,NT))
	M_cm = np.zeros((NT))
	K_cm = np.zeros((NT))
	K_dot = np.zeros((NT))

	R70 = np.zeros(NT)
	for t in range(NT):
		C = np.copy(X[:,0,t])
		C = np.sort(C)
		R70[t] = C[int(np.ceil(I/100*70))]
	
	for t in range(NT):
		for i in range(I-1):
			if X[i,0,t] < R70[t]:
				for j in range(3):
					X_cm[j+1,t] += X[i,j+1,t] * M[i,t]
					V_cm[j+1,t] += V[i,j+1,t] * M[i,t]
				M_cm[t] += M[i,t]
		for j in range(3):
			X_cm[j+1,t] = X_cm[j+1,t] / M_cm[t]
			V_cm[j+1,t] = V_cm[j+1,t] / M_cm[t]
			_X_cm[j+1,t] += X_cm[j+1,t]
			_V_cm[j+1,t] += V_cm[j+1,t]
		X_cm[0,t] = np.sqrt(X_cm[1,t]**2 + X_cm[2,t]**2 + X_cm[3,t]**2)
		V_cm[0,t] = np.sqrt(V_cm[1,t]**2 + V_cm[2,t]**2 + V_cm[3,t]**2)
		K_cm[t] = 0.5 * (V_cm[0,t])**2 * 1e10
		K_dot[t] += ( V[i,1,t] * V_cm[1,t] + V[i,2,t] * V_cm[2,t] + V[i,3,t] * V_cm[3,t] ) * 1e10

	P_rf = np.zeros((NT))
	K_rf = np.zeros((NT))
	E_rf = np.zeros((NT))

	for i in range(I):
		for j in range(3):
			X[i,j+1,:] = X[i,j+1,:] - X_cm[j+1,:]
			V[i,j+1,:] = V[i,j+1,:] - V_cm[j+1,:]
		X[i,0,:] = np.sqrt(X[i,1,:]**2 + X[i,2,:]**2 + X[i,3,:]**2)
		V[i,0,:] = np.sqrt(V[i,1,:]**2 + V[i,2,:]**2 + V[i,3,:]**2)
		# K[i,:] = 0.5 * (V[i,0,:])**2 * 1e10
		E[i,:] = P[i,:] + K[i,:]
		P_rf[:] += 0.5 * P[i,:]
		K_rf[:] += K[i,:]
	E_rf[:] = P_rf[:] + K_rf[:]

R_35_t = np.zeros(NT)
R_75_t = np.zeros(NT)
R_50_t = np.zeros(NT)
R_65_t = np.zeros(NT)

for t in range(NT):
	C = np.copy(X[:,0,t])
	C = np.sort(C)
	R_35_t[t] = C[int(np.ceil(I/10*3.5)-1)]
	R_75_t[t] = C[int(np.ceil(I/10*7.5)-1)]
	R_50_t[t] = C[int(np.ceil(I/10*5.0)-1)]
	R_65_t[t] = C[int(np.ceil(I/10*6.5)-1)]

t_test    = int(2.3 * np.argmin(R_35_t[:]))
r_35_test = np.amin(R_35_t[:])
r_35_end  = R_35_t[-1]
r_50_end  = R_50_t[-1]
r_65_end  = R_65_t[-1]
r_75_end  = R_75_t[-1]

time_prog_CM_2 = timer()

print("Remnant CM computation time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_CM_2 - time_prog_CM_1)))
print()

#######################################################################################################
#######################################################################################################

# PLOT LAGRANGIAN RADII AS FUNTIONS OF TIME - REMNANT C.M. R.F.

fig_LRCM , ax_LRCM = plt.subplots(figsize=(5.5,5.5))
factor_tr = t_max / lim_3d
ax_LRCM.set_aspect(factor_tr)
ax_LRCM.grid(linestyle=':', which='both')
ax_LRCM.set_xlim(0, t_max)
ax_LRCM.set_ylim(0, lim_3d)
ax_LRCM.set_title('Lagrangian radii as functions of time - Remnant R.F.',fontsize=10)
ax_LRCM.set_xlabel(r'$t\;$[Myr]')
ax_LRCM.set_ylabel(r'$r\;$[pc]')


RLCM = np.zeros((9,NT))
for k in range(9):
	for t in range(NT):
		C = np.copy(X[:,0,t])
		C = np.sort(C)
		RLCM[k,t] = C[int(np.ceil(I/10*(k+1))-1)]
	ax_LRCM.plot(T , RLCM[k,:] , linestyle='' , marker='o' , markersize=0.75, label='{:d}0'.format(k+1) + r'$\%\; M_{tot}$') # 
ax_LRCM.vlines(T_collapse, ymin=0, ymax=lim_3d, linestyle='--', color='black', label=r'$T_{coll} = \sqrt{\frac{3\pi}{32 G \rho_{0}}}$')
	
ax_LRCM.legend(frameon=True, bbox_to_anchor=(1.01,1), title=r'$\begin{array}{rcl} \;\;N \!\!&\!\! = \!\!&\!\! 10^{4} \\ M_{tot} \!\!&\!\! = \!\!&\!\! 10^{4} \, M_{\odot} \\ \;\;a & = & 5 \; \mathrm{pc} \end{array}$'+'\n',fontsize=10)

fig_LRCM.tight_layout()

fig_LRCM.savefig("C2_Results_PNG/Lagrangian_Radii_CM_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_LRCM.savefig("C2_Results_EPS/Lagrangian_Radii_CM_{:s}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Lagrangian radii, RF CM')
print()

##################################################################################################

# PLOT ENERGIES (TOT, KIN, POT) AS FUNCTIONS OF TIME, CHECK VIRIAL EQUILIBRIUM - REMNANT C.M. R.F.

fig_ECM , ax_ECM = plt.subplots(figsize=(5.5,5.5))

ax_ECM.grid(linestyle=':')
ax_ECM.set_xlim(0, t_max)
# ax_ECM.set_ylim(-1e17,5e17)
# ax_ECM.set_aspect(t_max / 6e17)
ax_ECM.set_title('Total energies as functions of time - Remnant R.F.\n',fontsize=10)
ax_ECM.set_xlabel(r'$t\;$[Myr]')
ax_ECM.set_ylabel(r'$E\;$[erg/g]')

ax_ECM.plot(T, E_rf, linestyle=':',  color='black', markersize=1, label=r'$E_{tot}$')
ax_ECM.plot(T, P_rf, linestyle='-.', color='black', markersize=1, label=r'$E_{pot}$')
ax_ECM.plot(T, K_rf, linestyle='--', color='black', markersize=1, label=r'$E_{kin}$')

E_Vir = np.zeros(NT)
for t in range(NT):
	for i in range(I):
		if X[i,0,t] < R_75_t[t]:
			E_Vir[t] += 0.5 * P[i,t] + 2. * K[i,t] 
ax_ECM.plot(T, E_Vir, color='black' , markersize=1, label=r'$2E_{kin}^{Remn}+E_{pot}^{Remn}$') # linestyle=(0, (3, 5, 1, 5, 1, 5)) 

ax_ECM.legend(frameon=True, loc=1)
fig_ECM.tight_layout()

fig_ECM.savefig("C2_Results_PNG/Energy_CM_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_ECM.savefig("C2_Results_EPS/Energy_CM_{:s}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Energy, RF CM')

##################################################################################################

# PLOT EVOLUTION OF THE POTENTIAL AS A FUNCTION OF THE RADIAL COORD. - REMNANT C.M. R.F.

fig_PhiCM = plt.figure(figsize=(6.5,6.5))
fig_PhiCM.suptitle('Evolution of the potential as a function of radius - Remnant R.F.',fontsize=10)
gs = GridSpec(2, 2, figure=fig_Phi)
ax_phiCM = []
ax_phiCM.append(fig_PhiCM.add_subplot(gs[0,0]))
ax_phiCM.append(fig_PhiCM.add_subplot(gs[0,1]))
ax_phiCM.append(fig_PhiCM.add_subplot(gs[1,0]))
ax_phiCM.append(fig_PhiCM.add_subplot(gs[1,1]))

tt = [0, int(np.floor(NT/4)), int(np.floor(NT/3)), int(np.floor(NT/2))]
for i in range(4):
	ttt = tt[i]
	ax_phiCM[i].grid(linestyle=':', which='both')
	ax_phiCM[i].set_xlim(0,a)
	ax_phiCM[i].set_title('\nPotential at $t$ = {:.3f} Myr\n'.format(T[ttt]),fontsize=10)
	ax_phiCM[i].set_xlabel(r'$r\;$[pc]')
	ax_phiCM[i].set_ylabel(r'$\Phi\;$[erg/g]')
	ax_phiCM[i].scatter(X[:,0,ttt], P[:,ttt], color='lightgrey', s=0.5, label=r'$\Phi(r)\,:\;simulation$')

ax_phiCM[0].set_xlim(0,a)
ax_phiCM[1].set_xlim(0,3.5)
ax_phiCM[2].set_xlim(0,1.5)
ax_phiCM[3].set_ylim(None,0)

fig_PhiCM.tight_layout()

fig_PhiCM.savefig("C2_Results_PNG/Potential_t_sample_CM_{:s}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_PhiCM.savefig("C2_Results_EPS/Potential_t_sample_CM_{:s}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Potential profile, RF CM')
print()

##################################################################################################
##################################################################################################

# CREATE SNAPSHOTS TO MAKE A MOVIE - REMNANT C.M. R.F. - OPTION FROM CMDLINE

if save == 'savesnaps=Y':
	time_snaps_1 = timer()

	for t in range(len(T)):
		snap = plt.figure(figsize=(10,10))
		ax_s = snap.add_subplot(111, projection='3d')
		ax_s.set_xlim(-1.1*a, + 1.1*a)
		ax_s.set_ylim(-1.1*a, + 1.1*a)
		ax_s.set_zlim(-1.1*a, + 1.1*a)
		ax_s.set_xlabel(r"$x\;$[pc]")
		ax_s.set_ylabel(r"$y\;$[pc]")
		ax_s.set_zlabel(r"$z\;$[pc]")
		ax_s.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax_s.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax_s.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax_s.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		ax_s.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		ax_s.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		ax_s.set_title('$t = {:.3f}$ Myr\n'.format(T[t]),fontsize=10)

		for i in range(I):
			ax_s.scatter(X[i,1,t] , X[i,2,t] , X[i,3,t] , s=0.05 , color='darkred')

		snap.savefig("C2_Snaps/snapshot_{:d}.png".format(t), bbox_inches='tight')
		plt.close(snap)
		stdout.write("\rSaving movie snapshots:     progress = {:3.2f} %".format(t/len(T)*100.))
	stdout.write("\rSaved all movie snapshots:  progress = {:3.2f} % \n".format(100.))

	time_snaps_2 = timer()

	print()
	print("Snapshots creation time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_snaps_2 - time_snaps_1)))
	print()

##################################################################################################
##################################################################################################

# PLOT ANGULAR MOMENTA AS FUNCTIONS OF TIME (MEAN, TOT, AT LAG_RADII) - REMNANT C.M. R.F. - BEWARE OF TRANSFORM. RULES WHEN CHANGING R.F.

fig_L , ax_L = plt.subplots(figsize=(5.5,5.5))

l      = np.zeros((I,4,NT))
l_tot  = np.zeros((4,NT))
l_mean = np.zeros(NT)


for t in range(NT):
	for i in range(I):
		l[i,1,t] = X[i,2,t] * V[i,3,t] - X[i,3,t] * V[i,2,t]
		l[i,2,t] = X[i,3,t] * V[i,1,t] - X[i,1,t] * V[i,3,t]
		l[i,3,t] = X[i,1,t] * V[i,2,t] - X[i,2,t] * V[i,1,t]
		l[i,0,t] = np.sqrt(l[i,1,t]**2 + l[i,2,t]**2 + l[i,3,t]**2)

		l_mean[t]  += l[i,0,t] / I

		# to check again that the ang mom is conserved and plot it, add X_cm, V_cm to X_i , V_i 
		l_tot[1,t] += (X[i,2,t] + _X_cm[2,t] ) * ( V[i,3,t] + _V_cm[3,t] ) - ( X[i,3,t] + _X_cm[3,t] ) * ( V[i,2,t] + _V_cm[2,t] )
		l_tot[2,t] += (X[i,3,t] + _X_cm[3,t] ) * ( V[i,1,t] + _V_cm[1,t] ) - ( X[i,1,t] + _X_cm[1,t] ) * ( V[i,3,t] + _V_cm[3,t] )
		l_tot[3,t] += (X[i,1,t] + _X_cm[1,t] ) * ( V[i,2,t] + _V_cm[2,t] ) - ( X[i,2,t] + _X_cm[2,t] ) * ( V[i,1,t] + _V_cm[1,t] )

	l_tot[0,t] = np.sqrt(l_tot[1,t]**2 + l_tot[2,t]**2 + l_tot[3,t]**2) / I


l_averaged = np.zeros((9,4,NT))
for k in range(9):
	for t in range(NT):
		C = np.copy(X[0:I-1,0,t])
		D1 = np.copy(l[0:I-1,1,t])
		D2 = np.copy(l[0:I-1,2,t])
		D3 = np.copy(l[0:I-1,3,t])
		sort_index = np.argsort(C)
		D1 = D1[sort_index]
		D2 = D2[sort_index]
		D3 = D3[sort_index]
		for j in range(200):
			i_new = int(np.ceil(I/10*(k+1))-100+j)
			l_averaged[k,1,t] += D1[i_new] / 200 
			l_averaged[k,2,t] += D1[i_new] / 200 
			l_averaged[k,3,t] += D1[i_new] / 200 
		l_averaged[k,0,t] = np.sqrt( l_averaged[k,1,t]**2 + l_averaged[k,2,t]**2 + l_averaged[k,3,t]**2 )

ax_L.grid(linestyle=':',which='both')
ax_L.set_xlim(0, t_max)
ax_L.set_ylim(0,2)		# (1e-1,1e1)
ax_L.set_aspect(t_max / 2.)
ax_L.set_title('Average angular momentum as a function of time - Remnant R.F.\n',fontsize=10)
ax_L.set_xlabel(r'$t\;$[Myr]')
ax_L.set_ylabel(r'$l\;$[pc km/s]')
# ax_L.set_yscale('log')

'''
for k in [0,2,4,6,8]: # range(9):
	ax_L.plot(T, l_averaged[k,0,:], ls=':', label=r'$\langle l(r=R_{Lag}^{' + '{:d}'.format(int(10*(k+1))) + r'\%})\rangle$')
'''
ax_L.plot(T, l_tot[0,:], color='black', label=r"$|\,\vec{l}_{tot} \,| = |\frac{1}{N} \sum_i{\,\vec{l}_i}\,|$")
ax_L.plot(T, l_mean, color='black' , ls='--', label=r"$\langle \,|\,\vec{l}'\,|\, \rangle  = \frac{1}{N} \sum_i{|\,\vec{l}_i'\,|}$")

ax_L.vlines(T_collapse, ymin=0., ymax=20., linestyle=':', color='black', label=r'$T_{coll} = \sqrt{\frac{3\pi}{32 G \rho_{0}}}$')

ax_L.legend(frameon=True, loc=7)
fig_L.tight_layout()

fig_L.savefig("C2_Results_PNG/Angular_Momentum_CMRF_{:}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_L.savefig("C2_Results_EPS/Angular_Momentum_CMRF_{:}.eps".format(plotfile), bbox_inches='tight')

print()
print('Fig saved: Angular momentum, RF CM')
print()

##################################################################################################
##################################################################################################

# PLOT VELOCITY COLOR MAPS ON PERPENDICULAR PLANE PROJECTION - REMNANT C.M. R.F.

fig_VX , ax_VX = plt.subplots(figsize=(5.5,5.5))
fig_VY , ax_VY = plt.subplots(figsize=(5.5,5.5))
fig_VZ , ax_VZ = plt.subplots(figsize=(5.5,5.5))
fig_V  = [fig_VX , fig_VY, fig_VZ]
ax_V   = [ax_VX, ax_VY, ax_VZ]
sc_V   = []
cbar_V = []

axis_n = ['', '$x$', '$y$', '$z$']
cbar_n = ['', r'$v_{x}$', r'$v_{y}$', r'$v_{z}$']
index_1 = [1,2,3]
index_2 = [2,3,1] 
index_3 = [3,1,2]

tt = -1 # -1
X_tt = [[],[],[],[]]
V_tt = [[],[],[],[]]
for i in range(I):
	if X[i,0,tt] <= R_75_t[tt]:
		for j in range(4):
			X_tt[j].append(X[i,j,tt])
			V_tt[j].append(V[i,j,tt])
X_tt = np.array(X_tt)
V_tt = np.array(V_tt)

print('len X[:,0,-1] = {:d}'.format(len(X[:,0,tt])))
print('len X_tt[:,0] = {:d}'.format(len(X_tt[:,0])))

zoom = 2. # 2. # 0.5

for i in range(3):
	j = index_1[i]
	k = index_2[i]
	q = index_3[i]
	ax_V[i].set_title('Velocity along the ' + axis_n[j] + '-axis on the ' + axis_n[k] + axis_n[q] + '-plane' + '\n' + '$ t = {:.3f} $ Myr - Remnant R.F.\n'.format(T[tt]),fontsize=10)
	ax_V[i].grid(linestyle=':', which='both')
	ax_V[i].set_xlim(-zoom,+zoom)
	ax_V[i].set_ylim(-zoom,+zoom)	
	ax_V[i].set_aspect(1)	
	sc_V.append( ax_V[i].scatter(X_tt[k,:], X_tt[q,:], c=V_tt[j,:], s=0.05, vmin=-0.001, vmax=+0.001) )
	cbar_V.append( fig_V[i].colorbar(sc_V[i], ax=ax_V[i], ticks=[-0.001, +0.001]) )
	
	counts_neg = 0
	counts_pos = 0
	counts_v_neg = 0
	counts_v_pos = 0
	counts_tot = len(X_tt[k,:])
	kk = 1
	if j == 1:
		kk = k
	elif j == 2:
		kk = q
	for num in range(counts_tot):
		if X_tt[kk,num] < 0.:
			counts_neg += 1
			if V_tt[j,num] > 0.:
				counts_v_pos += 1
		else:
			counts_pos += 1
			if V_tt[j,num] < 0.:
				counts_v_neg += 1
	pos = counts_v_pos / counts_neg * 100.
	neg = counts_v_neg / counts_pos * 100.
	
	if j == 1:
		string_label = '{:.2f} $\%$ particles with $ y < 0 $ have '.format(pos) + r'$ v_{x} > 0$' + '\n' + '{:.2f} $\%$ particles with $ y > 0 $ have '.format(neg) + r'$ v_{x} < 0$'
		ax_V[i].vlines(0, ymin=-zoom, ymax=+zoom, color='black', lw=0.75, label=string_label)
	elif j == 2:
		string_label = '{:.2f} $\%$ particles with $ x < 0 $ have '.format(pos) + r'$ v_{y} > 0$' + '\n' + '{:.2f} $\%$ particles with $ x > 0 $ have '.format(neg) + r'$ v_{y} < 0$'
		ax_V[i].hlines(0, xmin=-zoom, xmax=+zoom, color='black', lw=0.75, label=string_label)
	
	ax_V[i].legend(frameon=True,loc=1)
	cbar_V[i].ax.set_yticklabels(['$ < 0 $', '$ > 0 $']) 
	ax_V[i].set_xlabel(axis_n[k]  + '[pc]')
	ax_V[i].set_ylabel(axis_n[q]  + '[pc]')
	cbar_V[i].set_label(cbar_n[j] + '[km/s]')

for i in range(3):
	fig_V[i].savefig("C2_Results_PNG/Velocity_Small_{:}_cmap_CM_{:}.png".format(i+1, plotfile), bbox_inches='tight', dpi=400)
	fig_V[i].savefig("C2_Results_EPS/Velocity_Small_{:}_cmap_CM_{:}.eps".format(i+1, plotfile), bbox_inches='tight')


fig_VX , ax_VX = plt.subplots(figsize=(5.5,5.5))
fig_VY , ax_VY = plt.subplots(figsize=(5.5,5.5))
fig_VZ , ax_VZ = plt.subplots(figsize=(5.5,5.5))
fig_V  = [fig_VX , fig_VY, fig_VZ]
ax_V   = [ax_VX, ax_VY, ax_VZ]
sc_V   = []
cbar_V = []

zoom = 2. # 2. # 0.5

for i in range(3):
	j = index_1[i]
	k = index_2[i]
	q = index_3[i]
	ax_V[i].set_title('Velocity along the ' + axis_n[j] + '-axis on the ' + axis_n[k] + axis_n[q] + '-plane' + '\n' + '$ t = {:.3f} $ Myr - Remnant R.F.\n'.format(T[tt]),fontsize=10)
	ax_V[i].grid(linestyle=':', which='both')
	ax_V[i].set_xlim(-zoom,+zoom)
	ax_V[i].set_ylim(-zoom,+zoom)	
	ax_V[i].set_aspect(1)	
	sc_V.append( ax_V[i].scatter(X_tt[k,:], X_tt[q,:], c=V_tt[j,:], s=0.05, vmin=-15, vmax=+15) )
	cbar_V.append( fig_V[i].colorbar(sc_V[i], ax = ax_V[i]) )
	ax_V[i].set_xlabel(axis_n[k]  + '[pc]')
	ax_V[i].set_ylabel(axis_n[q]  + '[pc]')
	cbar_V[i].set_label(cbar_n[j] + '[km/s]')

for i in range(3):
	fig_V[i].savefig("C2_Results_PNG/Velocity_{:}_cmap_CM_{:}.png".format(i+1, plotfile), bbox_inches='tight', dpi=400)
	fig_V[i].savefig("C2_Results_EPS/Velocity_{:}_cmap_CM_{:}.eps".format(i+1, plotfile), bbox_inches='tight')


print()
print('Fig saved: Velcity color maps, RF CM')
print()


##################################################################################################
##################################################################################################

# PLOT REMNANT DENSITY PROFILE - REMNANT C.M. R.F.

fig_D , ax_D = plt.subplots(figsize=(5.5,5.5))

ax_D.set_title('Remnant density profile - Remnant R.F.\n',fontsize=10)
ax_D.grid(linestyle=':', which='both')
ax_D.set_xlabel(r'$r\;$[pc]')
ax_D.set_ylabel(r'$\rho$\;[M$_{\odot}$ \,pc$^{-3}]$')

ax_D.set_xscale('log')
ax_D.set_yscale('log')

R_remn = np.copy(X[:,0,tt])
R_remn = R_remn[R_remn < r_75_end]

tt = -1
m_i = M[0,tt]
r_min_tt = np.amin( R_remn )
r_max_tt = np.amax( R_remn )
oom_r_min_tt = np.log10( r_min_tt )
oom_r_max_tt = np.log10( r_max_tt )

# create log-spaced radial coordinate bins to create histogram 
R_log = np.logspace(oom_r_min_tt, oom_r_max_tt, 51)
R_log_c = np.logspace(oom_r_min_tt, oom_r_max_tt, 101)
R_log_c = R_log_c[1:100:2]

# Create volume bins (shells) to normalise the counts and obtain local densities
V_log = np.zeros(len(R_log))
for i in range(len(R_log)):
	V_log[i] = 4. * np.pi / 3. * R_log[i]**3
for i in range(len(R_log)-1):
	V_log[-1-i] = V_log[-1-i] - V_log[-2-i]

# create histogram and divide the counts by the volumes of the shells
histo_D , trash_D = np.histogram( X[:,0,tt] , bins=R_log )
D_log = []
for i in range(len(V_log)-1):
	D_log.append( histo_D[i] * m_i / V_log[i+1] )
D_log = np.array(D_log)

ax_D.plot(R_log_c, D_log, color='black', ls='', marker='o', markersize=1)

print()
print('Fig saved: Density profile, RF CM')
print()

##################################################################################################

# TRY TO FIT THE DENSITY PROFILE WITH A PLUMMER DENSITY PROFILE 

def rho_plummer(r,M,A):
	return M / ( 4. * np.pi / 3. * A**3 ) * (1 + r**2 / A**2 )**(- 5. / 2.)

def rho_plummer_b(r,A):
	return 7300. / ( 4. * np.pi / 3. * A**3 ) * (1 + r**2 / A**2 )**(- 5. / 2.)

cut = np.all([R_log_c >= 2e-2, R_log_c <= 7e0], axis=0)		# cut = np.all([R_log_c >= 1e-1, R_log_c <= 4e0], axis=0)

ax_D.set_xlim(2e-2,7e0)

popt, pcov = curve_fit(rho_plummer_b, R_log_c[cut], D_log[cut], p0=0.2, bounds=(0.,1.))
perr = np.sqrt(np.diag(pcov))
aa = popt[0]
ea = perr[0]
ax_D.plot(R_log_c[cut], rho_plummer_b(R_log_c[cut],aa), color='lightcoral', lw=0.85, label='Plummer Fit:\n'+r'$a=$\,'+'({:.4f} '.format(aa)+r'$\pm$'+' {:.4f}) pc'.format(ea)+'\n'+r'$M=$\,'+r'$73\% \,M_0 = 7.3 \cdot 10^3 \, M_{\odot}$')

ax_D.legend(frameon=True, loc=3)
fig_D.tight_layout()

fig_D.savefig("C2_Results_PNG/Density_Profile_Fit_{:}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_D.savefig("C2_Results_EPS/Density_Profile_Fit_{:}.eps".format(plotfile), bbox_inches='tight')

print()
print('Fig saved: Density profile, RF CM')
print()

##################################################################################################
##################################################################################################

# PLOT COORDINATE-SPACE HISTOGRAMS TO CHECK IF THE REMNANT HAS A SPHERICAL SYMMETRY

tt = -1

r_step = (r_max_tt - r_min_tt) / 40.
ph_lim  = 2. * np.pi
ph_step = np.pi / 12.
th_lim  = np.pi
th_step = np.pi / 12.

R  = np.zeros(I)
Th = np.zeros(I)
Ph = np.zeros(I)
for i in range(I):
	R[i]  = X[i,0,tt]
	Ph[i] = np.arctan2( X[i,2,tt] , X[i,1,tt])
	if Ph[i] < 0.:
		Ph[i] += 2. * np.pi
	Th[i] = np.arccos( X[i,3,tt] / X[i,0,tt])

fig_h = plt.figure(figsize=(7,6), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig_h)
ax_h_R  = fig_h.add_subplot(gs[0,0:3])
ax_h_Ph = fig_h.add_subplot(gs[1,0:2])
ax_h_Th = fig_h.add_subplot(gs[1,2:3])


R_bins = np.logspace(oom_r_min_tt, oom_r_max_tt, 41)
ax_h_R.hist(R[R < r_75_end], bins=R_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter_log(ax=ax_h_R, x_min=r_min_tt, x_max=r_max_tt, x_bins=R_bins, func=pdf_plum, npar=1)
ax_h_R.set_xscale('log')

# R_bins = np.linspace(r_min_tt, r_max_tt, 41)
# ax_h_R.hist(R, bins=R_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
# histo_pdf_plotter(ax=ax_h_R, x_min=r_min_tt, x_lim=r_max_tt, x_step=r_step, x_bins=R_bins, func=pdf_hern_r, npar=1)

ax_h_R.set_title('Position-space'+'\n'+'Radius pdf',fontsize=10)
ax_h_R.set_xlabel(r'$r\;$[pc]')
ax_h_R.set_xlim(r_min_tt, r_max_tt)
ax_h_R.set_ylim(None,None)
ax_h_R.grid(ls=':',which='both')

Th_bins = np.linspace(start=0, stop=th_lim+0.1*th_step, num=12)
ax_h_Th.hist(Th[R < r_75_end], bins=Th_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_h_Th, x_lim=th_lim, x_step=th_step, x_bins=Th_bins, func=pdf_th, npar=0)
ax_h_Th.set_title('Position-space'+'\n'+'Polar angle pdf',fontsize=10)
ax_h_Th.set_xlabel(r'$\vartheta \;$[rad]')
ax_h_Th.set_xlim(0,th_lim)
ax_h_Th.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 4))
ax_h_Th.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
ax_h_Th.grid(ls=':',which='both')

Ph_bins = np.linspace(start=0,stop=ph_lim+0.1*ph_step, num=24)
ax_h_Ph.hist(Ph[R < r_75_end], bins=Ph_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_h_Ph, x_lim=ph_lim, x_step=ph_step, x_bins=Ph_bins, func=pdf_ph, npar=0)
ax_h_Ph.set_title('Position-space'+'\n'+'Azimuthal angle pdf',fontsize=10)
ax_h_Ph.set_xlabel(r'$\varphi \;$[rad]')
ax_h_Ph.set_xlim(0,ph_lim)
ax_h_Ph.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax_h_Ph.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
ax_h_Ph.grid(ls=':',which='both')

fig_h.savefig("C2_Results_PNG/Histograms_t_end_CM_{:}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_h.savefig("C2_Results_EPS/Histograms_t_end_CM_{:}.eps".format(plotfile), bbox_inches='tight')
print()
print('Fig saved: Histograms, RF CM')
print()

##################################################################################################
##################################################################################################

# PLOT VELOCITY-SPACE HISTOGRAMS TO CHECK WHETHER THE REMNANT IS ISOTROPIC

tt = -1
R_remn = np.copy(X[:,0,tt])
V_remn = np.copy(V[:,0,tt])
V_remn = V_remn[R_remn < r_75_end]

m_i = M[0,tt]
v_min_tt = np.amin( V_remn )
v_max_tt = np.amax( V_remn )
oom_v_min_tt = np.log10( v_min_tt )
oom_v_max_tt = np.log10( v_max_tt )

v_step = (v_max_tt - v_min_tt) / 40.
ph_lim  = 2. * np.pi
ph_step = np.pi / 12.
th_lim  = np.pi
th_step = np.pi / 12.

R  = np.zeros(I)
P  = np.zeros(I)
Th = np.zeros(I)
Ph = np.zeros(I)
for i in range(I):
	R[i]  = X[i,0,tt]
	P[i]  = V[i,0,tt]
	Ph[i] = np.arctan2( V[i,2,tt] , V[i,1,tt])
	if Ph[i] < 0.:
		Ph[i] += 2. * np.pi
	Th[i] = np.arccos( V[i,3,tt] / V[i,0,tt])

fig_v = plt.figure(figsize=(7,6), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig_h)
ax_v_V  = fig_v.add_subplot(gs[0,0:3])
ax_v_Ph = fig_v.add_subplot(gs[1,0:2])
ax_v_Th = fig_v.add_subplot(gs[1,2:3])


V_bins = np.logspace(oom_r_min_tt, oom_r_max_tt, 41)
ax_v_V.hist(P[R < r_75_end], bins=R_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
# histo_pdf_plotter_log(ax=ax_h_R, x_min=r_min_tt, x_max=r_max_tt, x_bins=R_bins, func=pdf_hern_r, npar=1)
ax_v_V.set_xscale('log')

# R_bins = np.linspace(r_min_tt, r_max_tt, 41)
# ax_h_R.hist(R, bins=R_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
# histo_pdf_plotter(ax=ax_h_R, x_min=r_min_tt, x_lim=r_max_tt, x_step=r_step, x_bins=R_bins, func=pdf_hern_r, npar=1)

ax_v_V.set_title('Velocity-space'+'\n'+'Velocity module pdf',fontsize=10)
ax_v_V.set_xlabel(r'$v\;$[km/s]')
ax_v_V.set_xlim(v_min_tt, v_max_tt)
ax_v_V.set_ylim(None,None)
ax_v_V.grid(ls=':',which='both')

Th_bins = np.linspace(start=0, stop=th_lim+0.1*th_step, num=12)
ax_v_Th.hist(Th[R < r_75_end], bins=Th_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_v_Th, x_lim=th_lim, x_step=th_step, x_bins=Th_bins, func=pdf_th, npar=0)
ax_v_Th.set_title('Velocity-space'+'\n'+'Polar angle pdf',fontsize=10)
ax_v_Th.set_xlabel(r'$\vartheta \;$[rad]')
ax_v_Th.set_xlim(0,th_lim)
ax_v_Th.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 4))
ax_v_Th.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
ax_v_Th.grid(ls=':',which='both')

Ph_bins = np.linspace(start=0,stop=ph_lim+0.1*ph_step, num=24)
ax_v_Ph.hist(Ph[R < r_75_end], bins=Ph_bins, color='lightgrey', alpha=1, edgecolor='black', density=True)
histo_pdf_plotter(ax=ax_v_Ph, x_lim=ph_lim, x_step=ph_step, x_bins=Ph_bins, func=pdf_ph, npar=0)
ax_v_Ph.set_title('Velocity-space'+'\n'+'Azimuthal angle pdf',fontsize=10)
ax_v_Ph.set_xlabel(r'$\varphi \;$[rad]')
ax_v_Ph.set_xlim(0,ph_lim)
ax_v_Ph.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
ax_v_Ph.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.2f}$\,\pi$'.format(val/np.pi) if val !=0 else '0'))
ax_v_Ph.grid(ls=':',which='both')

fig_v.tight_layout()
fig_v.savefig("C2_Results_PNG/Histograms_V_t_end_CM_{:}.png".format(plotfile), bbox_inches='tight', dpi=400)
fig_v.savefig("C2_Results_EPS/Histograms_V_t_end_CM_{:}.eps".format(plotfile), bbox_inches='tight')

time_prog_end = timer()

print("Total running time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_end - time_prog_start)))
print()

# plt.show()