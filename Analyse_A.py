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
    "font.family": "serif",				# or sans-serif
    "font.serif" : ["Times New Roman"],	# or Helvetica
	"font.size"  : 10})	

#######################################################################################################
#######################################################################################################

if len(sys.argv) > 1:
	sys.exit('ARGV ERROR, TRY:   python   Analyse_A.py' + '\n')

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

#######################################################################################################
#######################################################################################################

time_prog_start = timer()

print()

plotfile = 'OUT_OCT_Exam_C1_10000.txt'
# plotfile = sys.argv[1]

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

R = []
D = []


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

T_collapse = np.sqrt( 3. * np.pi / 32. / rho_iu ) * T0
t_collapse_m = np.argmax(T > T_collapse)
t_collapse_M = np.argmin(T < T_collapse)
t_collapse = 0.
if np.abs(T[t_collapse_m] - T_collapse) < np.abs(T[t_collapse_M] - T_collapse):
	t_collapse = t_collapse_m
else:
	t_collapse = t_collapse_M


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

# Let us fill the 0-th component (at any t) of each particle's position and velocity with their moduli

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

# find some useful limits for the plots
t_max = np.max(T)
r_max = 0.
v_max = 0.
for i in range(I):
	for t in range(NT):
		r_max = np.amax(np.array([ np.amax( X[i,0,t] ) , r_max ]))
		v_max = np.amax(np.array([ np.amax( V[i,0,t] ) , v_max ]))
r_max = 1.1 * r_max
v_max = 1.1 * v_max

##################################################################################################
##################################################################################################

fig_l , ax_l = plt.subplots(figsize=(5.5,5.5))

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

ax_l.grid(linestyle=':',which='both')
ax_l.set_xlim(0, t_max)
ax_l.set_ylim(1e-1,2e1)
ax_l.set_title('Average angular momentum as a function of time\n',fontsize=10)
ax_l.set_xlabel(r'$t\;$[Myr]')
ax_l.set_ylabel(r'$l\;$[pc km/s]') # , rotation='horizontal', horizontalalignment='right'
ax_l.set_yscale('log')

ax_l.plot(T, l_tot[0,:], color='lightcoral', label=r'Case 1: $ |\,\vec{l}_{tot}\,| $')
ax_l.plot(T, l_mean, color='lightcoral' , ls='--', label=r'Case 1: $\langle\,|\, \vec{l}\,|\, \rangle $')
ax_l.vlines(T_collapse, ymin=1e-1, ymax=2e1, linestyle=':', color='black', label=r'$T_{collapse} $')


##################################################################################################
##################################################################################################

# Let us redefine the origin as that of the centre of mass at all times (only slightly different from (0,0,0))

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
		K[i,:] = 0.5 * (V[i,0,:])**2 * 1e10
		E[i,:] = P[i,:] + K[i,:]
		P_rf[:] += 0.5 * P[i,:]
		K_rf[:] += K[i,:]
	E_rf[:] = P_rf[:] + K_rf[:]

R_35_t = np.zeros(NT)
R_73_t = np.zeros(NT)

for t in range(NT):
	C = np.copy(X[:,0,t])
	C = np.sort(C)
	R_35_t[t] = C[int(np.ceil(I/10*3.5)-1)]
	R_73_t[t] = C[int(np.ceil(I/10*7.3)-1)]

t_test    = int(2.3 * np.argmin(R_35_t[:]))
r_35_test = np.amin(R_35_t[:])
r_35_end  = R_35_t[-1]
r_73_end  = R_73_t[-1]

time_prog_CM_2 = timer()

print("Remnant CM computation time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_CM_2 - time_prog_CM_1)))
print()

##################################################################################################
##################################################################################################

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

		# to check again that the ang mom is conserved and plot it, add X_cm, V_cm to X_i , V_i 
		l_tot[1,t] += ( (X[i,2,t] + _X_cm[2,t] ) * ( V[i,3,t] + _V_cm[3,t] ) - ( X[i,3,t] + _X_cm[3,t] ) * ( V[i,2,t] + _V_cm[2,t] ) ) / I
		l_tot[2,t] += ( (X[i,3,t] + _X_cm[3,t] ) * ( V[i,1,t] + _V_cm[1,t] ) - ( X[i,1,t] + _X_cm[1,t] ) * ( V[i,3,t] + _V_cm[3,t] ) ) / I
		l_tot[3,t] += ( (X[i,1,t] + _X_cm[1,t] ) * ( V[i,2,t] + _V_cm[2,t] ) - ( X[i,2,t] + _X_cm[2,t] ) * ( V[i,1,t] + _V_cm[1,t] ) ) / I
	l_tot[0,t] = np.sqrt(l_tot[1,t]**2 + l_tot[2,t]**2 + l_tot[3,t]**2)

for t in range(NT):
	H = 0
	for i in range(I):
		if X[i,0,t] < R_73_t[t]:
			H += 1
			l_mean[t] += l[i,0,t]
	l_mean[t] = l_mean[t] / H

ax_L.grid(linestyle=':',which='both')
ax_L.set_xlim(0, t_max)
ax_L.set_ylim(0.,1.75)				# (1e-1,3e0)
ax_L.set_title('Average angular momentum as a function of time - Remnant R.F.\n',fontsize=10)
ax_L.set_xlabel(r'$t\;$[Myr]')
ax_L.set_ylabel(r'$l\;$[pc km/s]') # , rotation='horizontal', horizontalalignment='right'
# ax_L.set_yscale('log')

ax_L.vlines(T_collapse, ymin=0., ymax=2., linestyle=':', color='black', label=r'$T_{collapse} $') # ymin=1e-1, ymax=2e1
ax_L.plot(T, l_tot[0,:], color='lightcoral', label=r"Case 1: $|\,\vec{l}_{tot} \,|$") # = \frac{1}{N} | \sum_i{\vec{r}_i\times \vec{v}_i} \,|
ax_L.plot(T, l_mean, color='lightcoral' , ls='--', label=r"Case 1: $\langle \,|\,\vec{l}'\,|\, \rangle $")  # = \frac{1}{N} \sum_i{|\vec{r}_i\times \vec{v}_i\,|}


##################################################################################################
##################################################################################################

tt = -1

fig_D , ax_D = plt.subplots(figsize=(5.5,5.5))

ax_D.set_title('Remnant density profile - Remnant R.F.\n',fontsize=10)
ax_D.grid(linestyle=':', which='both')
ax_D.set_xlabel(r'$r\;$[pc]')
ax_D.set_ylabel(r'$\rho$\;[M$_{\odot}$ \,pc$^{-3}]$') # , rotation='horizontal', horizontalalignment='right'
ax_D.set_xlim(3e-2,8e0)
ax_D.set_ylim(4e-1,1e6)
# ax_D.set_aspect(a / ( - ))
ax_D.set_xscale('log')
ax_D.set_yscale('log')

R_remn = np.copy(X[:,0,tt])
R_remn = R_remn[R_remn < r_73_end]

tt = -1
m_i = M[0,tt]
r_min_tt = np.amin( R_remn ) # X[:,0,tt]
r_max_tt = np.amax( R_remn ) # X[:,0,tt]
oom_r_min_tt = np.log10( r_min_tt )
oom_r_max_tt = np.log10( r_max_tt )

R_log = np.logspace(oom_r_min_tt, oom_r_max_tt, 51)
R_log_c = np.logspace(oom_r_min_tt, oom_r_max_tt, 101)
R_log_c = R_log_c[1:100:2]

# R_sort_tt = np.sort(X[0,:,tt])
V_log = np.zeros(len(R_log))
for i in range(len(R_log)):
	V_log[i] = 4. * np.pi / 3. * R_log[i]**3
for i in range(len(R_log)-1):
	V_log[-1-i] = V_log[-1-i] - V_log[-2-i]

histo_D , trash_D = np.histogram( X[:,0,tt] , bins=R_log )
D_log = []
for i in range(len(V_log)-1):
	D_log.append( histo_D[i] * m_i / V_log[i+1] )
D_log = np.array(D_log)

for i in range(len(D_log)):
	R.append(R_log_c[i])
	D.append(D_log[i])

ax_D.plot(R_log_c, D_log, color='lightcoral', ls='', marker='o', markersize=1, label='Case 1')

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

plotfile = 'OUT_OCT_Exam_C2_10000.txt'

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

T_collapse = np.sqrt( 3. * np.pi / 32. / rho_iu ) * T0
t_collapse_m = np.argmax(T > T_collapse)
t_collapse_M = np.argmin(T < T_collapse)
t_collapse = 0.
if np.abs(T[t_collapse_m] - T_collapse) < np.abs(T[t_collapse_M] - T_collapse):
	t_collapse = t_collapse_m
else:
	t_collapse = t_collapse_M


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

# Let us fill the 0-th component (at any t) of each particle's position and velocity with their moduli

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

# find some useful limits for the plots
t_max = np.max(T)
r_max = 0.
v_max = 0.
for i in range(I):
	for t in range(NT):
		r_max = np.amax(np.array([ np.amax( X[i,0,t] ) , r_max ]))
		v_max = np.amax(np.array([ np.amax( V[i,0,t] ) , v_max ]))
r_max = 1.1 * r_max
v_max = 1.1 * v_max

##################################################################################################
##################################################################################################

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

ax_l.set_title('Average angular momentum as a function of time\n',fontsize=10)
ax_l.set_xlabel(r'$t\;$[Myr]')
ax_l.set_ylabel(r'$l\;$[pc km/s]') # , rotation='horizontal', horizontalalignment='right'
ax_l.set_yscale('log')

ax_l.plot(T, l_tot[0,:], color='lightskyblue', label=r'Case 2: $ |\,\vec{l}_{tot}\,| $')
ax_l.plot(T, l_mean, color='lightskyblue' , ls='--', label=r'Case 2: $\langle\,|\, \vec{l}\,|\, \rangle $')

##################################################################################################
##################################################################################################

# Let us redefine the origin as that of the centre of mass at all times (only slightly different from (0,0,0))

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
		K[i,:] = 0.5 * (V[i,0,:])**2 * 1e10
		E[i,:] = P[i,:] + K[i,:]
		P_rf[:] += 0.5 * P[i,:]
		K_rf[:] += K[i,:]
	E_rf[:] = P_rf[:] + K_rf[:]

R_35_t = np.zeros(NT)
R_73_t = np.zeros(NT)

for t in range(NT):
	C = np.copy(X[:,0,t])
	C = np.sort(C)
	R_35_t[t] = C[int(np.ceil(I/10*3.5)-1)]
	R_73_t[t] = C[int(np.ceil(I/10*7.3)-1)]

t_test    = int(2.3 * np.argmin(R_35_t[:]))
r_35_test = np.amin(R_35_t[:])
r_35_end  = R_35_t[-1]
r_73_end  = R_73_t[-1]

time_prog_CM_2 = timer()

print("Remnant CM computation time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_CM_2 - time_prog_CM_1)))
print()

##################################################################################################
##################################################################################################

l      = np.zeros((I,4,NT))
l_tot  = np.zeros((4,NT))
l_mean = np.zeros(NT)

for t in range(NT):
	for i in range(I):
		l[i,1,t] = X[i,2,t] * V[i,3,t] - X[i,3,t] * V[i,2,t]
		l[i,2,t] = X[i,3,t] * V[i,1,t] - X[i,1,t] * V[i,3,t]
		l[i,3,t] = X[i,1,t] * V[i,2,t] - X[i,2,t] * V[i,1,t]
		l[i,0,t] = np.sqrt(l[i,1,t]**2 + l[i,2,t]**2 + l[i,3,t]**2)

		# to check again that the ang mom is conserved and plot it, add X_cm, V_cm to X_i , V_i 
		l_tot[1,t] += ( (X[i,2,t] + _X_cm[2,t] ) * ( V[i,3,t] + _V_cm[3,t] ) - ( X[i,3,t] + _X_cm[3,t] ) * ( V[i,2,t] + _V_cm[2,t] ) ) / I
		l_tot[2,t] += ( (X[i,3,t] + _X_cm[3,t] ) * ( V[i,1,t] + _V_cm[1,t] ) - ( X[i,1,t] + _X_cm[1,t] ) * ( V[i,3,t] + _V_cm[3,t] ) ) / I
		l_tot[3,t] += ( (X[i,1,t] + _X_cm[1,t] ) * ( V[i,2,t] + _V_cm[2,t] ) - ( X[i,2,t] + _X_cm[2,t] ) * ( V[i,1,t] + _V_cm[1,t] ) ) / I
	l_tot[0,t] = np.sqrt(l_tot[1,t]**2 + l_tot[2,t]**2 + l_tot[3,t]**2)

for t in range(NT):
	H = 0
	for i in range(I):
		if X[i,0,t] < R_73_t[t]:
			H += 1
			l_mean[t] += l[i,0,t]
	l_mean[t] = l_mean[t] / H

ax_L.grid(linestyle=':',which='both')
ax_L.set_xlim(0, t_max)
ax_L.set_title('Average angular momentum as a function of time - Remnant R.F.\n',fontsize=10)
ax_L.set_xlabel(r'$t\;$[Myr]')
ax_L.set_ylabel(r'$l\;$[pc km/s]') # , rotation='horizontal', horizontalalignment='right'

ax_L.plot(T, l_tot[0,:], color='lightskyblue', label=r"Case 2: $|\,\vec{l}_{tot} \,| $") 
ax_L.plot(T, l_mean, color='lightskyblue' , ls='--', label=r"Case 2: $\langle \,|\,\vec{l}'\,|\, \rangle $")


##################################################################################################
##################################################################################################

tt = -1

ax_D.set_title('Remnant density profile - Remnant R.F.\n',fontsize=10)
ax_D.grid(linestyle=':', which='both')
ax_D.set_xlabel(r'$r\;$[pc]')
ax_D.set_ylabel(r'$\rho$\;[M$_{\odot}$ \,pc$^{-3}]$') # , rotation='horizontal', horizontalalignment='right'
ax_D.set_xscale('log')
ax_D.set_yscale('log')

R_remn = np.copy(X[:,0,tt])
R_remn = R_remn[R_remn < r_73_end]

tt = -1
m_i = M[0,tt]
r_min_tt = np.amin( R_remn ) # X[:,0,tt]
r_max_tt = np.amax( R_remn ) # X[:,0,tt]
oom_r_min_tt = np.log10( r_min_tt )
oom_r_max_tt = np.log10( r_max_tt )

R_log = np.logspace(oom_r_min_tt, oom_r_max_tt, 51)
R_log_c = np.logspace(oom_r_min_tt, oom_r_max_tt, 101)
R_log_c = R_log_c[1:100:2]

# R_sort_tt = np.sort(X[0,:,tt])
V_log = np.zeros(len(R_log))
for i in range(len(R_log)):
	V_log[i] = 4. * np.pi / 3. * R_log[i]**3
for i in range(len(R_log)-1):
	V_log[-1-i] = V_log[-1-i] - V_log[-2-i]

histo_D , trash_D = np.histogram( X[:,0,tt] , bins=R_log )
D_log = []
for i in range(len(V_log)-1):
	D_log.append( histo_D[i] * m_i / V_log[i+1] )
D_log = np.array(D_log)

for i in range(len(D_log)):
	R.append(R_log_c[i])
	D.append(D_log[i])

ax_D.plot(R_log_c, D_log, color='lightskyblue', ls='', marker='o', markersize=1, label='Case 2')

fig_D.tight_layout()

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

plotfile = 'OUT_OCT_Exam_C3_10000.txt'

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

T_collapse = np.sqrt( 3. * np.pi / 32. / rho_iu ) * T0
t_collapse_m = np.argmax(T > T_collapse)
t_collapse_M = np.argmin(T < T_collapse)
t_collapse = 0.
if np.abs(T[t_collapse_m] - T_collapse) < np.abs(T[t_collapse_M] - T_collapse):
	t_collapse = t_collapse_m
else:
	t_collapse = t_collapse_M


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

# Let us fill the 0-th component (at any t) of each particle's position and velocity with their moduli

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

# find some useful limits for the plots
t_max = np.max(T)
r_max = 0.
v_max = 0.
for i in range(I):
	for t in range(NT):
		r_max = np.amax(np.array([ np.amax( X[i,0,t] ) , r_max ]))
		v_max = np.amax(np.array([ np.amax( V[i,0,t] ) , v_max ]))
r_max = 1.1 * r_max
v_max = 1.1 * v_max

##################################################################################################
##################################################################################################

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

ax_l.plot(T, l_tot[0,:], color='khaki', label=r'Case 3: $ |\,\vec{l}_{tot}\,| $')
ax_l.plot(T, l_mean, color='khaki' , ls='--', label=r'Case 3: $\langle\,|\, \vec{l}\,|\, \rangle $')

ax_l.legend(frameon=True) #, loc=2) #, bbox_to_anchor=(1.01,1)) 
fig_l.tight_layout()

##################################################################################################
##################################################################################################

# Let us redefine the origin as that of the centre of mass at all times (only slightly different from (0,0,0))

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
		K[i,:] = 0.5 * (V[i,0,:])**2 * 1e10
		E[i,:] = P[i,:] + K[i,:]
		P_rf[:] += 0.5 * P[i,:]
		K_rf[:] += K[i,:]
	E_rf[:] = P_rf[:] + K_rf[:]

R_35_t = np.zeros(NT)
R_73_t = np.zeros(NT)

for t in range(NT):
	C = np.copy(X[:,0,t])
	C = np.sort(C)
	R_35_t[t] = C[int(np.ceil(I/10*3.5)-1)]
	R_73_t[t] = C[int(np.ceil(I/10*7.3)-1)]

t_test    = int(2.3 * np.argmin(R_35_t[:]))
r_35_test = np.amin(R_35_t[:])
r_35_end  = R_35_t[-1]
r_73_end  = R_73_t[-1]

time_prog_CM_2 = timer()

print("Remnant CM computation time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_CM_2 - time_prog_CM_1)))
print()

##################################################################################################
##################################################################################################

l      = np.zeros((I,4,NT))
l_tot  = np.zeros((4,NT))
l_mean = np.zeros(NT)

for t in range(NT):
	for i in range(I):
		l[i,1,t] = X[i,2,t] * V[i,3,t] - X[i,3,t] * V[i,2,t]
		l[i,2,t] = X[i,3,t] * V[i,1,t] - X[i,1,t] * V[i,3,t]
		l[i,3,t] = X[i,1,t] * V[i,2,t] - X[i,2,t] * V[i,1,t]
		l[i,0,t] = np.sqrt(l[i,1,t]**2 + l[i,2,t]**2 + l[i,3,t]**2)

		# to check again that the ang mom is conserved and plot it, add X_cm, V_cm to X_i , V_i 
		l_tot[1,t] += ( (X[i,2,t] + _X_cm[2,t] ) * ( V[i,3,t] + _V_cm[3,t] ) - ( X[i,3,t] + _X_cm[3,t] ) * ( V[i,2,t] + _V_cm[2,t] ) ) / I
		l_tot[2,t] += ( (X[i,3,t] + _X_cm[3,t] ) * ( V[i,1,t] + _V_cm[1,t] ) - ( X[i,1,t] + _X_cm[1,t] ) * ( V[i,3,t] + _V_cm[3,t] ) ) / I
		l_tot[3,t] += ( (X[i,1,t] + _X_cm[1,t] ) * ( V[i,2,t] + _V_cm[2,t] ) - ( X[i,2,t] + _X_cm[2,t] ) * ( V[i,1,t] + _V_cm[1,t] ) ) / I
	l_tot[0,t] = np.sqrt(l_tot[1,t]**2 + l_tot[2,t]**2 + l_tot[3,t]**2)

for t in range(NT):
	H = 0
	for i in range(I):
		if X[i,0,t] < R_73_t[t]:
			H += 1
			l_mean[t] += l[i,0,t]
	l_mean[t] = l_mean[t] / H

ax_L.grid(linestyle=':',which='both')
ax_L.set_xlim(0, t_max)
ax_L.set_title('Average angular momentum as a function of time - Remnant R.F.\n',fontsize=10)
ax_L.set_xlabel(r'$t\;$[Myr]')
ax_L.set_ylabel(r'$l\;$[pc km/s]') # , rotation='horizontal', horizontalalignment='right'

ax_L.plot(T, l_tot[0,:], color='khaki', label=r"Case 3: $|\,\vec{l}_{tot} \,| $") 
ax_L.plot(T, l_mean, color='khaki' , ls='--', label=r"Case 3: $\langle \,|\,\vec{l}'\,|\, \rangle $")  # = \frac{1}{N} \sum_i{|\vec{r}_i\times \vec{v}_i\,|}

ax_L.legend(frameon=True, loc=7) #, bbox_to_anchor=(1.01,1)) 

##################################################################################################
##################################################################################################

tt = -1

R_remn = np.copy(X[:,0,tt])
R_remn = R_remn[R_remn < r_73_end]

tt = -1
m_i = M[0,tt]
r_min_tt = np.amin( R_remn ) # X[:,0,tt]
r_max_tt = np.amax( R_remn ) # X[:,0,tt]
oom_r_min_tt = np.log10( r_min_tt )
oom_r_max_tt = np.log10( r_max_tt )

R_log = np.logspace(oom_r_min_tt, oom_r_max_tt, 51)
R_log_c = np.logspace(oom_r_min_tt, oom_r_max_tt, 101)
R_log_c = R_log_c[1:100:2]

# R_sort_tt = np.sort(X[0,:,tt])
V_log = np.zeros(len(R_log))
for i in range(len(R_log)):
	V_log[i] = 4. * np.pi / 3. * R_log[i]**3
for i in range(len(R_log)-1):
	V_log[-1-i] = V_log[-1-i] - V_log[-2-i]

histo_D , trash_D = np.histogram( X[:,0,tt] , bins=R_log )
D_log = []
for i in range(len(V_log)-1):
	D_log.append( histo_D[i] * m_i / V_log[i+1] )
D_log = np.array(D_log)

for i in range(len(D_log)):
	R.append(R_log_c[i])
	D.append(D_log[i])

ax_D.plot(R_log_c, D_log, color='khaki', ls='', marker='o', markersize=1, label='Case 3')

ax_D.legend(frameon=True, loc=1) # , loc=4, bbox_to_anchor=(1.01,1)

ax_D.grid(ls=':',which='both')
fig_D.tight_layout()

##################################################################################################
##################################################################################################


fig_d , ax_d = plt.subplots(figsize=(5.5,5.5))

ax_d.set_title('Remnant density profile - Remnant R.F.\n',fontsize=10)
ax_d.grid(linestyle=':', which='both')
ax_d.set_xlabel(r'$r\;$[pc]')
ax_d.set_ylabel(r'$\rho$\;[M$_{\odot}$ \,pc$^{-3}]$') # , rotation='horizontal', horizontalalignment='right'
ax_d.set_xlim(3e-2,8e0)
ax_d.set_ylim(4e-1,1e6)
ax_d.set_xscale('log')
ax_d.set_yscale('log')

R = np.array(R)
D = np.array(D)
sort_R = np.argsort(R)
R = R[sort_R]
D = D[sort_R]


ax_d.plot(R, D, color='black', ls='', marker='o', markersize=1)

def rho_dehnen(r,A,gamma):
	return (3-gamma)*7300./(4*np.pi) * A / ( r**gamma * (r+A)**(4-gamma) )

def rho_dehnen_b(r,A,B,C):
	return (B-1-C)*7000./(4*np.pi) * A / ( r**C * (r+A)**(B-C) )

def rho_dehnen_c(r,p,A,B,C):
	return p / ( (r/A)**C * (1+r/A)**(B-C) )

def rho_plummer(r,M,A):
	return M / ( 4. * np.pi / 3. * A**3 ) * (1 + r**2 / A**2 )**(- 5. / 2.)

def rho_plummer_b(r,A):
	return 7300. / ( 4. * np.pi / 3. * A**3 ) * (1 + r**2 / A**2 )**(- 5. / 2.)

m0 = 7300.
a0 = 0.2
g0 = 1.
b0 = 4.
p0 = m0 / (4. * np.pi / 3. * a0**3)

print("rho0_dehn = {:2e}".format(p0))

# ax_d.plot(R, rho_dehnen(R,a0,g0), color='darkred', lw=0.7, ls=':', label='Guess 1:\n'+r'$a=$\,'+'{:.3f} pc'.format(a0)+'\n'+r'$M=$\,'+'{:.0f}'.format(m0)+r'\,M$_{\odot}$'+'\n'+r'$\gamma=$\,'+'{:.2f}'.format(g0)) # , linestyle=':' 

popt1, pcov1 = curve_fit(rho_dehnen, R, D, p0=[a0,g0], bounds=([0.2,0], [5,3]))
a1 = popt1[0]
g1 = popt1[1]
# ax_d.plot(R, rho_dehnen(R,a1,g1), color='darkred', lw=0.7, label='Fit 1:\n'+r'$a=$\,'+'{:.3f} pc'.format(a1)+'\n'+r'$M=$\,'+'{:.0f}'.format(m0)+r'\,M$_{\odot}$'+'\n'+r'$\gamma=$\,'+'{:.2f}'.format(g1)) # , linestyle=':' 


popt2, pcov2 = curve_fit(rho_dehnen_c, R, D, p0=[p0,a0,b0,g0], bounds=([0.,0.1,3.,0.], [2*p0,0.5,5.,3.]))
p2 = popt2[0]
a2 = popt2[1]
b2 = popt2[2]
g2 = popt2[3]
fit_string2 = 'Fit 2:\n' + r'$\rho_{0}=$\,' + '{:.2e}'.format(p2) + '\n' + r'$a=$\,' + '{:.3f} pc'.format(a2) + '\n' + r'$\beta=$\,' + '{:.2f}'.format(b2) + '\n' + r'$\gamma=$\,' + '{:.2f}'.format(g2) 
# ax_d.plot(R, rho_dehnen_c(R,p2,a2,b2,g2), color='darkred', lw=0.75, ls='-.', label=fit_string2) # , linestyle=':' 

popt3, pcov3 = curve_fit(rho_plummer, R, D, p0=[m0,a0], bounds=([7000.,0.], [7500.,1.]))
a3 = popt3[1]
m3 = popt3[0]
# ax_d.plot(R, rho_plummer(R,m3,a3), color='lightcoral', lw=0.7, label='Fit 3:\n'+r'$a=$\,'+'{:.3f} pc'.format(a3)+'\n'+r'$M=$\,'+'{:.0f}'.format(m3))

popt4, pcov4 = curve_fit(rho_plummer_b, R, D, p0=a0, bounds=(0., 1.))
perr4 = np.sqrt(np.diag(pcov4))
a4 = popt4[0]
ea4 = perr4[0]
ax_d.plot(R, rho_plummer_b(R,a4), color='lightcoral', lw=0.75, label='Plummer Fit:\n'+r'$a=$\,'+'({:.4f} '.format(a4)+r'$\pm$'+' {:.4f}) pc'.format(ea4)+'\n'+r'$M=$\,'+r'$73\% \,M_0 = 7.3 \cdot 10^3 \, M_{\odot}$')

# ax_d.plot(R, rho_plummer_b(R,a0), color='lightcoral', ls='-.', lw=0.7, label='Guess 2:\n'+r'$a=$\,'+'{:.3f} pc'.format(a0)+'\n'+r'$M=$\,'+'{:.0f}'.format(m0))

ax_d.legend(frameon=True)

##################################################################################################
##################################################################################################

fig_l.savefig("All_Results_PNG/Angular_Momentum.png", bbox_inches='tight', dpi=400)
fig_l.savefig("All_Results_EPS/Angular_Momentum.eps", bbox_inches='tight')
fig_L.savefig("All_Results_PNG/Angular_Momentum_CMRF.png", bbox_inches='tight', dpi=400)
fig_L.savefig("All_Results_EPS/Angular_Momentum_CMRF.eps", bbox_inches='tight')
fig_D.savefig("All_Results_PNG/Density_Profile_CMRF.png", bbox_inches='tight', dpi=400)
fig_D.savefig("All_Results_EPS/Density_Profile_CMRF.eps", bbox_inches='tight')
fig_d.savefig("All_Results_PNG/Density_Profile_Fit.png", bbox_inches='tight', dpi=400)
fig_d.savefig("All_Results_EPS/Density_Profile_Fit.eps", bbox_inches='tight')

time_prog_end = timer()

print("Total running time [hh/mm/ss] = {:}".format(datetime.timedelta(seconds = time_prog_end - time_prog_start)))
print()

# plt.show()