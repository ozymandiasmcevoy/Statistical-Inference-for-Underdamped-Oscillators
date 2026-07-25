# This script implements a complete reproducible simulation and parameter-estimation procedure for an underdamped mass-spring oscillator  (Ozymandias McEvoy)
# System choice: mechanically isomorphic to a series RLC circuit. 

# Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# Initialize reproducible random-number generator
rng = np.random.default_rng(2026)

# Declare random true physical parameters
mass_true = rng.uniform(1.0,10.0)
spring_constant_true = rng.uniform(1.0,10.0)
# Draw damping as a fraction of the critical-damping threshold
critical_damping = 2.0*np.sqrt(mass_true*spring_constant_true)
damping_coefficient_true = rng.uniform(0.05,0.90)*critical_damping
# Declare random true initial conditions
initial_position_true = rng.uniform(-1.0,1.0)
initial_velocity_true = rng.uniform(-1.0,1.0)

# Verify the system is underdamped: c < 2 * sqrt(m * k) otherwise solution regime changes
if damping_coefficient_true >= 2.0 * np.sqrt(mass_true * spring_constant_true):
	raise ValueError("The selected parameters are not underdamped.")

# Calculate derived physical parameters
gamma_true = damping_coefficient_true / (2.0 * mass_true)
omega0_true = np.sqrt(spring_constant_true / mass_true)
omega_d_true = np.sqrt(omega0_true**2 - gamma_true**2)

# Declare time points
time = np.linspace(0.0, 20.0, 2000)
N = len(time)

# Calculate the true analytical position
position_true = np.exp(-gamma_true*time)*(initial_position_true*np.cos(omega_d_true*time)+(initial_velocity_true+gamma_true*initial_position_true)/ omega_d_true*np.sin(omega_d_true*time))
# Calculate the true analytical velocity
velocity_true = np.exp(-gamma_true*time)*(initial_velocity_true*np.cos(omega_d_true*time)-(gamma_true*initial_velocity_true+omega0_true**2*initial_position_true)/omega_d_true*np.sin(omega_d_true*time))
# Calculate acceleration using the governing differential equation
acceleration_true = (-(damping_coefficient_true / mass_true) * velocity_true-(spring_constant_true / mass_true) * position_true)

#------------------------------------------------------ Assuming we observe a noisy position metric ---------------------------------------------------------------#

# Declare measurement-noise level
SNR_true = 3.0
position_noise_sigma_true = np.std(position_true)/SNR_true
# Generate noisy position observations
position_observed = rng.normal(loc=position_true,scale=position_noise_sigma_true)
# Calculate "observed velocity" from observed position
velocity_observed = np.gradient(position_observed,time)

# Estimate the initial position from the first observation
initial_position_guess = position_observed[0]
# Estimate the initial velocity from the slope over the first several observations
initial_velocity_guess = (position_observed[5]-position_observed[0])/(time[5]-time[0])
# Estimate the dominant oscillation frequency using the Fourier transform
time_step = time[1]-time[0]
position_centered = position_observed-np.mean(position_observed)
fourier_magnitudes = np.abs(np.fft.rfft(position_centered))
fourier_frequencies = np.fft.rfftfreq(N,d=time_step)
# Exclude the zero-frequency component
fourier_magnitudes[0] = 0.0

# Convert the dominant ordinary frequency to angular frequency
omega_d_guess = 2.0*np.pi*fourier_frequencies[np.argmax(fourier_magnitudes)]
# Use a moderate positive damping guess
gamma_guess = 0.20*omega_d_guess
# Declare initial parameter guesses gamma, omega_d, initial position, initial velocity
initial_parameter_guesses = np.array([
	gamma_guess,
	omega_d_guess,
	initial_position_guess,
	initial_velocity_guess])

# Define function for least squares to compute residuals from given fit
def oscillator_residuals(parameters,time,position_observed):
	# Pull the current Parameters individually
	gamma,omega_d,initial_position,initial_velocity = parameters
	# Compute the implied position
	position_estimated = np.exp(-gamma*time)*(initial_position*np.cos(omega_d*time)+(initial_velocity+gamma*initial_position)/omega_d*np.sin(omega_d*time))
	# Compute the residual error from the fit
	residuals = position_observed-position_estimated
	return residuals

#Establish the search bounds for each parameter gamma, omaga_d,init_pos,init_vel
upperbounds = [np.inf,np.inf,np.inf,np.inf]
lowerbounds =[.00000001,0.00000001,-np.inf,-np.inf]
Bounds = (lowerbounds,upperbounds)
# Estimate gamma, omega_d, x0, and v0
optimization_result = least_squares(oscillator_residuals, initial_parameter_guesses, args=(time,position_observed), bounds=Bounds)

# Extract parameter estimates
gamma_estimate = optimization_result.x[0]
omega_d_estimate = optimization_result.x[1]
initial_position_estimate = optimization_result.x[2]
initial_velocity_estimate = optimization_result.x[3]

# Calculate implied position trajectory
position_estimated = np.exp(-gamma_estimate*time)*(initial_position_estimate*np.cos(omega_d_estimate*time)+(initial_velocity_estimate+gamma_estimate*initial_position_estimate)/omega_d_estimate*np.sin(omega_d_estimate*time))
# Calculate fitted residuals
position_residuals = position_observed-position_estimated

# Estimate noise variance from conditional MLE
position_noise_variance_estimate = np.sum(position_residuals**2)/N
position_noise_sigma_estimate = np.sqrt(position_noise_variance_estimate)

# Recover physical parameter estimates
damping_coefficient_estimate = 2.0*mass_true*gamma_estimate
omega0_estimate = np.sqrt(omega_d_estimate**2+gamma_estimate**2)
spring_constant_estimate = mass_true*omega0_estimate**2


# Recover implied Velocity and Acceleration Trajectories
velocity_estimated = np.exp(-gamma_estimate*time)*(initial_velocity_estimate*np.cos(omega_d_estimate*time)-(gamma_estimate*initial_velocity_estimate+omega0_estimate**2*initial_position_estimate)/omega_d_estimate*np.sin(omega_d_estimate*time))
acceleration_estimated = -2.0*gamma_estimate*velocity_estimated-omega0_estimate**2*position_estimated

# Create estimate-versus-truth comparison matrix
parameter_comparison = np.array([[gamma_true,gamma_estimate],[omega_d_true,omega_d_estimate],[initial_position_true,initial_position_estimate],[initial_velocity_true,initial_velocity_estimate],[position_noise_sigma_true,position_noise_sigma_estimate]])

# Create bar-plot comparison of the true and estimated parameters
parameter_labels = [r"$\gamma$",r"$\omega_d$",r"$x_0$",r"$v_0$",r"$\sigma$"]
figure,axes = plt.subplots(1,5,figsize=(18,5))
for parameter_index in range(len(parameter_labels)):
	bars = axes[parameter_index].bar(["True","Estimate"],parameter_comparison[parameter_index,:],color=["#DC143C","#39FF14"],edgecolor="#FFCC00")
	axes[parameter_index].set_title(parameter_labels[parameter_index],color="#FFCC00",fontsize=14)
	axes[parameter_index].set_facecolor("#003366")
	axes[parameter_index].tick_params(axis="x",colors="#FFCC00")
	axes[parameter_index].tick_params(axis="y",colors="#FFCC00")
	axes[parameter_index].grid(True,axis="y",alpha=0.35)
	axes[parameter_index].set_axisbelow(True)
	axes[parameter_index].margins(y=0.15)
	for spine in axes[parameter_index].spines.values():
		spine.set_color("#FFCC00")
	axes[parameter_index].bar_label(bars,fmt="%.4f",padding=4,color="#FFCC00",fontsize=10)
figure.patch.set_facecolor("#003366")
figure.suptitle("Comparison of True and Estimated Oscillator Parameters",color="#FFCC00",fontsize=18)
plt.tight_layout(rect=[0.0,0.0,1.0,0.90])

# Plot the Estimated, True, and Observed Position
figure,axis = plt.subplots(figsize=(12,6))
axis.scatter(time,position_observed,s=5,alpha=0.20,label="Observed Position",color="#000080")
axis.plot(time,position_true,linestyle=":",linewidth=3,alpha=0.65,label="True Position",color="#800000")
axis.plot(time,position_estimated,linestyle=":",linewidth=2,label="Estimated Position",color="#228B22")
axis.set_xlabel("Time (s)")
axis.set_ylabel("Position (m)")
axis.set_title("Observed, True, and Estimated Position")
axis.legend()
axis.grid(True,alpha=0.35)

# Plot the Estimated, True, and "Observed Velocity"(derivative of observed position)
figure,axis = plt.subplots(figsize=(12,6))
axis.scatter(time,velocity_observed,s=5,alpha=0.20,label="Observed Velocity",color="#000080")
axis.plot(time,velocity_true,linestyle=":",linewidth=3,alpha=0.65,label="True Velocity",color="#800000")
axis.plot(time,velocity_estimated,linestyle=":",linewidth=2,label="Estimated Velocity",color="#228B22")
axis.set_xlabel("Time (s)")
axis.set_ylabel("Velocity (m/s)")
axis.set_title("Derived Observed, True, and Estimated Velocity")
axis.legend()
axis.grid(True,alpha=0.35)

plt.show()