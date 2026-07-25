# This script implements a complete reproducible simulation and parameter-estimation procedure for an underdamped mass-spring oscillator  (Ozymandias McEvoy)
# System choice: mechanically isomorphic to a series RLC circuit.. 


# Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm,chi2


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

#------------------------------------------------------ Assuming we observe many noisy position metrics ---------------------------------------------------------------#

# Declare Number of Noisy Trajectories to Simulate 
L = 1000

# Declare measurement-noise level
SNR_true = 3.0
position_noise_sigma_true = np.std(position_true)/SNR_true

# Generate L independent noisy position time series
position_observed = rng.normal(loc=position_true,scale=position_noise_sigma_true,size=(L,N))
# Calculate "observed velocity" from observed position
velocity_observed = np.gradient(position_observed,time,axis=1)

# Initialize arrays for parameter estimates
gamma_estimates = np.zeros(L)
omega_d_estimates = np.zeros(L)
initial_position_estimates = np.zeros(L)
initial_velocity_estimates = np.zeros(L)
position_noise_sigma_estimates = np.zeros(L)
optimization_success = np.zeros(L,dtype=bool)

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


# Loop over each of the L trajectories and fit
for simulation_index in range(L):

	# Extract the current noisy position time series
	position_observed_current = position_observed[simulation_index,:]


	# Estimate the initial position from the first observation of the given trajectory
	initial_position_guess = position_observed_current[0]
	# Estimate the initial velocity from the slope over the first several observations
	initial_velocity_guess = (position_observed_current[5]-position_observed_current[0])/(time[5]-time[0])
	# Estimate the dominant oscillation frequency using the Fourier transform
	time_step = time[1]-time[0]
	position_centered = position_observed_current-np.mean(position_observed_current)
	fourier_magnitudes = np.abs(np.fft.rfft(position_centered))
	fourier_frequencies = np.fft.rfftfreq(N,d=time_step)
	# Exclude the zero-frequency component
	fourier_magnitudes[0] = 0.0

	# Convert the dominant ordinary frequency to angular frequency
	omega_d_guess = 2.0*np.pi*fourier_frequencies[np.argmax(fourier_magnitudes)]
	# Use a moderate positive damping guess
	gamma_guess = 0.20*omega_d_guess
	# Declare initial parameter guesses gamma, omega_d, initial position, initial velocity
	initial_parameter_guesses = np.array([gamma_guess,omega_d_guess,initial_position_guess,initial_velocity_guess])

	# Estimate gamma, omega_d, x0, and v0
	optimization_result = least_squares(oscillator_residuals, initial_parameter_guesses, args=(time,position_observed_current), bounds=Bounds)
	optimization_success[simulation_index] = optimization_result.success


	# Pull the Parameter Estimates
	gamma_estimate            = optimization_result.x[0]
	omega_d_estimate          = optimization_result.x[1]
	initial_position_estimate = optimization_result.x[2]
	initial_velocity_estimate = optimization_result.x[3]

	# Store the Parameter Estimates for the current trajectory
	gamma_estimates[simulation_index] 			 = gamma_estimate		 
	omega_d_estimates[simulation_index] 		 = omega_d_estimate	 
	initial_position_estimates[simulation_index] = initial_position_estimate
	initial_velocity_estimates[simulation_index] = initial_velocity_estimate

	# Calculate implied position trajectory
	position_estimated = np.exp(-gamma_estimate*time)*(initial_position_estimate*np.cos(omega_d_estimate*time)+(initial_velocity_estimate+gamma_estimate*initial_position_estimate)/omega_d_estimate*np.sin(omega_d_estimate*time))
	# Calculate fitted residuals
	position_residuals = position_observed_current-position_estimated
	# Estimate noise variance from conditional MLE
	position_noise_variance_estimate = np.sum(position_residuals**2)/N
	position_noise_sigma_estimate    = np.sqrt(position_noise_variance_estimate)
	# Store the conditional variance mle
	position_noise_sigma_estimates[simulation_index] = position_noise_sigma_estimate

print(f"Optimization success rate: {100.0*np.mean(optimization_success):.2f}%")


# Concattenate Parameter Labels
parameter_labels = [r"$\hat{\gamma}$",r"$\hat{\omega}_d$",r"$\hat{x}_0$",r"$\hat{v}_0$",r"$\hat{\sigma}$"]
# Concattenate the true params
parameter_truth = np.array([gamma_true,omega_d_true,initial_position_true,initial_velocity_true,position_noise_sigma_true])
# Concattenate the estimated params
parameter_estimates = np.column_stack((gamma_estimates,omega_d_estimates,initial_position_estimates,initial_velocity_estimates,position_noise_sigma_estimates))

# Compute the means, biases, stdevs, and RMSEs
parameter_means = np.mean(parameter_estimates,axis=0)
parameter_biases = parameter_means-parameter_truth
parameter_standard_deviations = np.std(parameter_estimates,axis=0,ddof=1)
parameter_RMSEs = np.sqrt(np.mean((parameter_estimates-parameter_truth)**2,axis=0))

# Least-squares estimates should be approximately normal for large N
distribution_grids = []
distribution_densities = []
distribution_labels = []

for parameter_index in range(4):
	distribution_grid = np.linspace(np.min(parameter_estimates[:,parameter_index]),np.max(parameter_estimates[:,parameter_index]),500)
	distribution_density = norm.pdf(distribution_grid,loc=parameter_means[parameter_index],scale=parameter_standard_deviations[parameter_index])
	distribution_grids.append(distribution_grid)
	distribution_densities.append(distribution_density)
	distribution_labels.append("Fitted normal density")

# Construct the transformed scaled-chi-square density for sigma_hat
degrees_of_freedom = N-4
sigma_distribution_grid = np.linspace(np.min(position_noise_sigma_estimates),np.max(position_noise_sigma_estimates),500)
chi_square_argument = N*sigma_distribution_grid**2/position_noise_sigma_true**2
sigma_distribution_density = chi2.pdf(chi_square_argument,df=degrees_of_freedom)*(2.0*N*sigma_distribution_grid/position_noise_sigma_true**2)

distribution_grids.append(sigma_distribution_grid)
distribution_densities.append(sigma_distribution_density)
distribution_labels.append(r"Transformed $\chi^2$ density")

# Plot the Histograms, with a line at their means, of each parameter with an additional line indicating the true values
figure,axes = plt.subplots(1,5,figsize=(20,5))
for parameter_index in range(5):
	axes[parameter_index].hist(parameter_estimates[:,parameter_index],bins=30,density=True,alpha=0.75,color="#0047AB")
	axes[parameter_index].plot(distribution_grids[parameter_index],distribution_densities[parameter_index],linewidth=2,color="#BF00FF",label=distribution_labels[parameter_index])
	axes[parameter_index].axvline(parameter_truth[parameter_index],linestyle="-",linewidth=2,label="Truth",color="#800000")
	axes[parameter_index].axvline(parameter_means[parameter_index],linestyle=":",linewidth=3,label="Monte Carlo mean",color="#228B22")
	axes[parameter_index].set_title(f"{parameter_labels[parameter_index]}\nBias = {parameter_biases[parameter_index]:.4f}, RMSE = {parameter_RMSEs[parameter_index]:.4f}")
	axes[parameter_index].grid(True,alpha=0.35)
# Placed perfectly above the top border, centered
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.2))
axes[4].legend(loc="lower center", bbox_to_anchor=(0.5, 1.2))
figure.suptitle("Monte Carlo Sampling Distributions of Oscillator Parameter Estimates",fontsize=16)
plt.tight_layout(rect=[0.0,0.0,1.0,0.90])

plt.show()
