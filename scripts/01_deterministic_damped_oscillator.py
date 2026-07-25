# This script implements a complete deterministic simulation of an underdamped mass-spring oscillator.
# System choice: mechanically isomorphic to a series RLC circuit. (Ozymandias McEvoy)

# Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt

# Declare true physical parameters
mass_true = 1.0                 # kg
spring_constant_true = 4.0      # N / m
damping_coefficient_true = 0.6  # N s / m

# Declare true initial conditions
initial_position_true = 0.10    # m, relative to static equilibrium
initial_velocity_true = 0.00    # m / s

# Verify the system is underdamped: c < 2 * sqrt(m * k) otherwise solution regime changes
if damping_coefficient_true >= 2.0 * np.sqrt(mass_true * spring_constant_true):
    raise ValueError("The selected parameters are not underdamped.")

# Calculate derived physical parameters
gamma_true = damping_coefficient_true / (2.0 * mass_true)
omega0_true = np.sqrt(spring_constant_true / mass_true)
omega_d_true = np.sqrt(omega0_true**2 - gamma_true**2)

# Declare time points
time = np.linspace(0.0, 20.0, 2000)

# Calculate the true analytical position
position_true = np.exp(-gamma_true*time)*(initial_position_true*np.cos(omega_d_true*time)+(initial_velocity_true+gamma_true*initial_position_true)/ omega_d_true*np.sin(omega_d_true*time))
# Calculate the true analytical velocity
velocity_true = np.exp(-gamma_true*time)*(initial_velocity_true*np.cos(omega_d_true*time)-(gamma_true*initial_velocity_true+omega0_true**2*initial_position_true)/omega_d_true*np.sin(omega_d_true*time))
# Calculate acceleration using the governing differential equation
acceleration_true = (-(damping_coefficient_true / mass_true) * velocity_true-(spring_constant_true / mass_true) * position_true)

# Calculate the true squared analytical position
position_true_squared =position_true**2
# Calculate the true squared analytical velocity
velocity_true_squared = velocity_true**2

# The Kinetic energy over time is thus 
T = (1/2)*mass_true*velocity_true_squared
# The Potential energy over time is thus
U = (1/2)*spring_constant_true*position_true_squared


# The dynamic Energy over time is then 
E = T+U

# The theoretical rate of change of mechanical energy is determined by viscous dissipation
Energy_Derivative_Theory = -damping_coefficient_true*velocity_true_squared
# Numerical derivative of total mechanical energy
energy_derivative_numerical = np.gradient(E,time)


# Calculate the displacement-envelope amplitude
envelope_amplitude = np.sqrt(initial_position_true**2+ ((initial_velocity_true + gamma_true * initial_position_true)/ omega_d_true)**2)
position_envelope_positive = (envelope_amplitude * np.exp(-gamma_true * time))
position_envelope_negative = (-envelope_amplitude * np.exp(-gamma_true * time))
# Calculate the velocity-envelope amplitude
velocity_envelope_amplitude = np.sqrt(initial_velocity_true**2+((gamma_true*initial_velocity_true+omega0_true**2*initial_position_true)/omega_d_true)**2)
velocity_envelope_positive = velocity_envelope_amplitude*np.exp(-gamma_true*time)
velocity_envelope_negative = -velocity_envelope_amplitude*np.exp(-gamma_true*time)

# Validate the numerical energy derivative away from the boundaries
energy_derivative_error = np.max(np.abs(energy_derivative_numerical[1:-1] - Energy_Derivative_Theory[1:-1]))

# Validate Newton's second law
force_due_to_damping_true = (-damping_coefficient_true * velocity_true)
force_due_to_spring_true = (-spring_constant_true * position_true)
net_force_true = (force_due_to_damping_true + force_due_to_spring_true)
newton_law_error = np.max( np.abs(mass_true * acceleration_true- net_force_true))

print(f"Natural angular frequency: {omega0_true:.6f} rad/s")
print(f"Damped angular frequency:  {omega_d_true:.6f} rad/s")
print(f"Decay rate:                {gamma_true:.6f} 1/s")
print(f"Maximum Newton-law error:  {newton_law_error:.3e} N")
print(f"Maximum energy-rate error: {energy_derivative_error:.3e} W")


# Figure 1: Position and exponential decay envelope
plt.figure(figsize=(10, 6))
plt.plot(time,position_true,label="Position")
plt.plot(time,position_envelope_positive,linestyle="--",label="Positive decay envelope")
plt.plot(time,position_envelope_negative,linestyle="--",label="Negative decay envelope")
plt.xlabel("Time (s)")
plt.ylabel("Position relative to equilibrium (m)")
plt.title("Underdamped Oscillator Position")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Figure 2: Velocity and exponential decay envelope
plt.figure(figsize=(10,6))
plt.plot(time,velocity_true,label="Velocity")
plt.plot(time,velocity_envelope_positive,linestyle="--",label="Positive decay envelope")
plt.plot(time,velocity_envelope_negative,linestyle="--",label="Negative decay envelope")
plt.axhline(0.0,linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Underdamped Oscillator Velocity")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Figure 3: Phase-space trajectory
plt.figure(figsize=(8, 7))
plt.plot(position_true,velocity_true*mass_true,color="blue",label="Phase-space trajectory")
plt.scatter(position_true[0],mass_true*velocity_true[0],color="green", label="Initial state")
plt.scatter(position_true[-1],mass_true*velocity_true[-1],color="red",label="Final state")
plt.xlabel("Position (m)")
plt.ylabel("Momentum (kg*m/s)")
plt.title("Damped Oscillator Phase Space")
plt.grid(True)
plt.legend()
plt.tight_layout()



# Figure 4: Mechanical energies
plt.figure(figsize=(10, 6))
plt.plot(time,T,label="Kinetic energy")
plt.plot(time,U,label="Potential energy",)
plt.plot(time,E,linewidth=2.0,label="Total mechanical energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Mechanical Energy of the Damped Oscillator")
plt.grid(True)
plt.legend()
plt.tight_layout()



# Figure 5: Energy-loss-rate validation
plt.figure(figsize=(10, 6))
plt.plot(time,Energy_Derivative_Theory,linewidth=2.0, label=r"Theoretical $-c v(t)^2$")
plt.plot(time,energy_derivative_numerical,linestyle="--",label=r"Numerical $dE/dt$")
plt.xlabel("Time (s)")
plt.ylabel("Rate of change of energy (W)")
plt.title("Energy Dissipation Validation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
