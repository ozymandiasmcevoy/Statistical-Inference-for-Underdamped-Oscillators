# Statistical Inference for Damped Oscillators

Statistical inference and Monte Carlo analysis for an underdamped oscillator using physics-informed simulation and nonlinear least squares.

This project develops a complete forward-and-inverse modeling workflow for the classical underdamped mass–spring oscillator. It begins with analytical simulation and physical validation, then estimates unknown oscillator parameters from noisy position measurements, and finally studies the sampling behavior of those estimators through Monte Carlo simulation.

The mechanical system is mathematically analogous to a series RLC circuit.

## Project Overview

The project is organized into three stages:

1. **Forward analytical dynamics**
   Simulate the exact underdamped oscillator trajectory and validate the governing physical relationships.

2. **Single-trajectory parameter estimation**
   Add Gaussian measurement noise to the position trajectory and estimate the oscillator parameters using nonlinear least squares.

3. **Monte Carlo estimator analysis**
   Repeat the noisy estimation experiment across many independent trajectories to measure estimator bias, variance, RMSE, convergence behavior, and sampling distributions.

## Observation Model

The noisy position observations are modeled as

$$
y_t=x(t;\theta)+\varepsilon_t,
\qquad
\varepsilon_t\overset{\text{iid}}{\sim}N(0,\sigma^2),
$$

where

$$
\theta=
\begin{bmatrix}
\gamma & \omega_d & x_0 & v_0
\end{bmatrix}^{\mathsf T}.
$$

The underdamped analytical position trajectory is

$$
x(t)=e^{-\gamma t}
\left[
x_0\cos(\omega_d t)
+
\frac{v_0+\gamma x_0}{\omega_d}\sin(\omega_d t)
\right].
$$

The physical parameters are related through

$$
\gamma=\frac{c}{2m},
\qquad
\omega_0=\sqrt{\frac{k}{m}},
\qquad
\omega_d=\sqrt{\omega_0^2-\gamma^2},
$$

where

* m is the mass,
* k is the spring constant,
* c is the viscous damping coefficient,
* $\omega_0$ is the undamped natural angular frequency,
* $\omega_d$ is the damped angular frequency.

The underdamped regime requires

$$
c<2\sqrt{mk}.
$$


## 01 — Forward Analytical Dynamics

The first script implements the exact analytical solution for an underdamped mass–spring oscillator with known physical parameters and initial conditions.

It computes:

* position,
* velocity,
* acceleration,
* displacement and velocity decay envelopes,
* momentum-space trajectories,
* kinetic energy,
* potential energy,
* total mechanical energy.

The governing differential equation is

$$
m\ddot{x}+c\dot{x}+kx=0.
$$

Newton’s second law is numerically validated using

$$
m\ddot{x}=-c\dot{x}-kx.
$$

The script also verifies the theoretical mechanical-energy dissipation rate

$$
\frac{dE}{dt}=-c,v(t)^2,
$$

where

$$
E(t)=\frac{1}{2}mv(t)^2+\frac{1}{2}kx(t)^2.
$$

The numerical derivative of total mechanical energy is compared directly against the theoretical viscous-dissipation rate.

## 02 — Single-Trajectory Parameter Estimation

The second script treats the exact position trajectory as an unobserved physical signal and generates noisy position measurements at a specified signal-to-noise ratio.

The signal-to-noise ratio is defined using

$$
\sigma=\frac{\mathrm{SD}(x)}{\mathrm{SNR}}
$$

The following parameters are estimated simultaneously:

$$
\gamma,\qquad \omega_d,\qquad x_0,\qquad v_0.
$$

The estimator minimizes the residual sum of squared error $\mathrm{SSE}(\theta)$

$$
\sum_{t=1}^{N}\left[y_t-x(t;\theta)\right]^2.
$$

Because the observation errors are Gaussian, nonlinear least squares is equivalent to conditional maximum-likelihood estimation for the trajectory parameters.

### Data-Informed Initialization

Nonlinear oscillator fitting can be sensitive to arbitrary starting values, particularly for the damped frequency. The script therefore constructs initialization values from the observed trajectory:

* $x_0$ is initialized from the first observed position;
* $v_0$ is initialized using an early finite-difference slope;
* $\omega_d$ is initialized using the dominant Fourier frequency;
* $\gamma$ is initialized as a moderate positive fraction of the frequency estimate.

After estimation, the physical damping coefficient and spring constant are recovered through

$$
\hat{c}=2m\hat{\gamma},
$$

and

$$
\hat{k}=m\left(\hat{\omega}_d^2+\hat{\gamma}^2\right).
$$

The estimated analytical model is also used to reconstruct the velocity and acceleration trajectories, even though only position is supplied to the optimizer.

A numerically differentiated velocity is calculated from the noisy observed position for comparison. This derived velocity is expected to be substantially noisier because numerical differentiation amplifies measurement noise.

## 03 — Monte Carlo Estimator Performance

The third script generates (L) independent noisy position trajectories from one fixed physical system:

$$
y_{\ell,t}=x(t;\theta)+\varepsilon_{\ell,t},
\qquad
\ell=1,\ldots,L.
$$

The nonlinear least-squares procedure is repeated independently for every simulated trajectory.

The script records the Monte Carlo sampling distributions of

$$
\hat{\gamma},
\qquad
\hat{\omega}_d,
\qquad
\hat{x}_0,
\qquad
\hat{v}_0,
\qquad
\hat{\sigma}.
$$

Estimator performance is summarized using:

### Bias

$$
\mathrm{Bias}(\hat{\theta})=E[\hat{\theta}]-\theta.
$$

The LLN Monte Carlo estimate is

$$
\mathrm{\mathrm{Bias}}(\hat{\theta})=
\frac{1}{L}
\sum_{\ell=1}^{L}\hat{\theta}_{\ell}
-\theta.
$$

### Empirical Standard Deviation

$$
\widehat{\mathrm{SD}}(\hat{\theta})=\sqrt{\frac{1}{L-1}\sum_{\ell=1}^{L}\left(\hat{\theta}_{\ell}-\overline{\hat{\theta}}\right)^2}.
$$

### Root-Mean-Square Error

$$
\mathrm{RMSE}(\hat{\theta})=\sqrt{\frac{1}{L}\sum_{\ell=1}^{L}\left(\hat{\theta}_{\ell}-\theta\right)^2}.
$$

The optimization success rate is also recorded across all simulated trajectories.

## Sampling-Distribution Overlays

For large (N), the nonlinear least-squares estimates are expected to be approximately jointly normal when:

* the trajectory model is correctly specified,
* the parameters are identifiable,
* the true values are away from parameter boundaries,
* the optimizer consistently reaches the correct minimum,
* the observation errors are independent with finite variance.

The marginal distributions of

$$
\hat{\gamma},
\qquad
\hat{\omega}_d,
\qquad
\hat{x}_0,
\qquad
\hat{v}_0
$$

are therefore compared with fitted normal densities using their Monte Carlo means and empirical standard deviations.

These curves are diagnostic fitted-normal overlays rather than exact finite-sample theoretical distributions.

## Noise-Variance Distribution

The conditional maximum-likelihood estimator of the observation variance is

$$
\hat{\sigma}^2=\frac{\mathrm{SSE}}{N}.
$$

Because four trajectory parameters are estimated, the residual sum of squares approximately satisfies

$$
\frac{N\hat{\sigma}^2}{\sigma^2}\approx\chi^2_{N-4}.
$$

Equivalently,

$$
\hat{\sigma}^2\approx\frac{\sigma^2}{N}\chi^2_{N-4}.
$$

Thus, the variance estimator has an approximately scaled chi-square distribution.

The scripts store the estimated standard deviation rather than the variance. Therefore,

$$
\hat{\sigma}\approx\frac{\sigma}{\sqrt{N}}\chi_{N-4},
$$

where (\chi_{N-4}) denotes a chi distribution.

The Monte Carlo histogram of (\hat{\sigma}) is consequently compared against a transformed scaled-chi-square density.

Because the oscillator model is nonlinear, this residual-distribution result is approximate rather than exactly identical to the corresponding result from linear Gaussian regression.

## Main Outputs

The project produces visualizations of:

* analytical position and exponential decay envelopes,
* analytical velocity and velocity envelopes,
* phase-space decay,
* kinetic, potential, and total mechanical energy,
* theoretical and numerical energy-loss rates,
* noisy, true, and estimated position trajectories,
* derived observed, true, and estimated velocity trajectories,
* parameter truth-versus-estimate comparisons,
* Monte Carlo sampling distributions,
* fitted normal density overlays,
* transformed chi-distribution overlays,
* estimator bias and RMSE.

The random-number generator is seeded to make the simulations reproducible.

## Mechanical–Electrical Analogy

The damped mass–spring system

$$
m\ddot{x}+c\dot{x}+kx=0
$$

is mathematically analogous to an undriven series RLC circuit.

Under the charge-based correspondence,

$$
m\leftrightarrow L,
\qquad
c\leftrightarrow R,
\qquad
k\leftrightarrow \frac{1}{C},
\qquad
x\leftrightarrow q.
$$

The mechanical equation then corresponds to

$$
L\ddot{q}+R\dot{q}+\frac{1}{C}q=0.
$$

The same inferential framework can therefore be applied to other damped second-order systems with equivalent mathematical structure.

## Author

Ozymandias McEvoy
