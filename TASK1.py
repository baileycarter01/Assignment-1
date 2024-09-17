#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import time

# Parameters
L = 0.1  # Length of the channel (m)
H = 0.01 # Height of the channel (m)
dx = 0.001  # Grid spacing (m)
dy = 0.001  # Grid spacing (m)
dt = 0.0001  # Time step (s)
T_END = 30  # Total simulation time (s)

# Constants for convection and heat transfer
k = 0.6     # Thermal conductivity (W/m.K)
q = 5000.0  # Heat flux (W/m^2)
mu = 1e-3   # Dynamic viscosity (Pa.s)
rho = 1000  # Density (kg/m^3)
cp = 4186   # Specific heat capacity (J/kg.K)
dp_dx = 2.5 # Pressure gradient (Pa/m)

# Thermal diffusivity (alpha = k / (rho * cp))
alpha = k / (rho * cp)

# Velocity profile for Poiseuille flow (stronger convection effect)
def get_velocity(y):
    return (dp_dx * (H**2 - (y - H/2)**2)) / (8 * mu)  # Adjusted for stronger convection

# Grid setup
NX = int(L / dx)
NY = int(H / dy)
x = np.linspace(dx/2., L-dx/2., NX)
y = np.linspace(dy/2., H-dy/2., NY)
xx, yy = np.meshgrid(x, y, indexing='ij')

# Initial and boundary conditions
T_COOL = 300.0
T_HOT = 700.0  # Hot wall condition

# Initialize temperature field with initial condition
T = np.ones((NX, NY)) * T_COOL

# Set the left boundary to be heated (hot wall)
T[0, :] = T_HOT

# Time stepping
simulated_time = 0
iteration = 0
PLOT_EVERY = 100
max_temps = []

# Plot setup
fig = plt.figure(figsize=(12, 12))
ax0 = fig.add_subplot(2, 2, 1)

tic = time.time()

while simulated_time < T_END:
    # Calculate velocity profile
    u = get_velocity(yy)
    v = np.zeros_like(u)  # No flow in y direction

    # Calculate fluxes - interior
    x_flux = np.zeros((NX, NY))
    y_flux = np.zeros((NX, NY))

    # Convection + Diffusion (flux) in x and y directions
    x_flux[1:, :] = u[1:, :] * (T[1:, :] - T[:-1, :]) / dx  # Convection in x
    y_flux[:, 1:] = v[:, 1:] * (T[:, 1:] - T[:, :-1]) / dy  # Convection in y

    # Add the diffusion term (thermal diffusivity * second derivative)
    x_diffusion = np.zeros_like(T)
    y_diffusion = np.zeros_like(T)
    
    x_diffusion[1:-1, 1:-1] = alpha * (
        (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dx**2)
    )
    y_diffusion[1:-1, 1:-1] = alpha * (
        (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / (dy**2)
    )

    x_flux[1:, :] += x_diffusion[1:, :]  # Add diffusion in x
    y_flux[:, 1:] += y_diffusion[:, 1:]  # Add diffusion in y

    # Boundary conditions for fluxes (heat flux applied at the left and right walls)
    x_flux[0, :] = -q / k  # Left wall (hot)
    x_flux[-1, :] = q / k  # Right wall (cold)
    y_flux[:, 0] = -q / k  # Bottom wall
    y_flux[:, -1] = q / k  # Top wall

    # Update temperature (Diffusion + Convection)
    T_new = np.copy(T)
    T_new[1:-1, 1:-1] += dt * (
        (x_flux[2:, 1:-1] - x_flux[1:-1, 1:-1]) / dx +
        (y_flux[1:-1, 2:] - y_flux[1:-1, 1:-1]) / dy
    )

    # Clip temperatures to avoid extreme values (helps with numerical stability)
    T = np.clip(T_new, T_COOL, T_HOT)

    # Track maximum temperature
    max_temps.append(np.max(T))

    # Plotting
    if np.mod(iteration, PLOT_EVERY) == 0:
        ax0.cla()
        ax0.contourf(xx, yy, T, cmap='hot', levels=50)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_title(f'Temperature at t = {simulated_time:.2f} s')
        ax0.set_aspect('equal')

        if simulated_time in [5, 10, 20, 30]:
            fig.savefig(f'heatmap_{int(simulated_time)}s.png')

    simulated_time += dt
    iteration += 1

# Final plots
# Heat-map at specified times
for t in [5, 10, 20, 30]:
    plt.figure()
    plt.contourf(xx, yy, T, cmap='hot', levels=50)
    plt.colorbar()
    plt.title(f'Temperature field at t = {t}s')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig(f'heatmap_{int(t)}s.png')
    plt.close()

# Vertical profiles at x = {0.125L, 0.25L, 0.5L}
x_positions = [0.125 * L, 0.25 * L, 0.5 * L]
for x_pos in x_positions:
    x_idx = int(x_pos / dx)
    plt.plot(y, T[x_idx, :], label=f'x = {x_pos}m')

plt.xlabel('y (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profiles at t = 30s')
plt.legend()
plt.savefig('temperature_profiles.png')
plt.close()

# Maximum temperature plot
plt.figure()
plt.plot(np.arange(0, len(max_temps) * dt, dt), max_temps)
plt.xlabel('Time (s)')
plt.ylabel('Maximum Temperature (K)')
plt.title('Maximum Temperature Over Time')
plt.savefig('max_temperature.png')
plt.close()

print("Total elapsed time:", round(time.time()-tic, 2), "s (", round((time.time()-tic)/60.0, 2), "min)")
