#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# System Parameters
LX = 1.0         # [m]
LY = 1.0         # [m]

# Fluid Properties
RHO = 1.0        # Density [kg/m^3]

# Boundary Conditions
UNORTH = 1.0     # [m/s]
USOUTH = 0.0
VEAST = 0.0
VWEST = 0.0

# Discretization
NX = 50
NY = 50
DT = 0.01
NUM_STEPS = 2000
PLOT_EVERY = 100

# Reynolds numbers for two cases
RE_LIST = [1000, 2500]
MU_LIST = [1 / re for re in RE_LIST]  # Dynamic viscosity for each Reynolds number

def initialize_fields():
    u = np.zeros((NX+1, NY+2), float)   # Velocity in x-direction
    v = np.zeros((NX+2, NY+1), float)   # Velocity in y-direction
    p = np.zeros((NX+2, NY+2), float)   # Pressure field
    return u, v, p

def apply_boundary_conditions(u, v):
    u[:, 0] = 2. * USOUTH - u[:, 1]
    u[:, -1] = 2. * UNORTH - u[:, -2]
    v[0, :] = 2. * VWEST - v[1, :]
    v[-1, :] = 2. * VEAST - v[-2, :]

def check_steady_state(prev_u, prev_v, u, v, threshold=1e-4):
    return (np.max(np.abs(u - prev_u)) < threshold) and (np.max(np.abs(v - prev_v)) < threshold)

def simulate_lid_driven_cavity(Re, mu):
    print(f"Running simulation for Re = {Re}")

    # Initialize fields
    u, v, p = initialize_fields()
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    prhs = np.zeros_like(p)

    # Grid spacing
    dx = LX / NX
    dy = LY / NY
    dxdy = dx * dy

    # Create mesh grid for visualization
    xnodes = np.linspace(0, LX, NX)
    ynodes = np.linspace(0, LY, NY)
    xx, yy = np.meshgrid(xnodes, ynodes)
    
    # Set initial boundary conditions
    apply_boundary_conditions(u, v)

    # Prepare for animation
    frames = []

    # Start timing
    start_time = time.time()

    prev_u = np.copy(u)
    prev_v = np.copy(v)
    
    for steps in range(NUM_STEPS):
        # Compute fluxes
        J_u_x = 0.25 * (u[:-1, 1:-1] + u[1:, 1:-1]) ** 2 - mu * (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
        J_u_y = 0.25 * (u[1:-1, 1:] + u[1:-1, :-1]) * (v[2:-1, :] + v[1:-2, :]) - mu * (u[1:-1, 1:] - u[1:-1, :-1]) / dy

        J_v_x = 0.25 * (u[:, 2:-1] + u[:, 1:-2]) * (v[1:, 1:-1] + v[:-1, 1:-1]) - mu * (v[1:, 1:-1] - v[:-1, 1:-1]) / dx
        J_v_y = 0.25 * (v[1:-1, 1:] + v[1:-1, :-1]) ** 2 - mu * (v[1:-1, 1:] - v[1:-1, :-1]) / dy

        # Update intermediate velocities
        ut[1:-1, 1:-1] = u[1:-1, 1:-1] - (DT / dxdy) * (dy * (J_u_x[1:, :] - J_u_x[:-1, :]) + dx * (J_u_y[:, 1:] - J_u_y[:, :-1]))
        vt[1:-1, 1:-1] = v[1:-1, 1:-1] - (DT / dxdy) * (dy * (J_v_x[1:, :] - J_v_x[:-1, :]) + dx * (J_v_y[:, 1:] - J_v_y[:, :-1]))

        # Enforce boundary conditions on intermediate velocities
        ut[:, 0] = 2. * USOUTH - ut[:, 1]
        ut[:, -1] = 2. * UNORTH - ut[:, -2]
        vt[0, :] = 2. * VWEST - vt[1, :]
        vt[-1, :] = 2. * VEAST - vt[-2, :]

        # Solve pressure Poisson equation
        divergence = (ut[1:, 1:-1] - ut[:-1, 1:-1]) / dx + (vt[1:-1, 1:] - vt[1:-1, :-1]) / dy
        prhs[1:-1, 1:-1] = divergence * RHO / DT
        p_next = np.zeros_like(p)

        for _ in range(50):  # Poisson solver iteration
            p_next[1:-1, 1:-1] = (-prhs[1:-1, 1:-1] * dxdy ** 2 + dy ** 2 * (p[:-2, 1:-1] + p[2:, 1:-1]) + dx ** 2 * (p[1:-1, :-2] + p[1:-1, 2:])) / (2 * dx ** 2 + 2 * dy ** 2)
            
            # Apply boundary conditions to the pressure field
            p_next[0, :] = p_next[1, :]   # Left boundary
            p_next[-1, :] = p_next[-2, :] # Right boundary
            p_next[:, 0] = p_next[:, 1]   # Bottom boundary
            p_next[:, -1] = p_next[:, -2] # Top boundary

            # Update p with p_next for the next iteration
            p = p_next.copy()
        
        # Correct the velocity field
        u[1:-1, 1:-1] = ut[1:-1, 1:-1] - DT * (1. / dx) * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / RHO
        v[1:-1, 1:-1] = vt[1:-1, 1:-1] - DT * (1. / dy) * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / RHO

        # Apply boundary conditions
        apply_boundary_conditions(u, v)

        # Check for steady-state
        if steps % 100 == 0:
            if check_steady_state(prev_u, prev_v, u, v):
                print(f"Steady state reached at step {steps}")
                break

            prev_u = np.copy(u)
            prev_v = np.copy(v)

        # Collect frames for animation
        if steps % PLOT_EVERY == 0:
            u_sliced = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
            v_sliced = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
            fig, ax = plt.subplots()
            cp = ax.contourf(xx, yy, np.sqrt(u_sliced**2 + v_sliced**2), cmap='viridis')
            quiver = ax.quiver(xx, yy, u_sliced, v_sliced, color='k')
            ax.set_title(f'Velocity Field at Step {steps}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(cp, ax=ax, label='Velocity Magnitude')
            plt.savefig(f'frame_{steps}_Re{Re}.png')
            frames.append([plt.gca().patch])  # Add the current frame to the frames list
            plt.close(fig)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation for Re = {Re} completed in {elapsed_time:.2f} seconds")
    
    return u, v, frames

def plot_profiles(u, v, Re, reference_data):
    xnodes = np.linspace(0, LX, NX)
    ynodes = np.linspace(0, LY, NY)
    x_center = int(NX / 2)
    u_profile = u[x_center, 1:-1]
    v_profile = v[1:-1, x_center]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(xnodes, u_profile, label=f'Horizontal profile (Re={Re})')
    plt.plot(reference_data['x'], reference_data['u'], 'k--', label='Reference')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Horizontal Velocity Profile (Re={Re})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ynodes, v_profile, label=f'Vertical profile (Re={Re})')
    plt.plot(reference_data['y'], reference_data['v'], 'k--', label='Reference')
    plt.xlabel('y')
    plt.ylabel('v')
    plt.title(f'Vertical Velocity Profile (Re={Re})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'profile_plots_Re{Re}.png')
    plt.close()

def plot_streamlines(u, v, Re):
    xnodes = np.linspace(0, LX, NX)
    ynodes = np.linspace(0, LY, NY)
    xx, yy = np.meshgrid(xnodes, ynodes)
    
    # Slice u and v to match xx and yy dimensions
    u_sliced = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
    v_sliced = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
    
    plt.figure(figsize=(8, 8))
    strm = plt.streamplot(xx, yy, u_sliced, v_sliced, color=np.sqrt(u_sliced**2 + v_sliced**2), linewidth=1, cmap='viridis')
    plt.title(f'Streamlines for Re = {Re}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(strm.lines, label='Velocity magnitude')
    plt.savefig(f'streamlines_Re{Re}.png')
    plt.close()

def create_velocity_field_animation(Re, frames):
    if not frames:
        print(f"No frames to create animation for Re = {Re}.")
        return

    fig, ax = plt.subplots()
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
    try:
        ani.save(f'velocity_field_animation_Re{Re}.gif', writer='pillow')
        print(f"Animation saved as velocity_field_animation_Re{Re}.gif")
    except Exception as e:
        print(f"Error saving animation: {e}")
    plt.close(fig)

# Main simulation loop
for Re, mu in zip(RE_LIST, MU_LIST):
    u, v, frames = simulate_lid_driven_cavity(Re, mu)
    plot_profiles(u, v, Re, reference_data={'x': np.linspace(0, 1, 50), 'u': np.sin(np.linspace(0, 1, 50)*np.pi), 'y': np.linspace(0, 1, 50), 'v': np.cos(np.linspace(0, 1, 50)*np.pi)})
    plot_streamlines(u, v, Re)
    create_velocity_field_animation(Re, frames)
