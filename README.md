# Assignment-1
Prerequisites
Before running the code, ensure you have the following Python packages installed:

numpy
matplotlib
time
You can install these packages using pip:

bash
Copy code
pip install numpy matplotlib
Task 1: Heat Transfer in a Channel
This script simulates heat transfer in a 2D channel, where convection and diffusion processes are considered. The script also generates contour plots for temperature distribution and temperature profiles at various locations.

Running the Simulation
To run the heat transfer simulation and generate the plots, execute the following command:

bash
Copy code
python task1_heat_transfer.py
This will generate and save the following plots:

heatmap_5s.png, heatmap_10s.png, heatmap_20s.png, heatmap_30s.png: Contour plots of the temperature distribution at different times.
temperature_profiles.png: Temperature profiles at different x-positions (0.125L, 0.25L, and 0.5L).
max_temperature.png: Plot of maximum temperature over time.
Output Files
Heatmaps: Heatmap plots for temperature distribution at different time steps (t = 5s, t = 10s, t = 20s, and t = 30s).
Temperature Profiles: Temperature profiles at x = 0.125L, 0.25L, and 0.5L.
Maximum Temperature: A plot showing the maximum temperature as a function of time.
Task 3: Lid-Driven Cavity Flow Simulation
This script simulates the lid-driven cavity flow problem for two different Reynolds numbers (Re = 1000 and Re = 2500). The results include velocity field plots, velocity profiles, and animations.

Running the Simulation
To run the lid-driven cavity flow simulation for Reynolds numbers 1000 and 2500, use the following command:

bash
Copy code
python task3_lid_driven_cavity.py
The script will generate and save the following plots and animations:

Velocity field plots for different time steps.
Velocity profiles (horizontal and vertical) compared to reference data.
An animation showing the development of the velocity field over time for each Reynolds number.
Output Files
Velocity Field Plots: Velocity field at specific time steps for both Re = 1000 and Re = 2500.
Velocity Profiles: Horizontal and vertical velocity profiles for both Reynolds numbers compared to reference data.
Streamlines: Streamline plots to visualize regions of circulation for both Reynolds numbers.
Animation: An animation showing the evolution of the velocity field over time for Re = 1000.
Notes
The simulation checks for steady-state conditions and stops if a steady state is detected before the maximum number of steps.
The PLOT_EVERY variable controls how often the plots are saved. You can adjust this value for different visualization intervals.
Conclusion
This repository contains the simulation code for heat transfer and lid-driven cavity flow problems. Follow the instructions to generate the results and visualize the physical phenomena using Python and matplotlib. Feel free to modify the parameters, grid resolution, and Reynolds numbers for further exploration.
