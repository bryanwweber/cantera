# This file is part of Cantera. See License.txt in the top-level directory or
# at https://cantera.org/license.txt for license and copyright information.

"""
Diffusion flame two-point control
=================================

This example uses the two-point flame control feature to march solutions
down the stable and unstable burning branch for a counterflow diffusion flame.
A hydrogen-oxygen diffusion flame at 1 bar is studied.

Requires: cantera >= 3.0, matplotlib >= 2.0

.. tags:: Python, combustion, 1D flow, diffusion flame, strained flame, extinction,
          saving output, plotting, two-point control
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import cantera as ct


# PART 1: INITIALIZATION

# Set up an initial hydrogen-oxygen counterflow flame at 1 bar and low strain
# rate (maximum axial velocity gradient = 2414 1/s)

reaction_mechanism = 'h2o2.yaml'
gas = ct.Solution(reaction_mechanism)
width = 18.e-3  # 18mm wide
f = ct.CounterflowDiffusionFlame(gas, width=width)

# Define the operating pressure and boundary conditions
f.P = 1.e5  # 1 bar
f.fuel_inlet.mdot = 0.5  # kg/m^2/s
f.fuel_inlet.X = 'H2:1'
f.fuel_inlet.T = 300  # K
f.oxidizer_inlet.mdot = 3.0  # kg/m^2/s
f.oxidizer_inlet.X = 'O2:1'
f.oxidizer_inlet.T = 500  # K

# Set refinement parameters
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.03)

# Define a limit for the maximum temperature below which the flame is
# considered as extinguished and the computation is aborted
temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T)

# Initialize and solve
print('Creating the initial solution')
f.solve(loglevel=0, auto=True)

# Define output locations
output_path = Path() / "diffusion_flame_continuation_data"
output_path.mkdir(parents=True, exist_ok=True)

hdf_output = "native" in ct.hdf_support()
if hdf_output:
    file_name = output_path / "flame_data.h5"
    file_name.unlink(missing_ok=True)

def names(test):
    if hdf_output:
        # use internal container structure for HDF
        file_name = output_path / "flame_data.h5"
        return file_name, test
    # use separate files for YAML
    file_name = output_path / f"{test}.yaml".replace("-", "_").replace("/", "_")
    return file_name, "solution"

file_name, entry = names("initial-solution")
f.save(file_name, name=entry, description="Initial solution")


# PART 2: CONTINUATION

f.two_point_control_enabled = True
spacing = 0.95
temperature_increment = 1.0 # Kelvin
maximum_temperature = []
a_max = []
for i in range(5000):
    control_temperature = np.min(f.T) + spacing*(np.max(f.T) - np.min(f.T))
    print(f'Iteration {i}: Control temperature = {control_temperature} K')
    f.set_left_control_point(control_temperature)
    f.set_right_control_point(control_temperature)

    # This decrement is what drives the two-point control. If failure
    # occurs, try decreasing the decrement.
    f.left_control_point_temperature -= temperature_increment
    f.right_control_point_temperature -= temperature_increment

    try:
        f.solve(loglevel=0)
    except ct.CanteraError as e:
        print('Error: Did not converge')
        break

    maximum_temperature.append(np.max(f.T))
    a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))


# Plot the maximum temperature versus the maximum axial velocity gradient
plt.figure()
plt.plot(a_max, maximum_temperature)
plt.xlabel('Maximum Axial Velocity Gradient [1/s]')
plt.ylabel('Maximum Temperature [K]')
plt.savefig(output_path / "figure_max_temperature_vs_max_velocity_gradient.png")
plt.close()

# Plot maximum_temperature against number of iterations
plt.figure()
plt.plot(range(len(maximum_temperature)), maximum_temperature)
plt.xlabel('Number of Continuation Steps')
plt.ylabel('Maximum Temperature [K]')
plt.savefig(output_path / "figure_max_temperature_iterations.png")
plt.close()



