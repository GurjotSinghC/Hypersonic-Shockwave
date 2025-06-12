# # # # Python code to read image
# # # import cv2

# # # # To read image from disk, we use
# # # # cv2.imread function, in below method,
# # # img = cv2.imread("fixedproper.jpg", cv2.IMREAD_COLOR)
# # # cv2.namedWindow('wundu',cv2.WINDOW_NORMAL)
# # # # Creating GUI window to display an image on screen
# # # # first Parameter is windows title (should be in string format)
# # # # Second Parameter is image array
# # # cv2.imshow("wundu", img)

# # # # To hold the window on screen, we use cv2.waitKey method
# # # # Once it detected the close input, it will release the control
# # # # To the next line
# # # # First Parameter is for holding screen for specified milliseconds
# # # # It should be positive integer. If 0 pass an parameter, then it will
# # # # hold the screen until user close it.
# # # cv2.waitKey(0)

# # # # It is for removing/deleting created GUI window from screen
# # # # and memory
# # # cv2.destroyAllWindows()

# # # # import json

# # # # x = '{"employee":{"name":["y":5555], "age":30, "city":"New York"}}'

# # # # y = json.loads(x)

# # # # print(y["employee"]["name"])

# # # if(1>3):
# # #      a = input()

# # # else:
# # #     print("YOLO")

# # # if(a):
# # #     print("value =",a)
# # a = ""

# # # print(len(a))

# # import numpy as np
# # import matplotlib.pyplot as plt

# # # --- 1. Define Simulation Parameters ---
# # grid_size = 50  # Number of grid points in x and y directions (e.g., 50x50 grid)
# # tolerance = 1e-4 # Convergence criterion: stop when max temperature change is below this
# # max_iterations = 10000 # Maximum number of iterations to prevent infinite loops

# # # Define boundary temperatures
# # T_hot_top = 100.0   # Temperature at the top edge
# # T_cold_bottom = 0.0 # Temperature at the bottom edge
# # T_left = 75.0       # Temperature at the left edge
# # T_right = 25.0      # Temperature at the right edge

# # # --- 2. Initialize the Temperature Field (Grid) ---
# # # Create a 2D NumPy array for temperature, initialized to zeros.
# # # This array will store the temperature at each grid point.
# # T = np.zeros((grid_size, grid_size))

# # # Apply boundary conditions (Dirichlet boundaries - fixed temperatures)
# # # Top edge: Set the first row (index 0) to T_hot_top
# # T[0, :] = T_hot_top
# # # Bottom edge: Set the last row (index grid_size-1) to T_cold_bottom
# # T[grid_size - 1, :] = T_cold_bottom
# # # Left edge: Set the first column (index 0) to T_left
# # T[:, 0] = T_left
# # # Right edge: Set the last column (index grid_size-1) to T_right
# # T[:, grid_size - 1] = T_right

# # # The corners might be overwritten multiple times, which is fine;
# # # their values are fixed by the most recent boundary condition application.

# # print("Initial temperature grid with boundary conditions:")
# # print(T)

# # # --- 3. Iterative Solver (Gauss-Seidel Method) ---
# # # This loop will repeatedly update the temperature at each interior point
# # # until the solution converges (i.e., temperatures stop changing significantly).

# # iteration = 0
# # max_delta = 1.0 # Initialize max_delta to a value greater than tolerance

# # print("\nStarting iterative solver...")
# # while max_delta > tolerance and iteration < max_iterations:
# #     # Create a copy of the current temperature field to store the previous state.
# #     # This is necessary to calculate the change (delta) after updating the grid.
# #     T_old = T.copy()

# #     # Iterate over the interior points of the grid.
# #     # We skip the boundary points as their temperatures are fixed.
# #     for i in range(1, grid_size - 1): # Rows (y-direction)
# #         for j in range(1, grid_size - 1): # Columns (x-direction)
# #             # Apply the finite difference equation (Gauss-Seidel update)
# #             # T[i,j] is updated using the *most recently computed* neighbor values.
# #             T[i, j] = 0.25 * (T[i + 1, j] + T[i - 1, j] + \
# #                               T[i, j + 1] + T[i, j - 1])

# #     # Calculate the maximum absolute difference between the new and old temperature fields.
# #     # This 'max_delta' tells us how much the solution changed in this iteration.
# #     max_delta = np.max(np.abs(T - T_old))

# #     iteration += 1
# #     if iteration % 500 == 0: # Print update every 500 iterations
# #         print(f"Iteration: {iteration}, Max Delta: {max_delta:.6f}")

# # print(f"\nSolver finished after {iteration} iterations.")
# # print(f"Final Max Delta: {max_delta:.6f}")

# # # --- 4. Visualize the Temperature Field (View Window) ---
# # # Matplotlib is used to create a heatmap of the temperature distribution.

# # plt.figure(figsize=(8, 6)) # Set the figure size for better readability
# # # Use imshow to display the 2D array as an image/heatmap.
# # # 'origin='lower'' sets the (0,0) index to the bottom-left of the plot.
# # # 'cmap='jet'' sets the colormap (e.g., 'jet', 'viridis', 'hot', 'cool').
# # plt.imshow(T, cmap='jet', origin='lower', extent=[0, grid_size, 0, grid_size])

# # # Add a color bar to show the temperature scale
# # plt.colorbar(label='Temperature ($^{\circ}C$)')

# # # Add title and labels
# # plt.title('2D Steady-State Heat Conduction', fontsize=16)
# # plt.xlabel('X-coordinate', fontsize=12)
# # plt.ylabel('Y-coordinate', fontsize=12)

# # # Add grid lines for better visualization of discrete points
# # plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)

# # # Display the plot
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# # import relevant solver library/wrapper (e.g., pyXFOIL, or code for subprocess calls)

# def run_airfoil_solver(angle_of_attack, mach_number, reynolds_number, airfoil_points=None):
#     """
#     This function would interface with your chosen fluid dynamics solver.
#     It takes flow conditions and potentially perturbed airfoil points,
#     runs the simulation, and returns lift and drag coefficients.
#     """
#     # Example: If using XFOIL via subprocess
#     # Generate XFOIL input script
#     # Run XFOIL: subprocess.run(['xfoil', '< input_script.txt'], capture_output=True, text=True)
#     # Parse XFOIL output to get CL and CD
#     # For simplicity, let's use dummy values for now:
#     cl = 0.5 + 0.05 * angle_of_attack + np.random.normal(0, 0.01) # Dummy lift
#     cd = 0.02 + 0.001 * angle_of_attack**2 + np.random.normal(0, 0.0005) # Dummy drag
#     return cl, cd

# # --- Monte Carlo Simulation Setup ---
# num_samples = 1000

# # Input parameter distributions
# # Angle of attack: Normal distribution (mean=5 deg, std_dev=1 deg)
# mean_alpha = 5.0
# std_alpha = 1.0
# sampled_alphas = np.random.normal(mean_alpha, std_alpha, num_samples)

# # Mach number: Uniform distribution (0.1 to 0.3)
# min_mach = 0.1
# max_mach = 0.3
# sampled_machs = np.random.uniform(min_mach, max_mach, num_samples)

# # Reynolds number: Normal distribution (mean=3e6, std_dev=0.5e6)
# mean_re = 3e6
# std_re = 0.5e6
# sampled_res = np.random.normal(mean_re, std_re, num_samples)

# # Initialize lists to store results
# results_cl = []
# results_cd = []

# # --- Monte Carlo Loop ---
# print("Running Monte Carlo simulation...")
# for i in range(num_samples):
#     alpha = sampled_alphas[i]
#     mach = sampled_machs[i]
#     re = sampled_res[i]

#     # In a real simulation, you might also perturb airfoil geometry here
#     # perturbed_airfoil = original_airfoil + np.random.normal(0, tolerance_std_dev, original_airfoil.shape)

#     cl, cd = run_airfoil_solver(alpha, mach, re) # Pass perturbed_airfoil if applicable
#     results_cl.append(cl)
#     results_cd.append(cd)

# print("Simulation complete. Analyzing results...")

# # --- Analysis and Visualization ---
# results_cl = np.array(results_cl)
# results_cd = np.array(results_cd)

# print(f"\nMean Lift Coefficient (CL): {np.mean(results_cl):.4f}")
# print(f"Std Dev Lift Coefficient (CL): {np.std(results_cl):.4f}")
# print(f"Mean Drag Coefficient (CD): {np.mean(results_cd):.4f}")
# print(f"Std Dev Drag Coefficient (CD): {np.std(results_cd):.4f}")

# # Plotting
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.hist(results_cl, bins=30, edgecolor='black', alpha=0.7)
# plt.title('Distribution of Lift Coefficient ($C_L$)')
# plt.xlabel('$C_L$')
# plt.ylabel('Frequency')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.hist(results_cd, bins=30, edgecolor='black', alpha=0.7)
# plt.title('Distribution of Drag Coefficient ($C_D$)')
# plt.xlabel('$C_D$')
# plt.ylabel('Frequency')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Optional: Scatter plot of CL vs. CD
# plt.figure(figsize=(8, 6))
# plt.scatter(results_cd, results_cl, alpha=0.5, s=10)
# plt.title('Lift Coefficient vs. Drag Coefficient')
# plt.xlabel('$C_D$')
# plt.ylabel('$C_L$')
# plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Constants and Upstream Conditions ---
# Ratio of specific heats for air (gamma)
gamma = 1.4

# Upstream Mach number (M1) - This represents the flow speed before the test section.
# Hypersonic flow typically refers to Mach numbers greater than 5.
M1 = 10

# Oblique shock wave angle (beta) in degrees.
# This angle is crucial as it determines the strength and characteristics of the shock.
# For a given M1 and wedge angle, there can be two possible shock angles (strong/weak),
# or no solution if the wedge angle is too large.
# For this simplified visualization, we assume a specific shock angle and then calculate
# the corresponding wedge (deflection) angle.
beta_deg = 20.0 # it was previosuly 30 
beta_rad = np.deg2rad(beta_deg)

# --- 2. Oblique Shock Wave Calculations ---
# These equations are derived from the conservation laws (mass, momentum, energy)
# applied across a stationary oblique shock wave.

# Calculate the normal component of the upstream Mach number (Mn1).
# This component determines whether a shock forms (Mn1 > 1).
Mn1 = M1 * np.sin(beta_rad)

# Check for shock formation criterion. If Mn1 is less than 1, a shock cannot form.
if Mn1 < 1.0:
    print(f"Warning: Normal Mach number before shock (Mn1 = {Mn1:.2f}) is less than 1.")
    print("This indicates that for the given M1 and beta, a stable oblique shock may not form.")
    print("Please adjust beta to ensure Mn1 > 1 for a physically realistic shock.")
    # For M1=5, beta must be roughly > 11.5 degrees for a shock.

# Calculate the normal component of the Mach number after the shock (Mn2).
Mn2_squared = (Mn1**2 + 2 / (gamma - 1)) / ((2 * gamma / (gamma - 1)) * Mn1**2 - 1)
Mn2 = np.sqrt(Mn2_squared)

# Calculate the deflection angle (theta) of the flow.
# This is the angle of the wedge that would produce this specific shock angle.
# The formula is derived from the oblique shock relations.
tan_theta = 2 * (1 / np.tan(beta_rad)) * \
            (M1**2 * np.sin(beta_rad)**2 - 1) / \
            (M1**2 * (gamma + np.cos(2 * beta_rad)) + 2)
theta_rad = np.arctan(tan_theta)
theta_deg = np.rad2deg(theta_rad)

# Calculate the Mach number after the shock (M2).
M2 = Mn2 / np.sin(beta_rad - theta_rad)

# Calculate the pressure ratio across the shock (P2/P1).
# P1 is the upstream pressure, P2 is the downstream pressure.
P2_P1 = 1 + (2 * gamma / (gamma + 1)) * (Mn1**2 - 1)

# Calculate the density ratio across the shock (rho2/rho1).
# rho1 is the upstream density, rho2 is the downstream density.
rho2_rho1 = ((gamma + 1) * Mn1**2) / ((gamma - 1) * Mn1**2 + 2)

# Calculate the temperature ratio across the shock (T2/T1).
# T1 is the upstream temperature, T2 is the downstream temperature.
T2_T1 = P2_P1 / rho2_rho1

# Print the calculated flow properties for user information
print("\n--- Calculated Flow Properties Across Oblique Shock ---")
print(f"Upstream Mach Number (M1): {M1:.2f}")
print(f"Assumed Oblique Shock Angle (beta): {beta_deg:.2f} degrees")
print(f"Calculated Deflection Angle (theta): {theta_deg:.2f} degrees")
print(f"Normal Mach Number before shock (Mn1): {Mn1:.2f}")
print(f"Normal Mach Number after shock (Mn2): {Mn2:.2f}")
print(f"Downstream Mach Number (M2): {M2:.2f}")
print(f"Pressure Ratio (P2/P1): {P2_P1:.2f}")
print(f"Density Ratio (rho2/rho1): {rho2_rho1:.2f}")
print(f"Temperature Ratio (T2/T1): {T2_T1:.2f}")

# --- 3. Visualization Setup ---
# Define the spatial domain for the wind tunnel visualization
x_min, x_max = 0, 10 # X-axis range (length of the test section)
y_min, y_max = -3, 3 # Y-axis range (height of the test section, centered at y=0)
nx, ny = 400, 240 # Number of grid points (resolution of the visualization)

# Create a 2D grid using numpy's meshgrid for plotting contours
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Define the apex (leading edge) of the wedge model in the tunnel
x_apex = 2.0
y_apex = 0.0

# Initialize the property fields (Mach number, pressure ratio, etc.)
# All points are initially set to upstream conditions (M1, P1/P1=1, etc.)
mach_field = np.full(X.shape, M1)
pressure_ratio_field = np.full(X.shape, 1.0)
density_ratio_field = np.full(X.shape, 1.0)
temperature_ratio_field = np.full(X.shape, 1.0)

# Identify the region downstream of the oblique shock wave.
# The shock wave originates from the wedge apex and extends outwards at 'beta' angle.
# We consider both upper (y >= 0) and lower (y < 0) symmetric shock waves.

# Condition for points downstream of the upper shock (for y >= 0):
# These points are to the right of the apex AND below the upper shock line.
# The upper shock line equation is: (Y - y_apex) = tan(beta_rad) * (X - x_apex)
downstream_upper_condition = (Y >= y_apex) & ((Y - y_apex) < np.tan(beta_rad) * (X - x_apex))

# Condition for points downstream of the lower shock (for y < 0):
# These points are to the right of the apex AND above the lower shock line.
# The lower shock line equation is: (Y - y_apex) = -tan(beta_rad) * (X - x_apex)
downstream_lower_condition = (Y < y_apex) & ((Y - y_apex) > -np.tan(beta_rad) * (X - x_apex))

# Combine conditions: a point is downstream if it's to the right of the apex AND
# satisfies either the upper or lower shock condition.
downstream_region = (X > x_apex) & (downstream_upper_condition | downstream_lower_condition)

# Apply the calculated post-shock values to the identified downstream region.
mach_field[downstream_region] = M2
pressure_ratio_field[downstream_region] = P2_P1
density_ratio_field[downstream_region] = rho2_rho1
temperature_ratio_field[downstream_region] = T2_T1

# Define the geometry of the wedge for plotting.
# The wedge starts at the apex and extends downstream at the deflection angle theta.
wedge_length = 4.0 # Length of the wedge
wedge_end_x = x_apex + wedge_length
wedge_upper_y = y_apex # The top surface of the wedge is at y_apex
wedge_lower_y = y_apex - np.tan(theta_rad) * wedge_length # The bottom surface angle

# Coordinates to draw the triangular wedge.
wedge_x = [x_apex, wedge_end_x, x_apex]
wedge_y = [y_apex, wedge_lower_y, y_apex]

# --- 4. Plotting the Visualization ---
# Create a figure with two subplots: one for Mach number and one for Pressure Ratio.
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle(f"Hypersonic Wind Tunnel Test: Flow Over a Wedge\n"
             f"(Upstream M={M1}, Assumed Shock Angle={beta_deg}°, Calculated Wedge Angle={theta_deg:.1f}°)",
             fontsize=14)

# --- Plot 1: Mach Number Contour ---
ax1 = axes[0]
im1 = ax1.imshow(mach_field, extent=[x_min, x_max, y_min, y_max], origin='lower',
                   cmap='viridis', aspect='auto', interpolation='bilinear',
                   vmin=min(M1, M2), vmax=max(M1, M2)) # Set color scale limits based on min/max Mach
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Mach Number ($M$)')
ax1.set_title('Mach Number Contour')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_aspect('equal', adjustable='box') # Maintain aspect ratio for accurate geometry

# Draw the wedge as a filled gray polygon
ax1.fill(wedge_x, wedge_y, color='gray', label='Wedge Model')

# Draw the oblique shock wave lines (dashed red lines)
# The shock extends from the apex outwards at the beta angle.
# We draw a line for the upper and lower symmetric shocks.
ax1.plot([x_apex, x_apex + np.cos(beta_rad) * (x_max - x_apex)],
         [y_apex, y_apex + np.sin(beta_rad) * (x_max - x_apex)],
         'r--', linewidth=2, label=f'Oblique Shock Wave (β={beta_deg}°)')
ax1.plot([x_apex, x_apex + np.cos(beta_rad) * (x_max - x_apex)],
         [y_apex, y_apex - np.sin(beta_rad) * (x_max - x_apex)],
         'r--', linewidth=2)

# Draw the wedge surface line (solid black line)
ax1.plot([x_apex, wedge_end_x], [y_apex, wedge_lower_y], 'k-', linewidth=2, label=f'Wedge Surface (θ={theta_deg:.1f}°)')

# Add text annotations for upstream and downstream Mach numbers
ax1.text(x_apex - 0.5, y_apex + 0.5, f'$M_1={M1}$', color='blue', fontsize=12, ha='right')
ax1.text(x_apex + 4, y_apex + 0.5, f'$M_2={M2:.2f}$', color='orange', fontsize=12, ha='left')
ax1.legend(loc='upper right')
ax1.grid(False) # Turn off grid for cleaner contour plot

# --- Plot 2: Pressure Ratio Contour ---
ax2 = axes[1]
im2 = ax2.imshow(pressure_ratio_field, extent=[x_min, x_max, y_min, y_max], origin='lower',
                   cmap='plasma', aspect='auto', interpolation='bilinear',
                   vmin=1.0, vmax=P2_P1) # Set color scale limits
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Pressure Ratio ($P/P_1$)')
ax2.set_title('Pressure Ratio Contour')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_aspect('equal', adjustable='box')

# Redraw the wedge and shock wave for the second plot for consistency
ax2.fill(wedge_x, wedge_y, color='gray')
ax2.plot([x_apex, x_apex + np.cos(beta_rad) * (x_max - x_apex)],
         [y_apex, y_apex + np.sin(beta_rad) * (x_max - x_apex)],
         'r--', linewidth=2)
ax2.plot([x_apex, x_apex + np.cos(beta_rad) * (x_max - x_apex)],
         [y_apex, y_apex - np.sin(beta_rad) * (x_max - x_apex)],
         'r--', linewidth=2)
ax2.plot([x_apex, wedge_end_x], [y_apex, wedge_lower_y], 'k-', linewidth=2)

# Add text annotations for upstream and downstream pressure ratios
ax2.text(x_apex - 0.5, y_apex + 0.5, '$P_1$', color='blue', fontsize=12, ha='right')
ax2.text(x_apex + 4, y_apex + 0.5, f'$P_2/P_1={P2_P1:.2f}$', color='orange', fontsize=12, ha='left')
ax2.grid(False)

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
