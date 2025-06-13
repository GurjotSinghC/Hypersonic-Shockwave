
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
