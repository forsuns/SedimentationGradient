globals().clear()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import rcParams

# 전체 글꼴 변경
rcParams['font.family'] = 'Helvetica'  # 예: Arial 폰트로 설정
rcParams['font.size'] = 20
  
# Figure configuration
fig_width = 10  # inches
fig_height = 8  # inches

# Constants
alpha = 1.0
rho0 = 1025.0
Cx = 1.0
vsink = 0.0000036

# Data loading
avg_data = pd.read_csv('./avg411.dat', delim_whitespace=True, header=None, names=["ll", "xxx", "yyy", "bot", "spd"])
dxdy_data = pd.read_csv('../../Run/ss1/dxdy.dat', delim_whitespace=True, header=None, names=["ii", "jj", "dx", "dy", "dep", "bot", "z0"])

# Extract values
spd = avg_data['spd'].values
bot = avg_data['bot'].values
h = -bot

# Adjust invalid data
h[h <= 4.0] = np.nan
spd[spd <= 0.00001] = np.nan

# Calculate sedimentation gradient
sedg = (alpha / spd) * (Cx / rho0) * (vsink / h)
sdr = sedg.copy()

# Load sediment data
sed1_data  = pd.read_csv('../../Run_Final/ss441/sed1.dat', delim_whitespace=True, skiprows=1, header=None)
sed31_data = pd.read_csv('../../Run_Final/ss441/sed31.dat', delim_whitespace=True, skiprows=1, header=None)

# Velocity calculation
vel1 = (sed31_data[5] - sed1_data[5]) * 1000  # Example column indexing

# Outlier removal
sdr[np.abs(sdr) > 1000] = np.nan

# Data selection
xx_vector = []
yy_vector = []
mag = []
vel = []

for i in range(len(sdr)):
    if np.isfinite(sdr[i]) and vel1[i] > 0:
        xx_vector.append(avg_data.iloc[i]['xxx'])
        yy_vector.append(avg_data.iloc[i]['yyy'])
        mag.append(vel1[i])
        vel.append(sdr[i])

xx_vector = np.array(xx_vector)
yy_vector = np.array(yy_vector)
mag = np.array(mag)
vel = np.array(vel)

# Plot configuration
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

# Standardization
vel= (vel - np.min(vel)) / (np.max(vel) - np.min(vel))
mag= (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

plt.figure(figsize=(fig_width, fig_height))
plt.scatter(vel, mag, label="Data", s=30)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(vel, mag)
fit_line = slope * vel + intercept

plt.plot(vel, fit_line, color='k', linewidth=1.5, label=f"Fit: R²={r_value**2:.2f}")

# Print p-value
print(f"P-value: {p_value:.4e}")

# Plot details
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('k', fontsize=20)
plt.ylabel('Sedimentation Rate', fontsize=20)
#plt.title('Sedimentation Analysis', fontsize=20)
plt.legend()
plt.grid(True)

# Add text annotations
plt.text(xmin + (xmax - xmin) * 0.1, ymax - (ymax - ymin) * 0.10, f"Slope = {slope:.4e}", fontsize=20)
plt.text(xmin + (xmax - xmin) * 0.1, ymax - (ymax - ymin) * 0.15, f"Y-int = {intercept:.2e}", fontsize=20)
plt.text(xmin + (xmax - xmin) * 0.1, ymax - (ymax - ymin) * 0.20, f"R = {r_value:.2f}", fontsize=20)
plt.text(xmin + (xmax - xmin) * 0.1, ymax - (ymax - ymin) * 0.25, f"P-value = {p_value:.2e}", fontsize=20)

plt.tight_layout()  # 여백 자동 조정

# Save the plot
plt.savefig('linear_441.png', dpi=200)
plt.show()
