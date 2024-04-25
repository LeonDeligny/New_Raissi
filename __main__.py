import scipy.io

import pandas as pd, numpy as np

pd.set_option('display.max_rows', None)  # Set the option to display all rows

data = scipy.io.loadmat("cylinder_nektar_wake.mat")

U_star = data['U_star'] # N x 2 x T
p_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

# Averaging U, V and p over the time dimension
U_mean = U_star.mean(axis=2)[:, 0]  # Averaging over time, selecting U component
V_mean = U_star.mean(axis=2)[:, 1]  # Averaging over time, selecting V component
p_mean = p_star.mean(axis=1)  # Average pressure over time

# Create a DataFrame with averaged values
df_train = pd.DataFrame({
    'x': X_star[:, 0],
    'y': X_star[:, 1],
    'u': U_mean,
    'v': V_mean,
    'p': p_mean
})

x = np.linspace(-15.0, 10.0, 2501)
y = np.linspace(-6.0, 6.0, 1201)
xx, yy = np.meshgrid(x, y)
grid = np.column_stack([xx.ravel(), yy.ravel()])
df = pd.DataFrame(grid, columns=['x', 'y'])

# Assigning 'sdf' value based on the condition of being on the circle
df['sdf'] = np.sqrt(df['x']**2 + df['y']**2) - 0.5

df.loc[df['x'] == -15.0, 'u'] = 1
df.loc[df['x'] == -15.0, 'v'] = 0
df.loc[df['sdf'] <= 0.0, 'u'] = 0
df.loc[df['sdf'] <= 0.0, 'v'] = 0

import torch

mask_inlet = df['x'] == -15.0
mask_sdf = df['sdf'] <= 0.0
mask_top = df['y'] == 6.0
mask_bot = df['y'] == -6.0

from PINN_Raissi import PINN_Raissi

# Train the model
model = PINN_Raissi(df_train, mask_inlet, mask_sdf, mask_top, mask_bot)
    
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

print(f"Started Training.")
model.train(501)
print(f"Finished Training.")

from plot import plot_predictions_vs_test

# Prediction u_pred, v_pred, p_pred, nut_pred
u_pred, v_pred, p_pred = model.predict(df_train)

# Plotting
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), u_pred, df_train['u'], 'u', 'PINN_Raissi_FourierF')
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), v_pred, df_train['v'], 'v', 'PINN_Raissi_FourierF')
plot_predictions_vs_test(df_train['x'].astype(float).values.flatten(), df_train['y'].astype(float).values.flatten(), p_pred, df_train['p'], 'p', 'PINN_Raissi_FourierF')
