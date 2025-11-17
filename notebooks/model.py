from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
from potions.model import Layer, Hillslope, Model, ForcingData, run_hydro_model
from potions.hydro import SoilZone, GroundZone

# 1. Define the model structure using the new API

# Create instances of the concrete hydrologic zones with their parameters
soil = SoilZone(tt=0, fc=200, lp=0.5, beta=2, k0=0.1, thr=150)
ground = GroundZone(k=0.01, alpha=2, perc=0.1)

# Assemble zones into layers
soil_layer = Layer(soil)
ground_layer = Layer(ground)

# Assemble layers into a hillslope
hillslope = Hillslope(layers=[soil_layer, ground_layer])

# Create the main model object.
# The `scales` argument defines the relative area of each hillslope and
# each forcing source within that hillslope. Here, we have one hillslope
# that covers 100% of the area, driven by one forcing source.
model = Model(layers=[hillslope], scales=[[1.0]])

# 2. Prepare forcing data and initial conditions

# Create a time series of forcing data
num_days = 365
dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=num_days, freq="D"))

precip = pd.Series(np.random.rand(num_days) * 10, index=dates)
temp = pd.Series(np.sin(np.arange(num_days) / 365 * 2 * np.pi) * 15 + 10, index=dates)
pet = pd.Series(np.random.rand(num_days) * 5, index=dates)

# The `run_hydro_model` function expects a list of ForcingData objects,
# one for each forcing source defined in the model's scales.
forcing_data = [ForcingData(precip=precip, temp=temp, pet=pet)]

# Set the initial state (storage) for each zone in the model
# The order matches the flattened model: [soil, ground]
initial_state = np.array([50.0, 200.0])

# 3. Run the simulation

print("Running hydrologic simulation...")
results_df = run_hydro_model(
    model=model,
    init_state=initial_state,
    forc=forcing_data,
    dates=dates,
    dt=1.0,  # Daily time step
)

# 4. Display results

print("Simulation complete.")
print("Output DataFrame shape:", results_df.shape)
print("Output columns:", results_df.columns.tolist())
print("\nFirst 5 rows of results:")
print(results_df.head())
