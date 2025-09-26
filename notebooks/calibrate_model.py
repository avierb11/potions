"""
This script provides an example of how to calibrate a "Potions" hydrologic model
using both optimization and Markov-Chain Monte-Carlo (MCMC) methods.
"""

from multiprocessing import Pool
import warnings
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, Series
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Callable
import emcee

import potions as pt

warnings.filterwarnings("ignore")

# --- 1. Load and Prepare Data ---

print("Loading data...")
data_path: str = "/home/andrew/Documents/Research/Projects/PyPotions/input/Sleepers_Results.txt"
in_df: DataFrame = pd.read_csv(data_path, sep="\\s+", index_col="Date", parse_dates=True)

q_obs: Series = in_df["Qobs"]
dates: Series = in_df.index  # type: ignore

# Prepare forcing data object for the model
forc: pt.ForcingData = pt.ForcingData(
    precip=in_df.Precipitation, temp=in_df.Temperature, pet=in_df.PET
)

# --- 2. Define Model Creation and Objective Functions ---


# Define parameter names to make indexing and splitting clear.
# This makes the script easier to modify if the model structure changes.
MODEL_PARAM_NAMES = ['tt', 'fmax', 'fc', 'lp', 'beta', 'k0', 'thr', 'k', 'alpha']
INIT_STATE_NAMES = ['s_snow_init', 's_soil_init', 's_ground_init']

NUM_MODEL_PARAMS = len(MODEL_PARAM_NAMES)
def create_model(params: NDArray) -> pt.Model:
    """Creates a model instance from a parameter array."""
    tt, fmax, fc, lp, beta, k0, thr, k, alpha = params

    snow_zone = pt.SnowZone(tt, fmax)
    soil_zone = pt.SoilZone(tt, fc, lp, beta, k0, thr)
    ground_zone = pt.GroundZone(k, alpha, 0.0)

    hs = pt.Hillslope([
        pt.Layer([snow_zone]),
        pt.Layer([soil_zone]),
        pt.Layer([ground_zone])
    ])

    # Use the generic Model class
    model = pt.Model([hs], scales=[[1.0]])
    return model

def run_model(params: NDArray) -> DataFrame:
    """Runs the model for a given set of parameters."""
    # Split the parameter vector into model parameters and initial states
    model_params = params[:NUM_MODEL_PARAMS]
    init_state = params[NUM_MODEL_PARAMS:]

    model = create_model(model_params)
    return pt.run_hydro_model(
        model=model,
        init_state=init_state,
        forc=[forc],
        dates=dates,
        dt=1.0
    )

def get_streamflow(df: DataFrame) -> Series:
    """Calculates total streamflow from the model output DataFrame."""
    # Sums the lateral flux from all zones, assuming this represents streamflow
    return df["q_lat_snow_0"] + df["q_lat_soil_1"] + df["q_lat_ground_2"]

def nse(q_sim: Series) -> float:
    """Calculates Nash-Sutcliffe Efficiency."""
    num = ((q_sim - q_obs) ** 2).sum()
    den = ((q_obs - q_obs.mean()) ** 2).sum()
    return 1.0 - num / den

def obj_func(params: NDArray) -> float:
    """The objective function to be minimized."""
    res = run_model(params)
    q_sim = get_streamflow(res)
    nse_val = nse(q_sim)
    # We want to maximize NSE, so we minimize -(NSE)
    # Adding 1.0 makes the ideal value 0, which is good for some optimizers.
    return -(nse_val - 1.0)

# --- 3. Define MCMC Probability Functions ---

def log_prior(params: NDArray, bounds: list[tuple[float, float]]) -> float:
    """Log prior probability. Returns 0 if params are in bounds, -inf otherwise."""
    for i, p_i in enumerate(params):
        if not (bounds[i][0] <= p_i <= bounds[i][1]):
            return -np.inf
    return 0.0

def log_likelihood(params: NDArray) -> tuple[float, float]:
    """Log likelihood function, which also returns the NSE value."""
    obj_val = obj_func(params)
    # We can recover the NSE value from the objective function value
    nse_val = 1.0 - obj_val
    # Reject parameter sets that are physically unrealistic or result in very poor fits
    if np.isnan(obj_val) or np.isinf(obj_val) or obj_val > 0.5:
        return -np.inf, -np.inf
    return -0.5 * (obj_val ** 2), nse_val

def log_probability(params: NDArray, bounds: list[tuple[float, float]]) -> tuple[float, float]:
    """The full log probability function, which also returns blobs."""
    lp = log_prior(params, bounds)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    ll, nse_val = log_likelihood(params)
    if not np.isfinite(ll):
        return -np.inf, -np.inf
    return lp + ll, nse_val

# --- 4. Set up and Run Optimization ---

init_guess: NDArray = np.array([
    0.0, 1.0, 100.0, 0.5, 1.0, 0.1, 50.0, 1e-3, 1.0,  # Model params
    0.0, 25.0, 50.0  # Initial states
])

bounds: list[tuple[float, float]] = [
    (-3, 3), (0.1, 5.0), (50, 1000), (0.1, 0.95), (0.5, 4.0),
    (1e-4, 0.5), (5, 150), (1e-5, 0.1), (0.25, 2.5),
    (0, 25), (0, 100), (0, 100),  # Bounds for initial states
]

print("Running optimization...")
opt_res = opt.minimize(obj_func, x0=init_guess, bounds=bounds)
print("Optimization complete.")

# --- 5. Plot Optimization Results ---

print("Plotting optimization results...")
res_optimized = run_model(opt_res.x)
q_riv_optimized = get_streamflow(res_optimized)
nse_val = nse(q_riv_optimized)

fig: Figure = plt.figure(figsize=(10, 6))
ax: Axes = fig.gca()

ax.plot(dates, q_riv_optimized, label="Simulated (Optimized)", color="Blue")
ax.plot(dates, q_obs, label="Measured", color="Black", alpha=0.5)
ax.set_ylim(0, 1.5 * q_riv_optimized.max())
ax.set_xlim(dates.min(), dates.max())
ax.set_ylabel("Streamflow (mm/day)")
ax.legend()

ax2: Axes = ax.twinx()
ax2.bar(dates, in_df.Precipitation, color="gray", alpha=0.5)
ax2.set_ylim(0, 2.5 * in_df["Precipitation"].max())
ax2.invert_yaxis()
ax2.set_ylabel("Precipitation (mm/day)")

fig.autofmt_xdate()
ax.set_title(f"Optimized NSE: {nse_val:.2f}", loc="left")
plt.savefig("optimized_hydrograph.png", dpi=300)
print("Saved optimized_hydrograph.png")

# --- 6. Set up and Run MCMC ---

ndim: int = len(init_guess)
num_walkers: int = 50
num_steps: int = 100
num_threads: int = 4

# Initialize walkers in a small ball around the optimized result
initial_pos = opt_res.x + 1e-2 * np.random.randn(num_walkers, ndim)

print("\nRunning MCMC...")
with Pool(num_threads) as pool:
    # Set up the sampler to store the NSE "blob"
    blobs_dtype = [("nse", float)]
    sampler = emcee.EnsembleSampler(
        num_walkers, ndim, log_probability, args=[bounds], pool=pool, blobs_dtype=blobs_dtype
    )
    sampler.run_mcmc(initial_pos, num_steps, progress=True)
print("MCMC complete.")

# --- 7. Analyze and Save Results ---
print("\nAnalyzing and saving MCMC results...")
flat_samples = sampler.get_chain(discard=25, thin=15, flat=True)
blobs = sampler.get_blobs(discard=25, thin=15, flat=True)
nse_samples = blobs["nse"]


# Combine parameter names for DataFrame columns
all_param_names = MODEL_PARAM_NAMES + INIT_STATE_NAMES

# Create a DataFrame from the MCMC samples and add the NSE values
results_df = pd.DataFrame(flat_samples, columns=all_param_names)
results_df["nse"] = nse_samples

# Save the combined results to a single CSV file
results_df.to_csv("mcmc_results.csv", index=False)

print(f"Saved {len(results_df)} MCMC samples to mcmc_results.csv")
print("\nFirst 5 rows of MCMC results:")
print(results_df.head())
