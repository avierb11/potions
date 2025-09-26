# Potions
A flexible, component-based hydrologic modeling framework in Python.

## 1. Project Summary

Potions is a Python library designed for simulating catchment-scale hydrologic and biogeochemical processes using a lumped approach. It allows users to construct watershed models by assembling fundamental units like. The framework is built to be flexible, enabling the exploration of different model structures and parameterizations with ease to test different conceptual models for critical zone functioning.

## 2. Core Problem

Traditional hydrologic models can be monolithic and difficult to adapt. Modifying their structure to test new hypotheses about water movement (e.g., adding a new soil layer or representing a riparian zone differently) can be a significant software engineering challenge. Potions aims to solve this by providing a "Lego-like" system where model structures are not hard-coded but are composed dynamically by the user.

## 3. Key Features

- **Component-Based Structure:** Build models from basic components:
    - `HydrologicZone`: The fundamental computational unit.
    - `Layer`: A horizontal collection of zones.
    - `Hillslope`: A vertical stack of layers.
    - `HydrologicModel`: A collection of hillslopes representing a full catchment.
- **Flexible Connectivity:** The model automatically computes lateral and vertical connectivity matrices (`lat_mat`, `vert_mat`) based on the user-defined structure.
- **Forcing Data Management:** A clear system for applying forcing data (precipitation, temperature, PET) to different parts of the model using scaling matrices.
- **NumPy-based Backend:** The core simulation loop is built on NumPy for efficient numerical computation.
- **Clear Data I/O:** Uses Pandas DataFrames for model results, ensuring easy analysis and plotting.

## 4. High-Level Architecture

The model is conceptualized as a directed acyclic graph (DAG) of `HydrologicZone`s.
1.  A user defines one or more `Hillslope`s, each composed of `Layer`s.
2.  The `HydrologicModel` class takes these hillslopes and "flattens" them into a 1D array of zones for sequential processing.
3.  It pre-computes connectivity matrices to define fluxes between zones.
4.  The `run_hydro_model` function iterates through time, applying forcing data and calling the `model.step()` method, which calculates the state and fluxes for every zone at each timestep.

## 5. Development Outline & Roadmap

-   **v0.1 (Core Engine):**
    -   [x] Implement core data structures (`HydrologicZone`, `Layer`, `Hillslope`, `HydrologicModel`).
    -   [x] Implement core model stepping logic and flux calculations.
    -   [ ] Add comprehensive unit tests for `get_vert_mat` and `get_lat_mat`.
    -   [ ] Refine the `HydrologicZone` interface.
-   **v0.2 (Usability & Features):**
    -   [ ] Develop a simple command-line interface (CLI) for running models from a configuration file.
    -   [ ] Implement more `HydrologicZone` types (e.g., different soil models, snow component).
    -   [ ] Add visualization utilities for plotting model structure and results.
-   **v0.3 (Performance & Optimization):**
    -   [ ] Profile the `run_hydro_model` loop for bottlenecks.
    -   [ ] Investigate parallelization options (e.g., per-hillslope).