# Fault Detection Using HAVOK Method

## Overview
This repository contains an implementation of the **HAVOK (Hankel Alternative View of Koopman)** method for fault detection in time-series data, utilizing an Excel file (`Rg200.xlsx`) as the input dataset. The code identifies intervals of fault occurrence by analyzing a specified column of the data through polynomial regression and dynamic system modeling. The project leverages libraries such as `pysindy`, `scipy`, and `sklearn` for signal processing, system identification, and anomaly detection.

The implementation is provided in both Jupyter Notebook (`.ipynb`) and Python script (`.py`) formats, allowing for interactive exploration and batch execution, respectively.

## Project Structure
- **`RG200.ipynb`**: Jupyter Notebook containing the fault detection code with visualizations and interactive execution.
- **`main.py`**: Python script version of the notebook, suitable for command-line or automated execution.
- **`Rg200.xlsx`**: Excel file containing the time-series data for fault detection analysis.

## Detailed Description

### Datasets
- **`Rg200.xlsx`**
  - **Purpose**: Input dataset containing time-series data, with a specific column (e.g., `Column4`) used for fault detection analysis.
  - **Details**: The file includes raw data points sampled at a specified time step (`dt`), which are processed to identify fault intervals. The structure assumes a single or multiple columns, with one designated for analysis.

### Code Files

#### `RG200.ipynb`
- **Purpose**: Jupyter Notebook implementation of the fault detection pipeline using the HAVOK method.
- **Details**:
  - **Libraries**: Imports `numpy`, `pandas`, `matplotlib.pyplot`, `scipy.integrate`, `scipy.signal`, `pysindy`, `optht`, and `sklearn.ensemble.IsolationForest` for data manipulation, signal processing, system identification, and anomaly detection.
  - **Function**: Defines `HAVOK(file, colname, dt, polyorder_1, polyorder_2, skip=None)`, which:
    - Reads data from `Rg200.xlsx` into a pandas DataFrame.
    - Processes the specified column (`colname`) with a time step (`dt`) and polynomial orders (`polyorder_1`, `polyorder_2`) for regression.
    - Applies the HAVOK method to model dynamics and identify fault intervals.
    - Outputs a polynomial expression representing the system dynamics (e.g., \(x_1' = \text{polynomial terms}\)).
  - **Execution**: Calls `HAVOK('Rg200.xlsx', 'Column4', 0.00005, 1, 2)` to analyze `Column4` with a time step of 0.00005 seconds and polynomial orders 1 and 2.
  - **Output**: Generates a dynamic model equation, such as \(x_1' = 8203.214 x_4^2 - 143.938 x_4 x_5 + \ldots\), indicating fault-related behavior.
- **Usage**: Ideal for debugging, visualization (e.g., plots of signals), and step-by-step analysis.

#### `main.py`
- **Purpose**: Python script version of `RG200.ipynb`, designed for non-interactive or automated execution.
- **Details**:
  - Contains the same `HAVOK` function and logic as the notebook, converted to a `.py` format.
  - Can be run from the command line (e.g., `python main.py`) with appropriate arguments or hardcoded parameters.
  - Lacks interactive cells but retains the core functionality for batch processing.
- **Usage**: Suitable for integration into larger workflows or deployment on servers.

### Fault Detection Methodology
- **Theory**: The HAVOK method decomposes time-series data into a linear combination of Koopman modes, enabling the identification of dynamic regimes. Faults are detected as deviations in these dynamics, modeled via polynomial terms.
- **Math/Formula**:
  - Input data: \(x(t)\), sampled at \(\Delta t = dt\).
  - Hankel matrix construction: \(H = [x(t), x(t + dt), \ldots, x(t + (n-1)dt)]\).
  - SVD decomposition: \(H \approx U \Sigma V^T\), where \(U\) contains temporal modes.
  - Polynomial regression: \(x_1' = \sum_{i,j} c_{ij} x_i x_j\), with coefficients \(c_{ij}\) fitted using `pysindy`.
  - Fault intervals: Identified by anomalies in \(x_1'\) using `IsolationForest` or threshold-based methods.
- **Process**: The code skips specified rows (via `skip`), fits a model, and outputs the polynomial to highlight fault dynamics.
