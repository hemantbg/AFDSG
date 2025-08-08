### Datasets
The project relies on four `.csv` files containing data for training and evaluation across different phases of the Fed-Meta-Align framework.

- **`iot1_data.csv`**
  - **Purpose**: Contains fault classification data specific to the IoT1 device, used for local training, Phase 1 meta-initialization, Phase 2 federated aggregation, and Phase 3 personalization.
  - **Details**: Includes features and labels reflecting IoT1’s heterogeneous data distribution.

- **`iot2_data.csv`**
  - **Purpose**: Contains fault classification data specific to the IoT2 device, used similarly across all phases.
  - **Details**: Reflects IoT2’s unique data characteristics, contributing to non-IID challenges.

- **`iot3_data.csv`**
  - **Purpose**: Contains fault classification data specific to the IoT3 device, used in the 3-device novel architecture.
  - **Details**: Adds further heterogeneity, enabling scalability testing.

- **`server_phase0_data.csv`**
  - **Purpose**: Dataset used in Phase 0 for foundational pre-training of the base model on a public or generalized dataset.
  - **Details**: Likely sourced from a public dataset (e.g., AI4I 2020), providing a starting point for the framework.
