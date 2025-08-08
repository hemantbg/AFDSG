# Fed-Meta-Align: Federated Learning for TinyML Fault Classification

## Overview
This repository contains the implementation of **Fed-Meta-Align**, a novel four-phase federated learning framework designed for robust and personalized fault classification on heterogeneous TinyML devices. The project addresses the challenges of non-IID (non-independently and identically distributed) data in IoT environments, outperforming traditional baselines like FedAvg and FedProx. The code is structured to support baseline comparisons, the main novel architecture with 2 and 3 devices, and additional analysis for accuracy plots and timing comparisons.

The project is developed using Python, with Jupyter Notebooks (`.ipynb` files) for experimentation and visualization. The structure includes folders for baseline methods and individual files for the novel architecture and analysis.

## Project Structure
- **`local/`**: Contains code for local model training on individual devices (IoT1, IoT2, IoT3) as baseline comparisons.
- **`fedavg/`**: Contains two `.ipynb` files implementing the FedAvg algorithm, with and without personalization.
- **`fedprox/`**: Contains two `.ipynb` files implementing the FedProx algorithm, with and without personalization.
- **`fault_2_devices.ipynb`**: Main implementation of the Fed-Meta-Align novel architecture with 2 devices.
- **`final_novel_3_devices_architecture.ipynb`**: Extended implementation of the Fed-Meta-Align novel architecture with 3 devices.
- **`final2_with_acc_plots.ipynb`**: Code for generating accuracy plots after each phase for each device.
- **`time.ipynb`**: Code for calculating prediction time comparisons between normal NN models and TFLite models.

## Detailed Description

### Baseline Comparisons
The following folders and files provide implementations of baseline methods for comparison with the Fed-Meta-Align framework.

#### `local/`
- **Purpose**: Contains code for training local models on individual devices (IoT1, IoT2, IoT3) without federated learning, serving as a baseline to evaluate the benefit of collaboration.
- **Files**:
  - `iot1.ipynb`: Local model training code specific to IoT1 device data.
  - `iot2.ipynb`: Local model training code specific to IoT2 device data.
  - `iot3.ipynb`: Local model training code specific to IoT3 device data.
  - `server.ipynb`: Code for a central server managing local model evaluations (if applicable).
- **Details**: These notebooks train models independently on each device's data, ignoring heterogeneity, to establish a performance baseline.

#### `fedavg/`
- **Purpose**: Implements the FedAvg (Federated Averaging) algorithm, a standard federated learning baseline, with two variations.
- **Files**:
  - `fedavg_without_personalization.ipynb`: Implements FedAvg with simple averaging of client model updates (\(\phi^{r+1} = \frac{1}{|C_r|} \sum_{t \in C_r} \phi_t^r\)) without personalization.
  - `fedavg_with_personalization.ipynb`: Extends FedAvg with an on-device personalization phase, fine-tuning the global model for each client.
- **Details**: The without-personalization version highlights client drift issues, while the with-personalization version adapts the model to local data distributions.

#### `fedprox/`
- **Purpose**: Implements the FedProx algorithm, an enhanced federated learning baseline with a proximal term, with two variations.
- **Files**:
  - `fedprox_without_personalization.ipynb`: Implements FedProx with a regularization term (\(\mathcal{L}_t = \mathcal{L}(\phi_t) + \frac{\mu}{2} \|\phi_t - \phi^r\|^2\)) without personalization.
  - `fedprox_with_personalization.ipynb`: Extends FedProx with personalization, fine-tuning the global model for each client.
- **Details**: The proximal term mitigates drift, and personalization further improves performance on heterogeneous data.

### Novel Architecture
The following files implement the core Fed-Meta-Align framework, introducing the four-phase pipeline.

#### `fault_2_devices.ipynb`
- **Purpose**: Main implementation of the Fed-Meta-Align novel architecture with 2 devices (e.g., IoT1 and IoT2).
- **Details**: Covers all four phases:
  - **Phase 0**: Foundational pre-training on a public dataset.
  - **Phase 1**: Serial meta-initialization to create a heterogeneity-aware model.
  - **Phase 2**: Similarity-aware federated aggregation using Adam updates and cosine similarity weighting.
  - **Phase 3**: On-device personalization with fine-tuning and quantization.
- **Key Equations**:
  - Meta-initialization update: \(w \leftarrow \text{Train}(w, S_{t,\text{P1}})\).
  - Aggregation update: \(\phi^{r+1} = \phi^r + \alpha \sum_{t \in C_r} \hat{w}_t \Delta_t^r\), where \(\hat{w}_t = \frac{s_t \times \max(c, \theta_t)}{\sum_{j \in C_r} s_j \times \max(c, \theta_j)}\).
- **Output**: Personalized TFLite models for 2 devices.

#### `final_novel_3_devices_architecture.ipynb`
- **Purpose**: Extended implementation of Fed-Meta-Align with 3 devices (e.g., IoT1, IoT2, IoT3).
- **Details**: Similar to `fault_2_devices.ipynb` but scaled to handle 3 devices, demonstrating scalability of the framework.
- **Key Equations**: Same as above, with adjustments for 3-client aggregation and personalization.
- **Output**: Personalized TFLite models for 3 devices.

### Analysis and Visualization
The following files support performance evaluation and visualization.

#### `final2_with_acc_plots.ipynb`
- **Purpose**: Generates accuracy plots after each phase (Serial Meta-Initialization, Parallel FL, On-Device Personalization) for each device.
- **Details**: Uses matplotlib or seaborn to visualize performance evolution, comparing IoT1, IoT2, and IoT3 accuracies (e.g., from `fig:stage_improvement` in the paper).
- **Key Output**: Plots showing stepwise accuracy improvements, validating the framework’s synergy.

#### `time.ipynb`
- **Purpose**: Calculates prediction time for normal neural network (NN) models versus TFLite models.
- **Details**: Measures inference latency (e.g., 242.00 ms for NN vs. 164.00 ms for TFLite on IoT1) to assess deployment efficiency.
- **Key Output**: Time comparison table or graph, highlighting the 1.5x-3.6x speedup from quantization.



For questions or collaborations, contact [Your Name/Email].

---
