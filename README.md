# Graph Classification with Causal Inference and GraphSAGE

## Overview
This repository contains an implementation of a novel graph-based classification framework that integrates **causal inference**, **GraphSAGE**, and **explainability tools** (SHAP, GNNExplainer, Integrated Gradients) for enhanced fault detection in electrical distribution systems (EDS). The approach transforms time-series data into causal graphs using transfer entropy (TE), employs a GraphSAGE model for multiclass classification, and provides interpretability through advanced explainability techniques. The project is rooted in research presented in two accompanying papers and is implemented using a Jupyter Notebook with a Python script version.

The code leverages libraries such as `torch`, `networkx`, `pywt`, and `sklearn`, and is designed to handle high-dimensional datasets with improved accuracy and transparency.

## Project Structure
- **`GraphClassification_multiclass.ipynb`**: Jupyter Notebook containing the main Graph Neural Network (GNN) code for multiclass graph classification.
- **`main.py`**: Python script version of the notebook, suitable for automated execution.
- **`graphsage_shap_paper.pdf`**: Research paper detailing the integration of causal graphs, GraphSAGE, and SHAP for fault classification (9 pages).
- **`Integrating_Causal_Graphs_and_GraphSAGE_for_Enhanced_Classification_and_Interpretability_Using_SHAP.pdf`**: Research paper exploring causal inference with GraphSAGE and explainability tools like GNNExplainer and Integrated Gradients (8 pages).
- **Datasets**: Assumed to be time-series data (e.g., voltage, current) from EDS, processed into graph structures (not explicitly provided but referenced in the code).

## Detailed Description

### Datasets
- **Input Data**: Time-series data (e.g., from EDS fault datasets or public datasets like Ionosphere) stored in a format compatible with pandas DataFrames.
  - **Purpose**: Used to construct causal graphs via transfer entropy and fed into the GraphSAGE model for classification.
  - **Details**: Includes features such as voltage and current, with labels for multiple fault classes (0-8, based on the code’s output counting).

### Code Files

#### `GraphClassification_multiclass.ipynb`
- **Purpose**: Main implementation of the multiclass graph classification pipeline using GraphSAGE.
- **Details**:
  - **Libraries**: Imports `pandas`, `numpy`, `pywt`, `torch`, `networkx`, `pyinform`, `sklearn.metrics`, `matplotlib.pyplot`, and `sklearn.neural_network.MLPClassifier`.
  - **Functions**:
    - `dim_red(data, factor)`: Performs dimensionality reduction using Fourier transform, retaining 25% of frequencies to preprocess time-series data.
    - Graph construction: Uses transfer entropy (`pyinform`) to build causal graphs, where nodes represent features and edges indicate causal relationships.
    - `model`: A GraphSAGE-based neural network (`torch.nn.Module`) that processes node features and edge indices for classification.
  - **Process**:
    - Loads and preprocesses data with `StandardScaler`.
    - Constructs graphs from time-series data using transfer entropy.
    - Trains the GraphSAGE model on graph structures.
    - Evaluates predictions with `argmax` and computes class distribution (e.g., `count_ans` for classes 0-8).
  - **Output**: Predicted class labels for 900 graphs, with an example accuracy calculation (e.g., `count_ans[0]/100`).
- **Usage**: Ideal for interactive experimentation, visualization (e.g., plots), and model tuning.

#### `main.py`
- **Purpose**: Python script version of `GraphClassification_multiclass.ipynb`, designed for non-interactive or automated execution.
- **Details**: Contains the same logic as the notebook, converted to a `.py` format, runnable via `python main.py`.
- **Usage**: Suitable for deployment or integration into larger workflows.

### Research Papers

#### `graphsage_shap_paper.pdf`
- **Purpose**: Details the integration of causal graphs, GraphSAGE, and SHAP for fault classification.
- **Key Contributions**:
  - Proposes using transfer entropy to construct causal graphs representing EDS variable relationships.
  - Employs GraphSAGE for classification, achieving 92.6% accuracy on the Ionosphere dataset and 94.7% on the EDS fault dataset.
  - Uses SHAP for interpretability, identifying key variables influencing predictions.
- **Authors**: Karthik Peddi, Mayukha Pal.
- **Details**: Highlights the importance of causal analysis for transparency in fault detection.

#### `Integrating_Causal_Graphs_and_GraphSAGE_for_Enhanced_Classification_and_Interpretability_Using_SHAP.pdf`
- **Purpose**: Explores causal inference with GraphSAGE and explainability tools.
- **Key Contributions**:
  - Constructs causal graphs using transfer entropy, with nodes as features (e.g., voltage, current) and edges as causal influences.
  - Achieves 99.44% accuracy on the EDS fault dataset using GraphSAGE.
  - Employs GNNExplainer and Captum’s Integrated Gradients to highlight influential nodes.
- **Authors**: Karthik Peddi, Sai Ram Aditya Parisineni, Hemanth Macharla, Mayukha Pal.
- **Details**: Emphasizes practical applicability for system reliability improvement.

### Methodology
- **Theory**: Combines causal inference (transfer entropy) with graph neural networks (GraphSAGE) to model temporal relationships and classify faults. Explainability tools (SHAP, GNNExplainer, Integrated Gradients) provide insights into causative factors.
- **Math/Formula**:
  - Transfer Entropy: \(TE_{X \to Y} = \sum p(y_{t+1}, y_t, x_t) \log \frac{p(y_{t+1} | y_t, x_t)}{p(y_{t+1} | y_t)}\).
  - GraphSAGE Update: \(h_v^{(k)} = \text{AGGREGATE}(\{h_u^{(k-1)}, \forall u \in \mathcal{N}(v)\})\), where \(h_v^{(k)}\) is the node embedding.
  - SHAP Value: \(\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_S(x) - f_{S \setminus \{i\}}(x)]\).
- **Process**: Preprocess data, build causal graphs, train GraphSAGE, and analyze predictions with explainability tools.
