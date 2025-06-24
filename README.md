Neural Network from Scratch: Income Bracket Classification with PyTorch
Project Overview
This project implements a complete feedforward neural network from scratch using PyTorch to predict whether an individual's annual income exceeds $50,000, based on the UCI Adult Income Dataset. It's designed to provide a deep understanding of core neural network concepts through hands-on implementation and systematic experimentation (ablation studies).

Task Objectives Covered
This project comprehensively addresses the following objectives as outlined in the task:

Data Preprocessing: Handled missing values, applied various encoding (One-Hot, Label, Ordinal) and scaling (Standard, MinMax, Robust) techniques, and performed train/validation/test splits.

Model Building (From Scratch): Constructed a multi-layered feedforward neural network using torch.nn.Module, incorporating dense layers, activation functions (ReLU, Tanh, LeakyReLU), Dropout, and Batch Normalization.

Custom Training Loop: Implemented a bespoke training and evaluation loop, utilizing nn.BCELoss and nn.BCEWithLogitsLoss with different optimizers (Adam, SGD, RMSprop) and early stopping.

Regularization & Overfitting Control: Explored the impact of various Dropout rates and L2 regularization (Weight Decay) to mitigate overfitting.

Model Evaluation: Computed and analyzed standard classification metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC) and generated confusion matrices.

Hyperparameter Tuning: Conducted a structured grid search across key hyperparameters to identify optimal model configurations.

Bias Analysis: Evaluated model fairness by assessing performance across sensitive demographic features ('sex', 'race').

Feature Importance (Bonus): Utilized permutation-based feature importance to identify the most influential features in the model's predictions.

Project Structure
The project is organized into two main Jupyter notebooks to facilitate clarity, review, and efficient execution:

NN_Ablation_Studies.ipynb:

Purpose: This notebook contains the detailed implementation and results of all conducted ablation studies and hyperparameter tuning experiments. It demonstrates the systematic process of comparing different preprocessing techniques, model architectures, optimizers, loss functions, and regularization strategies.

Execution Time: Please note that running this notebook from start to finish is time-consuming due to the extensive number of experiments. It is intended for review of the experimental methodology and findings rather than full end-to-end execution during a quick review.

Content:

Comprehensive ablation studies with comparative plots for:

Categorical Encoding & Numerical Scaling

Network Architecture (Depth, Width, Activations, BatchNorm)

Optimizers & Loss Functions

Regularization (Dropout & Weight Decay)

Detailed Hyperparameter Grid Search results, including a summary table of top configurations.

NN_Optimal_Model.ipynb:

Purpose: This notebook presents the final, best-performing neural network model, built using the optimal hyperparameters identified during the ablation studies. It's designed for quick execution to verify the model's performance and key analyses.

Execution Time: This notebook should run relatively quickly, providing a snapshot of the optimized model's capabilities.

Content:

Implementation of the chosen optimal preprocessing pipeline.

Definition and training of the best model architecture.

Final model evaluation with metrics and visualizations (training history, confusion matrix, performance summary).

In-depth Bias Analysis across demographic groups with visual plots.

Feature Importance analysis using permutation importance with visualizations.

How to Run
Prerequisites
Python 3.8+

Jupyter Notebook or JupyterLab

Required Python libraries:

pip install pandas numpy scikit-learn torch matplotlib seaborn

Execution Steps

Clone the Repository:
```bash
git clone https://github.com/kaamchor07/Neural-Network-Income-Classifier-PyTorch.git
cd Neural-Network-Income-Classifier-PyTorch

Launch Jupyter:

jupyter notebook
# or jupyter lab

Run the Optimal Model (Recommended for Quick Review):

Open NN_Optimal_Model.ipynb.

Run all cells (Cell > Run All).

Observe the final model's performance metrics, training history, confusion matrix, bias analysis, and feature importance plots.

Explore Ablation Studies (Optional, Time-Consuming):

Open NN_Ablation_Studies.ipynb.

Run all cells (Cell > Run All).

Review the detailed comparative results and insights from the extensive experimentation.

Key Findings (High-Level)
Optimal Preprocessing: One-Hot Encoding for categorical features combined with StandardScaler for numerical features generally yielded the best performance.

Effective Architecture: A neural network with a moderate depth (e.g., 2-3 hidden layers) and sufficient width (e.g., [128, 64] neurons) using LeakyReLU activation functions proved most effective. Batch Normalization was consistently beneficial.

Efficient Optimization: The Adam optimizer with nn.BCEWithLogitsLoss demonstrated superior convergence and stability.

Robust Regularization: A balanced application of Dropout (around 0.5) and a small amount of Weight Decay significantly helped in preventing overfitting and improving generalization.

Interpretability: Key features like age, capital-gain, hours-per-week, education, and marital-status were identified as highly influential in income prediction. Bias analysis highlighted performance disparities across gender and racial groups, signaling areas for future fairness interventions.

Future Work
Missing Value Strategy Comparison: Extend ablation studies to directly compare the impact of dropping missing value rows versus various imputation strategies.

Advanced Learning Rate Schedules: Investigate and integrate dynamic learning rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingWarmRestarts) to potentially further optimize training convergence and model performance.

Local Interpretability with SHAP: Implement SHAP (SHapley Additive exPlanations) to provide in-depth, individual prediction explanations, complementing the global permutation feature importance.

Class Imbalance Handling: Explore techniques such as oversampling (SMOTE), undersampling, or class weighting in the loss function to potentially improve performance on the minority income class (>50K).

Model Deployment: Develop a simple inference API or containerize the optimal model for potential deployment.
