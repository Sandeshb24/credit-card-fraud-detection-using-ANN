# Credit Card Fraud Detection using Artificial Neural Networks

## Project Overview
* This project focuses on building an Artificial Neural Network (ANN) Model to detect fraudulent credit card transactions.
* Credit card fraud detection is a critical task in financial industry, but it presents a significant challenge due to the highly imbalanced nature of real-world transaction datasets.
* Fraudulent transactions are extremely rare compared to legitimate ones, which can lead to models that perform poorly on the minority (fraudulent) class.
* This notebook addresses the class imbalance problem using the __Synthetic Minority Over-sampling Technique (SMOTE)__ to improve the model's ability to identify fraud.

## The Challenge: Class Imbalance
In typical credit card transaction datasets, the number of legitimate transactionsvastly outnumbers fraudulent ones `(eg. 99.9% legitimate and 0.1% fraudulent)`. If a machine learning model is trained directly on such imbalanced data, it tends to become biased towrds the majority class.
This often results in:
* __High overall accuracy:__ The model can achieve high accuracy simply by predicting "legitimate" for almost all transactions.
* __Poor Recall for the minority class:__ It fails to identify most of the actual fraudulent transactions, which is the most critical aspect of fraud detection.

To counteract this, __SMOTE__ is employed to synthesize new, realistic samples for the minority class, thereby balancing the dataset for effective model training.

## Dataset
* The dataset used in this project can be found here: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
* __Features:__ The dataset contains numerical features `V1` through `V28`, which are the result of a PCA transformation to protect user identities and sensitive information. It also includes an `Amount` feature representing the transaction amount.
* __Target Variable:__ The `Class` column indicates whether a transaction is __fraudulent (1)__ or __legitimate (0)__.
* __Imbalance:__ This dataset is highly imbalanced, with a very small percentage of fraudulent transactions.

__Note on Results:__ The exceptionally high performance metrics (Precision, Recall, F1-Score, AUC-ROC, Average Precision) achieved in this project are likely due to the specific characteristics of this creditcard_2023.csv dataset.
 It appears to have highly separable classes, possibly due to its synthetic nature or extensive pre-processing for benchmarking purposes. Real-world credit card fraud detection is typically much more challenging.
 
## Methodology
The project follows these key steps:

__Data Loading and Initial Exploration:__

* Loads the creditcard_2023.csv dataset using Pandas.

* Briefly checks for null values and data types.

__Data Preparation:__

* Separates features (X) from the target variable (y).

* Splits the data into training and testing sets using train_test_split with stratify=y to maintain the original class distribution in both sets.

__Feature Scaling:__

* Applies StandardScaler to normalize the numerical features. The scaler is fitted only on the training data and then used to transform both training and testing sets to prevent data leakage.

* The fitted StandardScaler object is saved (scaler.joblib) for consistent preprocessing of new, unseen data during deployment.

__Handling Class Imbalance with SMOTE:__

* `SMOTE` (Synthetic Minority Over-sampling Technique) is applied only to the training data. This generates synthetic samples for the minority class (fraudulent transactions), balancing the class distribution for the ANN training.

## Artificial Neural Network (ANN) Model Building:

* A Sequential Keras model is constructed.

* It includes multiple Dense (fully connected) hidden layers with ReLU activation functions.

* Dropout layers are incorporated after hidden layers to prevent overfitting by randomly setting a fraction of input units to zero during training.

* The output layer uses a single unit with a sigmoid activation function for binary classification.

* The model is compiled with the adam optimizer and binary_crossentropy loss, with accuracy as a monitored metric.


## Model Training:

* The ANN is trained on the SMOTE-resampled training data.

* EarlyStopping callback is used to monitor val_loss and stop training if the validation loss does not improve, further mitigating overfitting.

## Model Evaluation:

The trained model is evaluated on the original, unseen test set.

Key metrics for imbalanced classification are reported:
* __Confusion Matrix:__ Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

* __Precision:__ The proportion of correctly identified positive predictions.

* __Recall (Sensitivity):__ The proportion of actual positive cases that were correctly identified. (Crucial for fraud detection).

* __F1-Score:__ The harmonic mean of Precision and Recall, providing a balance between the two.

* __AUC-ROC (Area Under the Receiver Operating Characteristic Curve):__ Measures the model's ability to discriminate between classes.

* __Average Precision (PR-AUC):__ Measures the area under the Precision-Recall curve, often more informative than AUC-ROC for highly imbalanced datasets.

* The trained Keras model is saved (`my_fraud_detection_model.keras`) for future use and deployment.

## Visualization:

Plots are generated to visualize the training and validation accuracy and loss over epochs, helping to monitor training progress and identify potential overfitting.

__Project Files__
`credit-feraud-detection.ipynb`: The main Jupyter Notebook containing all the code for data preprocessing, model training, and evaluation.

`my_fraud_detection_model.keras`: The saved Keras model after training.

`scaler.joblib`: The saved StandardScaler object used for feature scaling.

`requirements.txt`: A list of all necessary Python libraries and their versions.

## Setup and Installation
To run this project locally, follow these steps:

__Clone the repository:__

`git clone <your-repository-url>`
`cd <your-repository-name>`

__Create a virtual environment (recommended):_

`python -m venv venv`
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

__Install the required libraries:__

`pip install -r requirements.txt`

Download the dataset: Place creditcard_2023.csv in the same directory as your notebook, or update the path in the notebook.

Ensure saved model and scaler are present: Make sure my_fraud_detection_model.keras and scaler.joblib are in the same directory as your notebook (or adjust paths if you place them elsewhere).

## Usage
Open the Jupyter Notebook:

`jupyter notebook credit-feraud-detection.ipynb`

Alternatively, upload and open it in Google Colab.

__Run all cells:__ Execute the cells sequentially to preprocess data, train the model, and view the evaluation metrics.

## Future Enhancements
* __Hyperparameter Tuning:__ Systematically tune ANN hyperparameters (number of layers, neurons per layer, dropout rates, learning rate) using techniques like GridSearchCV or RandomizedSearchCV.

* __Ensemble Methods:__ Experiment with ensemble models (e.g., Random Forest, Gradient Boosting) or stacking multiple models for potentially better performance.

* __Other Resampling Techniques:__ Explore advanced SMOTE variants (ADASYN, Borderline-SMOTE) or undersampling techniques (Tomek Links, Edited Nearest Neighbors).

* __Anomaly Detection:__ Consider framing the problem as an anomaly detection task using models like Isolation Forest or Autoencoders, especially if the fraud patterns are highly diverse.

* __Real-time Inference:__ Develop a simple API or web application (e.g., using Flask or Streamlit) to deploy the model for real-time fraud prediction.

# Accuracy scores visualisation and their metrics:

__Accuracy plot for Train and Test__:


![model_accuracy](https://github.com/user-attachments/assets/41d185a0-099e-4d09-bf29-2862e3068213)



__Loss Funtion plot for Train and Test__:


![model_loss](https://github.com/user-attachments/assets/7b745394-c8a7-44e0-aec4-0d9f356fe7d4)


`Precision: 0.9971`

`Recall (Sensitivity): 0.9967`

`F1-Score: 0.9969`

`AUC-ROC: 0.989`

`Average Precision (PR-AUC): 0.9997`
