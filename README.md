
# Chronic Heart Disease Prediction Model

## Project Overview

This project develops an **offline** (batch) machine-learning model to predict chronic heart disease using Keras (TensorFlow) and NumPy. Given patient health metrics, the model classifies whether a person has heart disease (binary outcome).  Heart disease is a leading cause of death globally, so such tools can assist in early risk screening. The codebase includes end-to-end pipelines (in Google collab notebooks) for data preparation, model training, and evaluation. Key points:

* **Purpose:** Classify heart disease presence/absence from clinical features.
* **Technology:** Python, Keras (TensorFlow), NumPy; uses a feed-forward neural network with Dense layers.
* **Scope:** Offline analysis only (not a real-time or clinical deployment).
* **Intended Use:** Research and educational demonstration; **not** production- or medically-certified.

For transparency and reproducibility, the repository provides two notebooks: one for **training** (`heart_disease_prediction_training.ipynb`) and one for **testing/evaluation** (`heart_model_testing.ipynb`).

## Dataset Description

We use a well-known heart disease dataset (e.g. the UCI Cleveland Heart Disease dataset). It contains several hundred patient records (≈300) with clinical measurements and a **target label** indicating disease presence. Each record includes features such as:

* **`age`**: patient age in years.
* **`sex`**: gender (1=male, 0=female).
* **`cp`**: chest pain type (categorical, e.g. typical angina).
* **`trestbps`**: resting blood pressure (mm Hg on admission).
* **`chol`**: serum cholesterol (mg/dl).
* **`fbs`**: fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
* **`restecg`**: resting electrocardiogram results (categorical).
* **`thalach`**: maximum heart rate achieved.
* **`exang`**: exercise-induced angina (1 = yes; 0 = no).
* **`oldpeak`**: ST depression induced by exercise relative to rest.

These features correspond to known risk factors (e.g. high blood pressure, high cholesterol) for heart disease. The original dataset has about 14 attributes, but we typically use a subset of \~10 relevant features. The **target label** (“heart disease”) is encoded 0 (no disease) or 1 (disease present) by combining the original multi-valued score into binary form. In practice, the dataset is roughly balanced between positive and negative cases, which helps the model learn equally from both classes.

## Model Architecture

The prediction model is an **artificial neural network (ANN)** built with Keras’ Sequential API. It consists of fully-connected (Dense) layers. For example, a prototypical architecture used in this project has:

* **Input layer:** Dimension = number of input features (e.g. 9–10).
* **Hidden Layer 1:** Dense layer with 5 neurons, ReLU activation.
* **Output Layer:** Dense layer with 1 neuron and sigmoid activation (for binary classification).


## Training Process

Data preprocessing and model training occur in `heart_disease_prediction_training.ipynb`. Key steps include:

* **Data Preprocessing:**  Clean the data (handle missing values if any), and apply feature scaling (e.g. standardization) so that numeric features like blood pressure and cholesterol have zero mean and unit variance. Categorical features (such as `cp` or `restecg`) are encoded (one-hot or label encoding) to numerical form.

* **Train/Test Split:** Partition the data into training and test sets (commonly 70/30 or 80/20 split). The training set is used to fit the model, while the test set is held out for final evaluation. Because the dataset is relatively small, cross-validation can also be performed (though our notebooks use a single split for simplicity).

* **Model Compilation:** Configure the neural network with optimizer, loss, and metrics. For example:

  ```python
  model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy']
  )
  ```

  This uses the Adam optimizer and **binary\_crossentropy** loss (appropriate for binary classification). Accuracy is tracked as the primary metric.

* **Training:** Run `model.fit()` on the training data for a fixed number of epochs (e.g. 100) with a suitable batch size (e.g. 32). Training iteratively updates the network weights to minimize the loss. The notebooks display training progress and loss/accuracy curves.

* **Evaluation:** After training, the model is evaluated on the test set. For example:

  ```python
  loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
  print(f"Test Accuracy: {accuracy:.2%}")
  ```

  This computes the final performance metrics (loss and accuracy) on unseen data. The example run in the reference got \~90.10% test accuracy.

All of the above is fully contained in the training notebook, along with any data augmentation or feature engineering experiments. The final trained model weights are then used in the testing notebook for further analysis.



## Key Results and Insights

* **Model Performance:** The trained ANN achieved about **80% accuracy** on unseen test data. The confusion matrix and classification report (produced in the testing notebook) show balanced performance on both classes. For example runs, precision and recall are also high, indicating the model is reliably identifying heart disease cases without too many false alarms. These results are comparable to baseline studies

* **Feature Insights:** Although this simple ANN does not produce explicit feature importance scores, the dataset features we used (age, chest pain, blood pressure, cholesterol, etc.) align with known clinical risk factors. For instance, patients with high blood pressure and high cholesterol (key risk factors) in the dataset tended to have heart disease. Exploratory plots in the notebooks confirm that distributions of these features differ between disease/no-disease groups. This consistency with medical knowledge provides confidence in the model’s reasoning.

* **Generalizability:** The dataset is relatively small, so performance can vary with different random splits. The notebooks show moderate variance between runs. Still, the model generalizes well in our experiments. For production use, one would want to gather more data, perform cross-validation, and possibly integrate domain features (e.g. imaging or genetic data) for a more robust predictor.

* **Practical Use:** The model and code demonstrate that even a straightforward neural network can effectively predict heart disease from clinical data. It can be used as a **tool for offline analysis**—for example, a clinician or researcher could input new patient data (in the same feature format) into the model to get a quick risk assessment. However, recall the limitations below.

## Disclaimer

* **Not for Clinical Use:** This model is **for research and personal use only**. It has *not* been validated for medical diagnosis or treatment decisions. Do **not** use it as a substitute for professional medical advice.
* **Experimental Project:** The code and results are intended for learning and demonstration. The model has not been stress-tested or certified in any production environment.
* **Accuracy May Vary:** Actual performance depends on data quality and population differences. The reported metrics are from a specific dataset and split; real-world accuracy could be lower. Users should be cautious and consult domain experts before applying this model.

Please review all results critically, and understand that this is **not** a production-grade system. Its purpose is to illustrate data science techniques in the context of heart disease prediction.

