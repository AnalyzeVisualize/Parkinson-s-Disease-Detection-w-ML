# Parkinson-s-Disease-Detection-w-ML

## Introduction

This project goal of this project is to be able to use voice recordings to both classify the status of and, if they were to have the disease utilizing machine learning techniques. The dataset used is sourced from the UCI Machine Learning Repository, containing various biomedical voice measurements from individuals with Parkinson's disease. 

## Data Description

### Dataset 1: 'parkinsons.data'

**Citation:**

"Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection"
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.
BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)
 
**Source:**
The dataset was created by Max Little of the University of Oxford, in collaboration with the National Centre for Voice and Speech, Denver, Colorado. The voice signals were recorded from 31 people, 23 of whom have Parkinson's disease (PD). Each row in the dataset corresponds to a voice recording from an individual, and the main goal is to discriminate between healthy individuals (status=0) and those with PD (status=1).
 
**Data Set Information:**

●	Each column represents a specific voice measure.
●	The 'name' column identifies the patient, and the 'status' column indicates the health status (0 for healthy, 1 for PD).
●	There are approximately six recordings per patient.

           For further information or comments, please contact Max Little (littlem '@'                robots.ox.ac.uk).



### Dataset 2: 'parkinsons_updrs.data'
**Citation:**
"Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests"
Tsanas A, Little MA, McSharry PE, Ramig LO.
IEEE Transactions on Biomedical Engineering (to appear).
**Source:**
The dataset was created by Athanasios Tsanas and Max Little of the University of Oxford, in collaboration with 10 medical centers in the US and Intel Corporation. The study used a telemonitoring device to record voice signals from 42 people with early-stage Parkinson's disease.
**Data Set Information:**
●	Biomedical voice measurements from 42 individuals with Parkinson's disease.
●	Columns include subject number, age, gender, time interval, motor UPDRS, total UPDRS, and 16 voice measures.
●	The goal is to predict motor and total UPDRS scores from the voice measures.
For further information or comments, please contact Athanasios Tsanas (tsanasthanasis '@' gmail.com) or Max Little (littlem '@' physics.ox.ac.uk).
# Data Processing
Once we have extracted the files from the source, we then make use of SQLite  to construct a database which we then extract the data and create data frames from.
We create tables with both of our separate datasets, as they store different values and data points. 

# Prediction Model Creation

This repository contains a Python script for predicting motor UPDRS scores in individuals with Parkinson's disease using various machine learning models. The dataset used for this analysis is stored in the 'Data' directory, and the primary script, 'predict_parkinsons.py', demonstrates the entire pipeline, including data preprocessing, model training, and evaluation.
## Dataset
For the prediction model we created we used the of 16 voice measurements from 42 people with early-stage Parkinson’s disease, which contained 5,875 recordings as well as calculated the total and motor values for the Unified Parkinson's Disease Rating Scale (UPDRS), which is what we would be predicting.
## Model Training
The dataset is split into training and testing sets using sklearn's train_test_split. Various regression models are trained, including Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
### Neural Network Model
A neural network model is built using TensorFlow and Keras, with a dense architecture. The model is trained, evaluated, and its performance is compared with other models.
### Model Evaluation
The performance of all models is evaluated based on Mean Absolute Error (MAE) and R-squared (R2) metrics
Mean Absolute Error Visualization, Per Model
 





R2 Visualization, Per Model

 

# Parkinson's Disease Classification with Machine Learning
This repository contains a classification analysis for detecting Parkinson's disease using machine learning models. The dataset used is stored in the file 'Data/parkinsons.data'. The repository includes Python code that performs the following tasks:
## 1. Data Preprocessing
●	Reads the dataset into a pandas DataFrame.
●	Removes the 'name' column from the DataFrame.
## 2. Exploratory Data Analysis
●	Visualizes the correlation matrix of selected features using a heatmap.
## 3. Model Building
We created several different models to see which would best allow us to create status classifications with the dataset. Those models were the following:

### Logistic Regression
●	Splits the data into training and testing sets.
●	Standardizes the data using the StandardScaler.
●	Creates and trains a Logistic Regression model.
●	Prints training data score, makes predictions, and prints accuracy score, confusion matrix, and classification report.
●	Visualizes the confusion matrix with a heatmap.
### Random Forest Classifier
●	Creates and trains a Random Forest Classifier.
●	Prints training and testing data scores.
●	Makes predictions and prints accuracy score, confusion matrix, and classification report.
●	Visualizes the confusion matrix with a heatmap.
### Support Vector Machine (SVM)
●	Creates and trains a Support Vector Machine (SVM) classifier.
●	Prints training data score, makes predictions, and prints accuracy score, confusion matrix, and classification report.
●	Visualizes the confusion matrix with a heatmap.
## Neural Network
●	Uses Keras Tuner to find optimal hyperparameters for a neural network model.
●	Creates and trains a neural network model using TensorFlow and Keras.
●	Prints the model summary, loss, and accuracy results.
●	Makes predictions, prints accuracy score, confusion matrix, and classification report.
●	Visualizes the confusion matrix with a heatmap.
## 4. Model Visualizations
●	Displays confusion matrix heatmaps for each model.
  

## Model Comparison
In the evaluation of various machine learning models for detecting Parkinson's disease, four different classifiers were employed: Logistic Regression, Random Forest Classifier, Support Vector Machine (SVM), and Neural Network. The summary of predictions and accuracy scores for each model is presented below:
 
 
### Other Comparisons Before Model Selection

In addition to comparing the above metrics, we also compared the classification reports for the different models. After taking these different model measurements into consideration we decided to proceed with a neural network.
## Optimizations:
In order to optimize our neural network we made use of the Keras Tuner python library, constructing a hyperband parameter tuner to facilitate the best parameters for our model. 
Optimizations regarding tuning are detailed in the following slides. Additionally, dense layers where all nodes were connected to preceding layers were found to provide highest accuracy.
Here is the final classification report for the model:
 
Optimized Sequential Model Parameters

 
## Conclusion:
The results demonstrate that the Neural Network model achieved the highest accuracy among the evaluated models, with an accuracy score of 95.92%. The Random Forest Classifier also performed well, yielding an accuracy score of 89.80%. The Support Vector Machine (SVM) and Logistic Regression models exhibited slightly lower accuracy scores of 87.76% and 85.71%, respectively.
The choice of the best-performing model depends on the specific requirements and priorities of the application. While the Neural Network showcased superior accuracy, it might involve more complexity and computational resources. Alternatively, the Random Forest Classifier offers a good balance between accuracy and interpretability.
In conclusion, this model comparison provides valuable insights into the effectiveness of different machine learning algorithms for Parkinson's disease classification. Further fine-tuning and exploration of hyperparameters could enhance the performance of the models, and additional considerations such as interpretability and resource constraints may influence the selection of the most suitable model for a given scenario.

## Contributors
- **Tim Cao**
- **Reianna Liu**
- **Alec Druggan**
- **Cindy Mateus**




## Citation


A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson.s disease progression by non-invasive 
speech tests',
IEEE Transactions on Biomedical Engineering
https://archive.ics.uci.edu/dataset/174/parkinsons
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3051371/

Note: “H&Y” refers to the Hoehn and Yahr PD stage, where higher values indicate greater level of disability.
