# **IDENTIFYING RENAL DISEASE PATIENT PROFILES USING UNSUPERVISED CLUSTERING OF LABORATORY TEST DATA**

Author: Hesler Bustos Chavez   
Project Context: Unsupervised Clustering for Healthcare Data Analysis

## Overview

This project presents a patient segmentation analysis of individuals diagnosed with chronic kidney disease (CKD) who received outpatient care within the Peruvian Social Health Insurance system (EsSalud) between 2020 and 2024. To achieve this, unsupervised machine learning techniques such as K-means, hierarchical clustering, and density-based clustering (DBSCAN) are applied. The use of these methods enables the identification of patterns and patient profiles within the data, which may contribute to a better understanding of renal disease.

## Objectives
* To identify patient profiles associated with chronic kidney disease using unsupervised machine learning techniques applied to laboratory test data.

* To evaluate the usefulness of clustering techniques for discovering hidden structures in healthcare data related to renal disease.

## 📊 Data Description
This study focuses on data whose samples were collected in 2024. The data were obtained from the open data portal of the Gobierno del Perú available at Peru's National Open Data Platform.

The dataset consists of 26 columns, which we have categorized into three key groups:

* Demographic Data: Patient age, sex, and geographic location (Department, Province, District).

* Clinical Information: Diagnosis (ICD-10), medical specialty, and healthcare facility identification (IPRESS).

* Laboratory Data: Creatinine, blood glucose, and test results dates.

Due to file size limitations, the dataset is not included in this repository. Instead, the data is retrieved from an external source (Dropbox) using a direct download link.

## Methodology
The dataset is first preprocessed through data cleaning and selection of relevant variables.Exploratory data analysis is conducted to understand the main characteristics of the data. Unsupervised machine learning techniques, including K-Means Clustering, hierarchical clustering, PAM (K-Medoids), and DBSCAN, are then applied to identify patterns and groups of patients with similar clinical features. Finally, the resulting clusters are analyzed and visualized to interpret potential patient profiles and underlying structures in the data. Cluster validation techniques, such as the silhouette score, are used to evaluate the quality and separation of the clusters.

## Results
Several clustering algorithms were evaluated to identify patterns among patients, including K-Means Clustering, hierarchical clustering, PAM (K-Medoids), and DBSCAN. Model performance was compared using the Silhouette Score. Among the evaluated methods, DBSCAN obtained the highest silhouette score (0.49), indicating better cluster separation and compactness. Therefore, DBSCAN was selected as the final model to identify patient profiles.

## Project Structure
renal-disease-clustering/
│
├── data/
│   ├── raw/                
│   └── processed/         
│
├── notebooks/
│   └── renal_disease_patient_clustering.ipynb
│
├── src/                   
│   ├── preprocessing.py
│   ├── modeling.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── README.md
├── requirements.txt
└── .gitignore

## Limitations & Future Work

- Due to RAM limitations, the analysis was restricted to data from a single year (2024).
- Use of a limited set of variables.

Future work:
- Incorporate more clinical variables
- Deploy model as an interactive application