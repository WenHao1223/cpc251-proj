# Seismic Bump Prediction - CPC251 Project (Seismic 6)

This project applies supervised machine learning techniques to predict the occurrence of high-energy seismic bumps in underground mining environments. It was developed as part of the CPC251 course project.

## 📊 Project Overview

Mining operations are susceptible to seismic activities, which can be hazardous. This project explores the classification of seismic bumps using various machine learning models and selects the best-performing classifier for deployment based on metrics such as recall, precision, F1-score, and accuracy.

The project is completed in two phases:
- [**`Part 1`**](Seismic/CPC251_Project_Part1_Seismic6.ipynb) – Baseline modelling and performance comparison
- [**`Part 2`**](Seismic/CPC251_Project_Part2_Seismic6.ipynb) – Model refinement, hyperparameter tuning, and performance optimization

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## ⚙️ Key Steps
1. Data Preprocessing & Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature Selection  
4. Model Training & Evaluation
   - [**`Part 1`**](Seismic/CPC251_Project_Part1_Seismic6.ipynb)
      - Decision Tree Classifier (DTC)
      - K-Nearest Neighbors (KNN)
      - Support Vector Machine (SVM)
   - [**`Part 2`**](Seismic/CPC251_Project_Part2_Seismic6.ipynb)
      - Decision Tree Classifier (DTC)
      - Neural Network (NN)
5. Hyperparameter Tuning using GridSearchCV  
6. Model Comparison and Final Selection

## 🟦 Part 1 - Machine Learning (DTC vs KNN vs SVM)

### 📁 File
[**`CPC251_Project_Part1_Seismic6.ipynb`**](Seismic/CPC251_Project_Part1_Seismic6.ipynb): Main Jupyter notebook containing the entire analysis.

### 🏆 Model Insights
The Decision Tree Classifier showed better interpretability and competitive performance, making it a suitable choice for this task over more complex models.


## 🟧 Part 2 - Deep Learning (DTC vs NN)

### 📁 File
[**`CPC251_Project_Part2_Seismic6.ipynb`**](Seismic/CPC251_Project_Part2_Seismic6.ipynb): Contains advanced modelling pipeline, SMOTE balancing, Neural Network training, and final evaluation.

### 🏆 Model Insights
Neural Network (NN) was chosen as the optimal model for this task due to its superior recall in detecting high-energy seismic bumps, a critical factor for reducing false negatives in safety-critical mining operations.
While the DTC provided balanced performance, NN’s higher sensitivity to the minority class made it the preferred choice.


## 📌 How to Run
1. Clone the repository.
2. Open the notebook [**`Part 1`**](Seismic/CPC251_Project_Part1_Seismic6.ipynb) or [**`Part 2`**](Seismic/CPC251_Project_Part2_Seismic6.ipynb) in Jupyter.
3. Install required libraries:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn keras scikeras imbalanced-learn
```
4. Execute all Jupyter cells in the notebook sequentially.

## 📚 License
This project is submitted for academic purposes only and not intended for real-world deployment without further validation.