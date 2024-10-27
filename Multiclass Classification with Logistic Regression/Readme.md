# Multiclass Classification on Raisin Dataset using Logistic Regression and SVM

This project demonstrates multiclass classification on the Raisin dataset using Logistic Regression and Support Vector Machine (SVM). The Raisin dataset consists of features that describe physical characteristics of raisin varieties (`Besni` and `Kecimen`), and the goal is to classify these raisin types using machine learning techniques.

## Project Overview

In this project, we implemented several classification models to predict the raisin type based on various physical attributes. We use logistic regression, which is commonly used for binary classification but adapted here for multiclass classification. We also explore Support Vector Machine (SVM) with different kernel functions (`rbf` and `linear`) to compare performance.

---

![image](https://github.com/user-attachments/assets/41d74030-b4cc-45ae-82b2-ff48de6730e8)



## Dataset

The Raisin dataset consists of 900 samples and 7 features:
- **Area**: Total number of pixels inside the boundary of the raisin.
- **Perimeter**: Length of the boundary.
- **MajorAxisLength**: Longest axis length.
- **MinorAxisLength**: Shortest axis length.
- **Eccentricity**: Measure of how elongated the shape is.
- **ConvexArea**: Number of pixels in the smallest convex shape that can enclose the raisin.
- **Extent**: Ratio of the pixels in the bounding box to the total pixels.

The target variable is:
- **Class**: Type of raisin (`Besni` or `Kecimen`).

---

## Project Structure

- **Data Preprocessing**: We load the data, handle missing values, and scale the features.
- **Modeling**: Logistic regression and SVM models are trained and evaluated.
- **Performance Metrics**: We use precision, recall, and F1-score to evaluate the model's performance.

---

## Code Breakdown

### 1. Loading the Data
We start by loading the data from the Excel file and performing some exploratory analysis. 

```python
import pandas as pd

# Load the dataset
df = pd.read_excel('Raisin_Dataset.xlsx')

# Display the first few rows
df.head()
```

We import necessary libraries and inspect the data. This allows us to check for any missing values and understand the feature distributions.

### 2. Data Preprocessing

Before training the model, it’s important to preprocess the data. This involves separating the target variable from the features and normalizing the feature values for better model performance.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- We split the dataset into training and testing sets (80% training, 20% testing).
- Feature scaling is done using `StandardScaler` to ensure all features are on the same scale, which is essential for models like SVM.

### 3. Logistic Regression Model

Logistic regression is used for binary classification tasks, but in this case, we apply it to a multiclass problem. Scikit-learn’s `LogisticRegression` class handles this using the `multinomial` strategy.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train a logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
report = classification_report(y_test, y_pred)
print(report)
```

- The logistic regression model is trained using the `multinomial` approach to handle multiclass classification.
- We use the `classification_report` to display metrics such as precision, recall, and F1-score.

### 4. Support Vector Machine (SVM) Models

To enhance the robustness of the project, we apply SVM models with two different kernel functions: `rbf` and `linear`.

#### SVM with RBF Kernel

```python
from sklearn.svm import SVC

# Train SVM with RBF kernel
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

- The Radial Basis Function (RBF) kernel maps the features into a higher-dimensional space to find a separating hyperplane for non-linearly separable data.

#### SVM with Linear Kernel

```python
# Train SVM with Linear kernel
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

- The linear kernel is applied to find a linear decision boundary in the feature space.

### 5. Model Performance

The final evaluation shows that both the logistic regression and SVM models perform well, with the SVM models achieving slightly higher accuracy and F1-scores.

---

## Results and Observations

- **Logistic Regression**: A strong baseline model with good performance.
- **SVM with RBF Kernel**: Achieves better results on non-linear data.
- **SVM with Linear Kernel**: Effective for linearly separable data, and it performs well in this case.
  
Overall, SVM with an RBF kernel produced the highest accuracy, while logistic regression also proved to be a solid model for this multiclass problem.

---

## How to Run the Code

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/raisin-classification.git
   cd raisin-classification
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook code.ipynb
   ```

---

## Conclusion

This project demonstrates the application of multiclass classification using logistic regression and SVM on the Raisin dataset. We explore different kernels for SVM and provide a comparative analysis of model performance. This work can be extended by testing other classifiers or feature engineering techniques to further improve accuracy.
