# 7.Model-selection
 implement by yourself at least two approaches from the following for multi-class classification: linear, k-nearest neighbour (kNN), support-vector machine (SVM), decision tree (DT), random forest (RF), or statistical classifier. You may follow Occam's razor in the process. 
![image](https://github.com/user-attachments/assets/712d659c-e6fd-4a87-a0c6-e9cda49b32bb)
## 📊 Classification Project Instructions (Python or Matlab)

In this project, we work with a prepared dataset to understand its structure and experiment with different classification methods.

---

### 🔁 Workflow

1. **Data onboarding**  
   Load and study the dataset to get familiar with its features and structure.

2. **Data standardisation**  
   Standardise the dataset so that all features have the same scale.  
   After this step, each feature should have the same average value and variability.  
   This improves the performance of many machine learning models.

3. **Data division**  
   The dataset has already been split into training and testing subsets.  
   Review these subsets and verify whether they follow a consistent distribution across the target classes.

📁 You are provided with both CSV and MAT versions of the training and test datasets.

---

### 🧠 Classification Approaches

After reviewing the data, you will move on to designing and testing classification models.

1. **Understanding the data**  
   The dataset contains several classes. However, the data is **not balanced**, meaning the number of samples varies across classes.  
   You may need to use validation techniques or rebalancing methods.

2. **Classifier design**  
   Choose at least two classification methods to test.  
   Possible approaches include:  
   - Linear classifiers  
   - k-nearest neighbours (kNN)  
   - Support vector machines (SVM)  
   - Decision trees (DT)  
   - Random forests (RF)  
   - Statistical classifiers  

   Try to keep your models as simple as possible while maintaining performance.

3. **Classifier training**  
   Train the models using the training set.  
   Make sure that your classifiers are learning meaningful patterns, and try to ensure they will generalise well to new, unseen data.  
   Optionally, perform hyperparameter tuning using part of the training data.

---

### ✅ Final Evaluation

Once the classifiers are trained:

- **Classification**  
  Use the trained models to predict the class labels for the test data.

- **Performance evaluation**  
  Compare how well each model performs using classification accuracy.  
  Consider also the accuracy for each individual class, and generate confusion matrices if needed.  
  Document the performance and behaviour of each model clearly.
![image](https://github.com/user-attachments/assets/d6b0a521-d8f3-49e2-ae28-e4e423448343)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_data = pd.read_csv('train_data.csv', header=None)
test_data = pd.read_csv('test_data.csv', header=None)
X_train = train_data[[0, 1, 2]].values 
y_train = train_data[3].values         
X_test = test_data[[0, 1, 2]].values   
y_test = test_data[3].values         
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)
num_classes = np.unique(y_train).shape[0]  
class_counts = pd.Series(y_train).value_counts() 

###########################################################  k-NN classifier
# Function to compute Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# k-NN classifier
def k_nearest_neighbors(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        # Calculate the distance between the test point and all training points
        distances = [euclidean_distance(test_point, x_train) for x_train in X_train]
        
        # Get the k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Majority vote: choose the most frequent class among the neighbors
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(int(most_common))  # Ensure the result is an integer, not np.float64
    return y_pred
k = 3 
y_pred_knn = k_nearest_neighbors(X_train_standardized, y_train, X_test_standardized, k)
###########################################################  Linear Classifier (multi-class SVM-like)
def linear_classifier(X_train, y_train, X_test):
    n_classes = np.unique(y_train).shape[0]
    weights = np.random.randn(n_classes, X_train.shape[1])
    bias = np.random.randn(n_classes)
    def decision_function(x, class_idx):
        return np.dot(x, weights[class_idx]) + bias[class_idx]
    y_pred = []
    for x in X_test:
        class_scores = [decision_function(x, i) for i in range(n_classes)]
        y_pred.append(int(np.argmax(class_scores) + 1))  # Ensure result is integer (class labels are 1, 2, 3)
    return y_pred
y_pred_linear = linear_classifier(X_train_standardized, y_train, X_test_standardized)
#################### Output number of classes and balance #####################
print("Number of classes data contains:", num_classes)
print("Data is imbalanced with class counts as follows:")
print(class_counts.to_string())
########################### Performance Evaluation ##############################
print("k-NN Model Performance:")
print("Accuracy (k-NN):", accuracy_score(y_test, y_pred_knn))
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix (k-NN):\n", conf_matrix_knn)
print("Classification Report (k-NN):\n", classification_report(y_test, y_pred_knn))
print("\nLinear Classifier Model Performance:")
print("Accuracy (Linear Classifier):", accuracy_score(y_test, y_pred_linear))
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
print("Confusion Matrix (Linear Classifier):\n", conf_matrix_linear)
print("Classification Report (Linear Classifier):\n", classification_report(y_test, y_pred_linear))
########################## Confusion Matrix Plots ##############################
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.title("Confusion Matrix for k-NN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_linear, annot=True, fmt="d", cmap="Oranges", xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.title("Confusion Matrix for Linear Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

```
![image](https://github.com/user-attachments/assets/a6d65e37-4277-4155-9ca2-aa50afa32c6d)
