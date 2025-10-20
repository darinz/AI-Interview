# Supervised Learning Cheat Sheet

> A quick reference for understanding and applying supervised learning methods in machine learning.


## Key Concepts

- **Supervised Learning** is a type of machine learning where the model is trained on a labeled dataset (i.e., input-output pairs).
- The goal is to **learn a mapping** from inputs `X` to outputs `Y`.


## Algorithms

| Type        | Examples                     | Use Case                         |
|-------------|------------------------------|----------------------------------|
| Regression  | Linear Regression, Ridge     | Predicting continuous values     |
| Classification | Logistic Regression, Decision Trees, SVM | Categorizing discrete labels |


## Important Formulas

- **Mean Squared Error (MSE)**:  
  `MSE = (1/n) * Σ(yᵢ - ŷᵢ)²`

- **Accuracy**:  
  `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- **Log Loss** (binary classification):  
  `- [y * log(p) + (1 - y) * log(1 - p)]`


## Visual Summary

```

Input (X) → [Model] → Output (Ŷ)
↘️
True Y
↓
Loss Function
↓
Optimization

````


## Tips

- Normalize/standardize data for algorithms like **SVM** or **KNN**.
- Use **cross-validation** to assess model performance robustly.
- **Overfitting** is common — use regularization (e.g., L1, L2) or pruning.

---

## Bias-Variance Trade-off

- **High Bias**: Model is too simple (underfits)
- **High Variance**: Model is too complex (overfits)

> Goal: Find the right balance using validation techniques and model tuning.


## Code Example (Scikit-Learn)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
````

---

## Resources

* [Supervised Learning — scikit-learn docs](https://scikit-learn.org/stable/supervised_learning.html)
* [StatQuest YouTube: ML Intuition](https://www.youtube.com/user/joshstarmer)
* [Hands-On ML Book – Chapter 1–2](https://github.com/ageron/handson-ml3)