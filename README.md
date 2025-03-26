# Megaline Plan Recommendation (Machine Learning Project)

## Project Summary

This is a machine learning project developed to help **Megaline**, a mobile telecom provider, recommend a more suitable mobile plan for its users.  
Based on user behavior data (calls, minutes, messages, internet usage), the task is to predict whether a user should switch to the **Smart** or **Ultra** plan.

The solution is implemented using Python and the `scikit-learn` machine learning framework, with additional support from `pandas`, `matplotlib`, and `seaborn` for data analysis and visualization.

---

## Objective

To train and evaluate classification models that can predict a user’s preferred plan (`Smart` or `Ultra`) based on their monthly usage behavior.  
The success criteria for the model is to achieve **at least 75% accuracy** on the test set.

---

## Machine Learning Workflow

1. **Exploratory Data Analysis (EDA):**
   - Used `pandas`, `matplotlib`, and `seaborn` to understand the data structure and distributions.
   - Visualized correlations between features.
   - Investigated class imbalance and behavioral patterns for Smart vs. Ultra users.

2. **Data Splitting:**
   - The dataset was split into training (60%), validation (20%), and test (20%) sets using `train_test_split()`.

3. **Model Training:**
   - Three classification models were trained and compared:
     - `DecisionTreeClassifier`
     - `RandomForestClassifier`
     - `LogisticRegression`

4. **Hyperparameter Tuning:**
   - Used loops to evaluate `max_depth` for Decision Tree and `n_estimators` for Random Forest.
   - Visualized accuracy changes based on hyperparameter values.

5. **Evaluation Metrics:**
   - Calculated and compared **accuracy**, **precision**, and **recall** for each model using `sklearn.metrics`.

6. **Model Selection:**
   - Chose the best-performing model based on validation metrics and tested it on the test set.

7. **Sanity Check:**
   - Used `DummyClassifier` as a baseline to verify that the selected model performs better than a naive approach.

8. **Interpretability:**
   - Visualized a simplified Decision Tree (`max_depth=3`) using `plot_tree()` to explain the logic behind the model's decisions.

---

## Results

| Model               | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| Decision Tree       | 0.796    | 0.780     | 0.467  |
| Random Forest       | 0.812    | 0.775     | 0.543  |
| Logistic Regression | 0.720    | 0.758     | 0.127  |

- `DummyClassifier` accuracy: **0.697**  
- Final model (`Random Forest`) test accuracy: **0.799**

---

## Final Conclusion

The **Random Forest** model provided the best overall performance and generalization to unseen data.  
It significantly outperformed the dummy classifier, confirming that it learned meaningful patterns from the data.

---

## Technologies and Libraries Used

- `pandas` — data manipulation
- `matplotlib`, `seaborn` — data visualization
- `scikit-learn`:
  - `train_test_split`
  - `DecisionTreeClassifier`
  - `RandomForestClassifier`
  - `LogisticRegression`
  - `DummyClassifier`
  - `accuracy_score`, `precision_score`, `recall_score`
  - `plot_tree`

---

## Author

Project by Marina Lozanskaya  
Completed as part of a machine learning practice project using open data.
