import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
try:
    from imblearn.over_sampling import SMOTE  # For handling imbalance if needed
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: 'imblearn' not installed. SMOTE will be skipped. Install with 'pip install imbalanced-learn'.")
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

data = pd.read_csv('full.csv')

data = data.dropna(subset=['Survived'])
print(f"Dataset size after removing rows with missing 'Survived': {data.shape}")

print("Dataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap of Missing Values")
plt.savefig('missing_values_heatmap.png')
plt.show()

# Basic statistics
print("\nDataset Description:")
print(data.describe())

# Visualize survival rate
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.savefig('survival_count.png')
plt.show()

# Check class balance
print("\nSurvival Class Distribution:")
class_dist = data['Survived'].value_counts(normalize=True)
print(class_dist)

# Survival rate by class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title("Survival Rate by Passenger Class")
plt.savefig('survival_by_class.png')
plt.show()

# Survival rate by sex
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title("Survival Rate by Sex")
plt.savefig('survival_by_sex.png')
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig('age_distribution.png')
plt.show()

# Outlier detection for Fare
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['Fare'])
plt.title("Fare Outlier Detection")
plt.savefig('fare_outliers.png')
plt.show()

# Step 2: Data Preprocessing
# Drop columns with too many missing values, irrelevant, or containing NaNs
data = data.drop(['Cabin', 'Name', 'Ticket', 'WikiId', 'Name_wiki', 'Hometown', 
                  'Boarded', 'Destination', 'Lifeboat', 'Body', 'Age_wiki', 'Class'], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])
data['Fare'] = imputer.fit_transform(data[['Fare']])
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Handle outliers in Fare (cap at 99th percentile)
fare_cap = data['Fare'].quantile(0.99)
data['Fare'] = data['Fare'].clip(upper=fare_cap)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# Check for any remaining missing values
print("\nMissing Values After Preprocessing:")
print(data.isnull().sum())

# Define features and target
X = data.drop(['PassengerId', 'Survived'], axis=1)
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance if significant (e.g., minority class < 40%)
if class_dist.min() < 0.4 and SMOTE_AVAILABLE:
    print("Class imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print("New training set size after SMOTE:", X_train_scaled.shape)
elif class_dist.min() < 0.4 and not SMOTE_AVAILABLE:
    print("Class imbalance detected, but SMOTE unavailable. Proceeding without balancing.")

# Step 3: Try Different Algorithms
algorithms = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "SVM": SVC(kernel='linear', probability=True)  # Enable probability for ROC-AUC
}

# Hyperparameter tuning for Random Forest
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print(f"Best Random Forest Params: {grid_search.best_params_}")
algorithms["Tuned Random Forest"] = grid_search.best_estimator_

results = {}
roc_auc_scores = {}
for name, model in algorithms.items():
    # Cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name} CV Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    
    # Fit and predict on test set
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    print(f"{name} Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        roc_auc_scores[name] = roc_auc
        print(f"{name} ROC-AUC: {roc_auc:.4f}")

# Visualize algorithm performance
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Algorithm Performance Comparison (CV Accuracy)")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.savefig('algorithm_comparison.png')
plt.show()

# Visualize ROC-AUC scores
if roc_auc_scores:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(roc_auc_scores.keys()), y=list(roc_auc_scores.values()))
    plt.title("ROC-AUC Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("ROC-AUC Score")
    plt.savefig('roc_auc_comparison.png')
    plt.show()

# Step 4: Feature Selection
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print("\nSelected Features:", selected_features)

# Visualize feature importance scores
scores = selector.scores_
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=scores)
plt.title("Feature Importance Scores")
plt.xticks(rotation=45)
plt.ylabel("F-Score")
plt.savefig('feature_importance.png')
plt.show()

# Step 5: Compare Algorithms with Feature Selection
results_selected = {}
roc_auc_scores_selected = {}
for name, model in algorithms.items():
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
    results_selected[name] = scores.mean()
    print(f"{name} with Feature Selection CV Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    
    # Test set evaluation
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, "predict_proba") else None
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        roc_auc_scores_selected[name] = roc_auc
        print(f"{name} with Feature Selection ROC-AUC: {roc_auc:.4f}")

# Visualize comparison with and without feature selection
comparison_df = pd.DataFrame({'All Features': results.values(), 'Selected Features': results_selected.values()}, index=results.keys())
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title("Accuracy Comparison: All Features vs Selected Features")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.savefig('feature_selection_comparison.png')
plt.show()

# Step 6: Additional Analysis (Correlation Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Features")
plt.savefig('correlation_heatmap.png')
plt.show()

# Save results for report
with open('results.txt', 'w') as f:
    f.write("Algorithm Results with All Features:\n")
    for name, score in results.items():
        f.write(f"{name}: {score:.4f}\n")
    f.write("\nAlgorithm Results with Selected Features:\n")
    for name, score in results_selected.items():
        f.write(f"{name}: {score:.4f}\n")
    f.write("\nROC-AUC Scores with All Features:\n")
    for name, score in roc_auc_scores.items():
        f.write(f"{name}: {score:.4f}\n")
    f.write("\nROC-AUC Scores with Selected Features:\n")
    for name, score in roc_auc_scores_selected.items():
        f.write(f"{name}: {score:.4f}\n")
    f.write("\nSelected Features:\n")
    f.write(str(selected_features))

print("Analysis complete. Results saved to 'results.txt'. All plots saved as PNG files.")