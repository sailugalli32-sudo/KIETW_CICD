# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset from scikit-learn; it includes 150 samples with 4 features each
iris = load_iris()
X, y = iris.data, iris.target  # X: features (sepal/petal measurements), y: target classes (0=setosa, 1=versicolor, 2=virginica)

# Split the dataset into training (80%) and testing (20%) sets for model evaluation; use random_state=42 for reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier with 100 decision trees (n_estimators=100) and a fixed random seed for consistency
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier by fitting it to the training data
clf.fit(X_train, y_train)

# Save the trained model to a file named 'iris_model.pkl' using joblib for efficient serialization
joblib.dump(clf, "iris_model.pkl")
print("Model saved as iris_model.pkl")  # Print a confirmation message