# Code that generates display's are made by ChatGPT
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# Load processed data for features, generated in Part1.py
df = pd.read_csv("car_cleaned.csv")

# Split features and target
x = df.drop('class', axis=1)                                                   # Features
classMap = {0: 'unacc', 1/3: 'acc', 2/3: 'good', 1: 'vgood'}                   # map to convert targets from float to string
y = df['class'].map(classMap)                                                  # Target

# Split into training and test sets (80/20)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# Track performance results
results = []


# Model 1: Decision Tree
dt_configs = [3, 5, 10]  # max_depth values

for depth in dt_configs:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)

    print(f"Decision Tree (max_depth={depth})")
    print(classification_report(yTest, yPred, digits=3))
    
    tree.plot_tree(decision_tree=model, rounded=True, fontsize=7)
    cm = confusion_matrix(yTest, yPred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f'Decision Tree (depth={depth})')
    plt.show()

    results.append({
        'model': 'DecisionTree',
        'param': f'max_depth={depth}',
        'accuracy': model.score(xTest, yTest)
    })


# Model 2: K-Nearest Neighbors
knn_configs = [3, 5, 7]  # number of neighbors

for k in knn_configs:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)

    print(f"KNN (k={k})")
    print(classification_report(yTest, yPred, digits=3))

    cm = confusion_matrix(yTest, yPred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f'KNN (k={k})')
    plt.show()

    results.append({
        'model': 'KNN',
        'param': f'k={k}',
        'accuracy': model.score(xTest, yTest)
    })


# Summary
print("Summary")
for r in results:
    print(f"{r['model']} ({r['param']}): Accuracy = {r['accuracy']:.3f}")