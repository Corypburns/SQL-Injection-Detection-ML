import os, pandas as pd, time, matplotlib.pyplot as mpl, csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from array import array
from datetime import datetime
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Paths
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/AdaBoost/AdaBoost-SQLiExtended'
dataset_path = '/home/cory/code/CISResearchSummer2025/DATASETS/SQLi-Extended/sqli-extended.csv'
os.makedirs(output_path, exist_ok=True)

# Clean timestamp
now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
output_file = os.path.join(output_path, f"XGBoostOutput_{now}.csv")

if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
    with open(output_file, mode='w') as f:
        f.write("Query, Prediction, Confidence, Actual\n")

# Load Dataset
df = pd.read_csv(dataset_path)

# Fill missing values
df = df[['Sentence', 'Label']].dropna()
df['Label'] = df['Label'].astype(int)

# Data labeled in the csv file
df['Sentence']
df['Label']

# Inputs and labels
input_data = df['Sentence']
label_data = df['Label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vector = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

# AdaBoost Model
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train_vec, y_train)

# Evaluation
prediction = model.predict(X_test_vec)
X_index = X_test.index
conf = model.predict_proba(X_test_vec)

# Confusion matrix values
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()

count = Counter(y_test)
pos = count[1]
neg = count[0]

# Write predictions to CSV
try:
    with open(output_file, mode='w') as f:
        f.write("Query,Prediction,Confidence,Actual\n")
        writer = csv.writer(f)
        for indx, (i, predict) in enumerate(zip(X_index, prediction)):
            query = df.loc[i, 'Sentence']
            label = df.loc[i, 'Label']
            confidence = (conf[indx][predict]) * 100
            print_statement = f"\Query: {query}\nLabel: {label}\nPrediction: {predict}\nConfidence: {confidence:.2f}"
            print(print_statement)
            writer.writerow([query, predict, f"{confidence:.2f}%", label])
            time.sleep(0.5)
        ConfusionMatrixDisplay.from_predictions(y_test, prediction)
        mpl.show()
except KeyboardInterrupt:
    print("\n\n=== INTERRUPTED ===")
    print(classification_report(y_test, prediction))
    os.chdir(output_path)
    with open('results.txt', mode='w') as f:
        f.write(classification_report(y_test, prediction))
        f.write("\n=== Confusion Matrix Stats ===\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"False Positive Rate (FPR): {fp / neg}\n")
        f.write(f"False Negative Rate (FNR): {fn / pos}\n")
    ConfusionMatrixDisplay.from_predictions(y_test, prediction)
    mpl.show()

