import xgboost as xgb, os, pandas as pd, time, matplotlib.pyplot as mpl, csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from array import array
from datetime import datetime
from collections import Counter

# Paths
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/XDGBoost/XGBoost-GAMBLERYU'
dataset_path = '/home/cory/code/CISResearchSummer2025/DATASETS/GAMBLERYU/GAMBLERYU.csv'
os.makedirs(output_path, exist_ok=True)

# Timestamp and file paths
now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
output_file = os.path.join(output_path, f"XGBoost_{now}.csv")
conf_matrix_file = os.path.join(output_path, f"ConfusionMatrix_{now}.png")

# Init output file
with open(output_file, mode='w') as f:
    f.write("Query,Prediction,Confidence,Actual\n")

# Load dataset
df = pd.read_csv(dataset_path)
df = df[['Query', 'Label']].dropna()
df['Label'] = df['Label'].astype(int)

# Input and label columns
input_data = df['Query']
label_data = df['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

# TF-IDF vectorization
vector = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

# XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_vec, y_train)

# Predictions
prediction = model.predict(X_test_vec)
conf = model.predict_proba(X_test_vec)
X_index = X_test.index

# Confusion matrix
cm = confusion_matrix(y_test, prediction)
tn, fp, fn, tp = cm.ravel()
count = Counter(y_test)
pos = count[1]
neg = count[0]

# Save predictions and show matrix
try:
    with open(output_file, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Prediction", "Confidence", "Actual"])
        for idx, (i, predict) in enumerate(zip(X_index, prediction)):
            query = df.loc[i, 'Query']
            label = y_test.loc[i]
            confidence = conf[idx][predict] * 100
            print(f"\nQuery: {query}\nLabel: {label}\nPrediction: {predict}\nConfidence: {confidence:.2f}")
            writer.writerow([query, predict, f"{confidence:.2f}%", label])
            if idx % 100 == 0:
                time.sleep(0.5)

        # Plot confusion matrix
        fig, ax = mpl.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', values_format='.0f')
        mpl.title("Confusion Matrix")
        mpl.tight_layout()
        fig.savefig(conf_matrix_file)
        mpl.show()

except KeyboardInterrupt:
    print("\n\n=== INTERRUPTED ===")
    print(classification_report(y_test, prediction))
    print(f"False Positive Rate (FPR): {fp / neg}")
    print(f"False Negative Rate (FNR): {fn / pos}")
    with open(os.path.join(output_path, 'results.txt'), mode='w') as f:
        f.write(classification_report(y_test, prediction))
        f.write("\n=== Confusion Matrix Stats ===\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"False Positive Rate (FPR): {fp / neg:.4f}\n")
        f.write(f"False Negative Rate (FNR): {fn / pos:.4f}\n")

    # Show confusion matrix even after interruption
    fig, ax = mpl.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='.0f')
    mpl.title("Confusion Matrix")
    mpl.tight_layout()
    fig.savefig(conf_matrix_file)
    mpl.show()
