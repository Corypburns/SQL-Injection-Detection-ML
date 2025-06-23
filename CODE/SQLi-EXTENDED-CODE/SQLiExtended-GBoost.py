import os, pandas as pd, time, matplotlib.pyplot as mpl, csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from array import array
from datetime import datetime
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier

# Paths
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/GBoost/GBoost-SQLiExtended'
dataset_path = '/home/cory/code/CISResearchSummer2025/DATASETS/SQLi-Extended/sqli-extended.csv'
os.makedirs(output_path, exist_ok=True)
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_locale'] = True

# Timestamp and output file
now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
output_file = os.path.join(output_path, f"GradientBoostOutput_{now}.csv")
conf_matrix_file = os.path.join(output_path, f"ConfusionMatrix_{now}.png")

# Load dataset
df = pd.read_csv(dataset_path)
df = df[['Sentence', 'Label']].dropna()
df['Label'] = df['Label'].astype(int)

# Inputs and labels
input_data = df['Sentence']
label_data = df['Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

# TF-IDF
vector = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

# GradientBoost model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train_vec, y_train)

# Predict
prediction = model.predict(X_test_vec)
conf = model.predict_proba(X_test_vec)
X_index = X_test.index

# Confusion matrix values
cm = confusion_matrix(y_test, prediction)
tn, fp, fn, tp = cm.ravel()
count = Counter(y_test)
pos = count[1]
neg = count[0]

# Write predictions and show confusion matrix
try:
    with open(output_file, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Prediction", "Confidence", "Actual"])
        for idx, (i, pred) in enumerate(zip(X_index, prediction)):
            query = df.loc[i, 'Sentence']
            label = y_test.loc[i]  # ✅ FIXED here
            confidence = conf[idx][pred] * 100
            print(f"\nQuery: {query}\nLabel: {label}\nPrediction: {pred}\nConfidence: {confidence:.2f}")
            writer.writerow([query, pred, f"{confidence:.2f}%", label])

            if idx % 100 == 0:  # ✅ Optional speedup
                time.sleep(0.5)

        # Plot confusion matrix with raw integers only
        fig, ax = mpl.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', values_format='.0f')
        mpl.title("Confusion Matrix")
        mpl.tight_layout()
        fig.savefig(conf_matrix_file)  # ✅ Save plot
        mpl.show()

except KeyboardInterrupt:
    print("\n\n=== INTERRUPTED ===")
    print(classification_report(y_test, prediction))
    print(f"False Positive Rate (FPR): {fp / neg}")
    print(f"False Negative Rate (FNR): {fn / pos}")
    with open(os.path.join(output_path, 'results.txt'), mode='w') as f:
        f.write(classification_report(y_test, prediction))
        f.write("\n=== Confusion Matrix Stats ===\n")
        f.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
        f.write(f"FPR: {fp / neg:.4f}, FNR: {fn / pos:.4f}\n")

    # Plot confusion matrix even after interrupt
    fig, ax = mpl.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='.0f')
    mpl.title("Confusion Matrix")
    mpl.tight_layout()
    fig.savefig(conf_matrix_file)
    mpl.show()
