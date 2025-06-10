import os
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths & setup
file_path = '/home/cory/code/CISResearchSummer2025/DATASETS/SQLiV3/SQLiV3_CLEANED.csv'
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/BERT/BERT-SQLiV3'
os.makedirs(output_path, exist_ok=True)
now = datetime.now().strftime("%y-%m-%d %Hh:%Mm:%Ss")
output_file = os.path.join(output_path, f'Output_{now}.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CSV
df = pd.read_csv(file_path)

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    texts = [str(text) if text is not None else "" for text in examples['Sentence']]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label'])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop (simple 3 epochs)
epochs = 3
model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['Label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Average loss: {avg_loss:.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['Label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

# Metrics and print
print(classification_report(all_labels, all_preds))
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# Save predictions
results_df = test_df.reset_index(drop=True)
results_df['Predicted_Label'] = all_preds
results_df['Prob_Class_0'] = [p[0] for p in all_probs]
results_df['Prob_Class_1'] = [p[1] for p in all_probs]
results_df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

