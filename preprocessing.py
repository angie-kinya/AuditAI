from transformers import BertTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Combining compliance documents excerpts and the synthetic audit report sample
texts = [
    compliance_texts["ISO27001"][:5000],
    compliance_texts["NIST"][:5000],
    compliance_texts["GDPR"][:5000],
    compliance_texts["COBIT"][:5000],
    compliance_texts["Report"][:5000]
]
labels = [0, 0, 0, 0, 1] # The 1 indicates the audit report

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# Create dataset
dataset = Dataset.from_dict({"text": texts, "label": labels})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset into training and validation sets
train_val = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val["train"]
eval_dataset = train_val["test"]