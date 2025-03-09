from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split

# Initialize tokenizer and model for question-answering(QA)
tokenizer_qa = BertTokenizerFast.from_pretrained("bert-base-uncased")
model_qa = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Define training arguments (Synthetic QA data)
qa_data = {
    "context": [
        "TechSecure Inc. demonstrates strong compliance in risk management and data protection, following guidelines from ISO 27001, NIST, COBIT, and GDPR.",
        "According to NIST, continuous monitoring and risk management are crucial for a robust cybersecurity framework."
    ],
    "question": [
        "Which frameworks are mentioned in the audit report?",
        "What does NIST emphasize in its framework?"
    ],
    "answers": [
        {"text": ["ISO 27001", "NIST", "COBIT", "GDPR"], "answer_start": [42, 54, 60, 67]},  # Multiple answers can be provided
        {"text": ["continuous monitoring", "risk management"], "answer_start": [10, 42]}
    ]
}

qa_dataset = Dataset.from_dict(qa_data)

# Preprocessing function for QA
def prepare_train_features(examples):
    # Tokenize contexts and questions (as pairs)
    tokenized_examples = tokenizer_qa(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map answer positions to token indices
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]
        # Take the first answer if there are multiple for simplicity
        answer_text = answer["text"][0]
        answer_start = answer["answer_start"][0]
        answer_end = answer_start + len(answer_text)

        # Find token indices corresponding to answer start and end
        token_start_index = 0
        while offsets[token_start_index][0] <= answer_start:
            token_start_index += 1
        token_start_index -= 1

        token_end_index = token_start_index
        while offsets[token_end_index][1] < answer_end:
            token_end_index += 1

        tokenized_examples["start_positions"].append(token_start_index)
        tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples

# Preprocess the QA dataset
tokenized_qa_dataset = qa_dataset.map(prepare_train_features, batched=True, remove_columns=qa_dataset.column_names)

# Split the dataset into training and evaluation sets
train_val_qa = tokenized_qa_dataset.train_test_split(test_size=0.2, seed=42)
train_qa = train_val_qa["train"]
eval_qa = train_val_qa["test"]

# Set up training arguments for QA
training_args_qa = TrainingArguments(
    output_dir="./auditai_qa_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=3e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
)

# Define a metric to evaluate the QA model
def compute_qa_metrics(eval_pred):
    # For demo, loss metrics will be computed
    return {}

qa_trainer = Trainer(
    model=model_qa,
    args=training_args_qa,
    train_dataset=train_qa,
    eval_dataset=eval_qa,
    compute_metrics=compute_qa_metrics
)

def qamodel_trainer():
    # Train the QA model
    qa_trainer.train()
    # Save the trained model
    qa_trainer.save_model("./auditai_qa_model")

def qa_eval_results():
    # Evaluate the QA model
    return qa_trainer.evaluate()