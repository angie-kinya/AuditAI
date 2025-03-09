from data_extraction import compliance_texts
from preprocessing import create_dataset, tokenize_dataset
from train_model import train_compliance, evaluate_compliance
from qa_model import qamodel_trainer, qa_eval_results

def main():
    # Step 1: Extract Data
    print("Step 1: Extracting data from PDF documents...")
    # compliance_texts is already imported from data_extraction
    print(f"Extracted {len(compliance_texts)} documents")
    print("Data extraction complete.\n")

    # Step 2: Preprocess & create Data for compliance classification
    print("Step 2: Preprocessing and creating data for compliance classification...")
    raw_dataset = create_dataset()
    tokenized_dataset = tokenize_dataset
    print("Dataset creation and tokenization complete.\n")

    # Step 3: Train and evaluate Compliance Classification Model
    print("Step 3: Training the compliance classification model...")
    train_compliance()
    eval_results = evaluate_compliance()
    print("Compliance Model Evaluation Results:")
    print(eval_results, "\n")

    # Step 4: Train and evaluate QA Model
    print("Step 4: Training the QA model...")
    qamodel_trainer()
    evaluate_qa_model = qa_eval_results()
    print("QA Model Evaluation Results:")
    print(evaluate_qa_model, "\n")

    print("All tasks completed.")

if __name__ == "__main__":
    main()
