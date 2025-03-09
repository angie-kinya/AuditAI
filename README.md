# AuditAI
# IT Compliance & Audit AI Chatbot

## Project Overview

This project is an AI-powered chatbot designed to assist with IT compliance and audit-related questions. The chatbot is trained on key compliance frameworks, including:

- ISO 27001 (Information Security Management)

- NIST (National Institute of Standards and Technology)

- COBIT (Control Objectives for Information and Related Technologies)

- GDPR (General Data Protection Regulation)

The model is built using BERT and fine-tuned for both compliance classification and question-answering tasks. It processes IT compliance documents and synthetic audit reports to provide reliable responses.

## Project Structure
```
├── data_extraction.py    # Extracts text from compliance documents & synthetic audit reports
├── preprocessing.py      # Cleans, processes, and tokenizes the extracted data
├── train_model.py        # Trains and evaluates the compliance classification model
├── qa_model.py           # Trains and evaluates the question-answering model
├── main.py               # Orchestrates the entire workflow
├── README.md             # Project documentation (this file)
```
## Steps to Run the Project

1. Extract Data

Run data_extraction.py to extract text from compliance documents and audit reports.

2. Preprocess Data

Run preprocessing.py to clean, structure, and tokenize the data for training.

3. Train the Compliance Model

Run train_model.py to train and evaluate the compliance classification model.

4. Train the QA Model

Run qa_model.py to fine-tune a BERT-based model for answering IT audit-related questions.

5. Run the Main Pipeline

Execute main.py to run all steps sequentially and test the model outputs.

## Next Steps

- Integrate with Flask: Deploy a backend API to serve the chatbot.

- Develop a Frontend: Create an interactive user interface for users to ask compliance-related questions.

- Enhance Model Performance: Fine-tune models further with additional datasets.

## Future Enhancements

- Expand support for more compliance frameworks.

- Improve response accuracy through reinforcement learning.

- Implement real-time document ingestion for up-to-date compliance assistance.

## Acknowledgments

This project is inspired by the need for AI-driven compliance assistance in IT audits, leveraging state-of-the-art NLP models to automate and simplify audit processes.