import PyPDF2
import os

def extract_text_from_pdf(file_path):
    text_data = ""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_data += text + "\n"
    return text_data

# Extract text from each compliance document
documents = {
    "ISO27001": "./documents/ISO27001.pdf",
    "GDPR": "./documents/GDPR.pdf",
    "NIST": "./documents/NIST_Cybersecurity_Framework.pdf",
    "COBIT": "./documents/COBIT_Framework.pdf",
    "Report": "./documents/SyntheticAuditReport.pdf"
}

compliance_texts = {}
for key, path in documents.items():
    compliance_texts[key] = extract_text_from_pdf(path)
    print(f"Extracted {len(compliance_texts[key])} characters from {key} document")
