import streamlit as st
from pypdf import PdfReader
from docx import Document

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="MedExplain AI",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🩺 MedExplain AI")

st.subheader("AI Medical Report Explainer")

st.write(
    "Upload your medical report and get a simple-language explanation."
)


# -----------------------------
# File Text Extraction
# -----------------------------

def extract_pdf_text(file):
    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(file):
    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your medical report",
    type=["pdf", "docx"]
)


if uploaded_file:

    st.success("Report uploaded successfully!")

    if uploaded_file.name.endswith(".pdf"):
        report_text = extract_pdf_text(uploaded_file)

    elif uploaded_file.name.endswith(".docx"):
        report_text = extract_docx_text(uploaded_file)

    else:
        report_text = ""


    # -----------------------------
    # Display Extracted Text
    # -----------------------------

    with st.expander("View extracted report text"):

        st.text(report_text)


    # -----------------------------
    # Ollama Model
    # -----------------------------

    llm = OllamaLLM(
        model="llama3.2"
    )


    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = PromptTemplate(
        input_variables=["report"],
        template="""
You are MedExplain AI, an AI assistant that explains
medical reports in simple language.

Analyze the following medical report.

Provide the response in this structure:

1. Simple Summary
2. Important Findings
3. Abnormal or Concerning Values
4. What These Findings May Mean
5. Questions to Ask a Doctor
6. Important Disclaimer

Use simple language that a non-medical person can understand.

Do not diagnose the patient.
Do not prescribe medicines.
Do not claim certainty when the report does not provide enough information.

Medical Report:

{report}
"""
    )


    # -----------------------------
    # Generate Explanation
    # -----------------------------

    if st.button("🔍 Explain My Report"):

        if not report_text.strip():

            st.error(
                "Could not extract text from the uploaded report."
            )

        else:

            with st.spinner(
                "Analyzing your medical report..."
            ):

                chain = prompt | llm

                response = chain.invoke(
                    {"report": report_text}
                )


            st.subheader("📋 Medical Report Explanation")

            st.write(response)