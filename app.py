import os
from dotenv import load_dotenv
import google.generativeai as genai
import pytesseract
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import requests
import time
import tempfile
from PyPDF2 import PdfReader  # Updated import

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini AI with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a directory for tessdata if it doesn't exist
if not os.path.exists('tessdata'):
    os.makedirs('tessdata')

# Download the English trained data for Tesseract
eng_traineddata_url = "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
eng_traineddata_path = './tessdata/eng.traineddata'

if not os.path.exists(eng_traineddata_path):
    response = requests.get(eng_traineddata_url)
    with open(eng_traineddata_path, 'wb') as f:
        f.write(response.content)

# Tesseract configuration
config_tesseract = '--tessdata-dir tessdata --psm 6'

# Define the input prompt with placeholders for missing fields
input_prompt = (
    "Extract the following details from the text of the invoice:\n"
    "1. Invoice Number\n"
    "2. Invoice Date\n"
    "3. Bill To Details\n"
    "4. GSTN Number\n"
    "5. Invoice Amount\n"
    "6. Product Name\n"
    "7. Supplier Name (the company selling the goods, usually mentioned at the top of the invoices)\n"
    "8. Amount of each product if there are many products, otherwise the total amount will be the Amount\n\n"
    "Return the results in a table format as follows:\n"
    "Invoice Number|Invoice Date|Bill To Details|GSTN Number|Invoice Amount|Product Name|Supplier Name|Amount\n"
    "If any field is not found, use '--' in its place.\n"
)

# Function to extract text from PDF using Tesseract
def extract_text_from_pdf(pdf_file):
    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        # Check the number of pages in the PDF
        with open(tmp_file_path, "rb") as f:
            reader = PdfReader(f)
            if len(reader.pages) != 1:
                raise ValueError("The PDF must be a single page.")

        # Convert the first page of the PDF to image
        pages = convert_from_path(tmp_file_path, dpi=300)
        first_page_image = pages[0]
        text = pytesseract.image_to_string(first_page_image, lang='eng', config=config_tesseract)
        return text, first_page_image
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, None

# Function to get response from Gemini API with retry logic
def get_gemini_response(text, prompt, retries=5, delay=5):
    model = genai.GenerativeModel("gemini-1.5-flash")
    formatted_prompt = f"{prompt}\nText: {text}"
    
    for attempt in range(retries):
        try:
            response = model.generate_content([formatted_prompt])
            if response and response.parts:
                return response.parts[0].text
            else:
                raise ValueError("No valid response parts were returned from the API.")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed after {retries} attempts: {e}")

# Streamlit app
st.title("Invoice Extraction App")

# File uploader for PDF files
uploaded_pdf = st.file_uploader("Upload an Invoice PDF", type=["pdf"])

if uploaded_pdf:
    # Extract text and image from the uploaded PDF
    extracted_text, page_image = extract_text_from_pdf(uploaded_pdf)
    
    if extracted_text and page_image:
        # Display the first page of the PDF
        st.subheader("First Page of the Invoice:")
        st.image(page_image)

        # Display extracted text for verification
        st.subheader("Extracted Text from Invoice:")
        st.text(extracted_text)

        # Process the text when the button is clicked
        if st.button("Process Invoice"):
            try:
                extracted_details = get_gemini_response(extracted_text, input_prompt)
                st.subheader("Extracted Details:")
                st.text(extracted_details)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
else:
    st.info("Please upload a PDF invoice to proceed.")
