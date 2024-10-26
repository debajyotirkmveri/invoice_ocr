import os
from dotenv import load_dotenv
import google.generativeai as genai
import pytesseract
from PIL import Image
import streamlit as st
import requests
import time
import tempfile
import fitz  # PyMuPDF

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini AI with the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define Tesseract configuration
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

# Define the input prompt
input_prompt = (
    "Extract the following details from the text of the invoice:\n"
    "1. Invoice Number\n"
    "2. Invoice Date\n"
    "3. Bill To Details\n"
    "4. GSTN Number\n"
    "5. Invoice Amount\n"
    "6. Product Name\n"
    "7. Supplier Name (the company selling the goods, usually mentioned at the top of the invoices)\n"
    "8. Amount of each product if there are many products in the invoice, otherwise the total amount will be the Amount\n\n"
    "If there are many different product names, please ensure that each product is listed separately.\n"
    "If 'Amount' is not found for a particular product, please fill that value with an empty string.\n"
    "Return the results in a table format as follows:\n"
    "Invoice Number|Invoice Date|Bill To Details|GSTN Number|Invoice Amount|Product Name|Supplier Name|Amount\n"
    "Separate each field with a | and ensure that the order is exactly as specified.\n"
    "If any field is not found, use '--' in its place.\n"
)

# Function to convert PDF to image
def pdf_to_image(uploaded_file):
    # Save the uploaded PDF file to a temporary location
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Open the PDF file
    doc = fitz.open(temp_pdf_path)
    page = doc.load_page(0)  # Always use the first page
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Close the document before deleting the temporary file
    doc.close()
    os.remove(temp_pdf_path)

    return img

# Function to extract text from an image using Tesseract
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image, lang='eng', config=config_tesseract)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return None

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
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed after {retries} attempts: {e}")

# Streamlit app
st.title("Invoice Extraction App")

# File uploader for PDF files
uploaded_pdf = st.file_uploader("Upload an Invoice PDF", type=["pdf"])

if uploaded_pdf:
    # Convert PDF to image
    page_image = pdf_to_image(uploaded_pdf)
    
    if page_image:
        # Display the first page of the PDF
        st.subheader("First Page of the Invoice:")
        st.image(page_image)

        # Extract text from the image
        extracted_text = extract_text_from_image(page_image)
        
        if extracted_text:
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
