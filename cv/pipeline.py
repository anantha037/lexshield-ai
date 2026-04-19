import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return thresh

def extract_text_from_image(image_path: str, lang:str='eng') -> str:
    processed_img = preprocess_image(image_path)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, lang=lang,config=custom_config)

    return text.strip()

def process_pdf(pdf_path: str, lang: str = 'eng') -> str:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    
    pages = convert_from_path(pdf_path)

    full_text = []

    for i, page in enumerate(pages):

        temp_img_path = f'temp_page_{i}.jpg'
        page.save(temp_img_path,'JPEG')

        page_text = extract_text_from_image(temp_img_path,lang=lang)
        full_text.append(page_text)

        os.remove(temp_img_path)

    return "\n\n--- PAGE BREAK ---\n\n".join(full_text)