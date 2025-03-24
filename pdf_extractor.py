import fitz
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import io
from io import BytesIO
import os

ocr = PaddleOCR(use_angle_cls=True, lang="en")  # Initialize OCR
  
def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file."""
    doc = fitz.open("pdf", pdf_bytes)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_images_from_pdf(pdf_bytes):
    """Extract text from images in a PDF using PaddleOCR."""
    image_texts = []
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
        for page_num in range(len(pdf_doc)):
            images = pdf_doc[page_num].get_images(full=True)
            print(f"📸 Page {page_num + 1}: {len(images)} images found")

            if not images:
                print(f"⚠️ No images found on page {page_num + 1}")
                continue  # Skip if no images exist

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                img_data = base_image["image"]

                # Convert image to PIL format
                image = Image.open(io.BytesIO(img_data))

                # ✅ Debug: Print OCR raw output before processing
                result = ocr.ocr(np.array(image), cls=True)
                print(f"🔍 OCR Raw Output (Page {page_num + 1}, Image {img_index + 1}):", result)

                # ✅ Fix: Skip None results
                if result is None or not isinstance(result, list) or len(result) == 0:
                    print(f"⚠️ No text detected on page {page_num + 1}, image {img_index + 1}")
                    continue  # Skip processing if no text is found

                extracted_text = " ".join([word[1][0] for res in result if res is not None for word in res if word is not None])
                image_texts.append(extracted_text)

    return image_texts


if __name__ == "__main__":
    with open("/Users/durotoyejoshua/Desktop/DS-Lab/RAG_System/uploads/insight-why-were-not-as-self-aware-as-we-think-and-how-seeing-ourselves-clearly-helps-us-succeed-at-work-and-in-life-pdfdrive-.pdf", "rb") as f:
        pdf_bytes = f.read()
    extracted_text = extract_text_from_pdf(pdf_bytes)
    extracted_images_text = extract_images_from_pdf(pdf_bytes)
    print("Extracted Text:", extracted_text[:500])
    print("OCR Extracted Text from Images:", extracted_images_text)