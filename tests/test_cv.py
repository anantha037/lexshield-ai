import os
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_document_ocr_endpoint():

    image_path = "sample_order.jpg"

    assert os.path.exists(image_path), f"Test image {image_path} not found!"

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/api/document",
            files={"file":("sample_order.jpg",image_file,"image/jpeg")}
        )
    
    assert response.status_code == 200

    data = response.json()

    assert data["status"].lower() == "success"
    assert "extracted_text" in data

    extracted_text = data["extracted_text"].upper()
    assert len(extracted_text) > 0
    assert "SUPREME COURT" in extracted_text