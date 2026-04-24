import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app
from cv.pipeline import preprocess_image, extract_text_from_image
 
client = TestClient(app)
 
SAMPLE_IMAGE_PATH = "tests/sample_legal.jpg"
 
 
def test_preprocess_returns_grayscale_array():
    dummy = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(dummy, "Test Legal Text", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    result = preprocess_image(dummy)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2   # grayscale = 2D
 
 
def test_ocr_on_synthetic_image():
    img = np.ones((100, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Indian Penal Code Section 420", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    preprocessed = preprocess_image(img)
    text = extract_text_from_image(preprocessed)
    assert isinstance(text, str)
    assert len(text) > 0
 
 
@pytest.mark.skipif(
    not os.path.exists(SAMPLE_IMAGE_PATH),
    reason="Place a sample image at tests/sample_legal.jpg to enable this test",
)
def test_document_endpoint_with_real_image():
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        response = client.post(
            "/api/v1/document/analyze",
            files={"file": ("sample_legal.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["file_type"] == "image"
    assert len(data["extracted_text"]) > 0
 
 
def test_document_endpoint_rejects_invalid_type():
    response = client.post(
        "/api/v1/document/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415