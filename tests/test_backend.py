from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_segment_demo():
    response = client.post("/api/segment/demo")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "volumes_cm3" in data
    assert "Whole Tumor (WT)" in data["volumes_cm3"]
    assert "Tumor Core (TC)" in data["volumes_cm3"]
    assert "Enhancing Tumor (ET)" in data["volumes_cm3"]

def test_segment_sample():
    response = client.post("/api/segment/sample/sample_patient_flair.nii.gz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction_quality"] == "good"
    assert data["volumes_cm3"]["Whole Tumor (WT)"] > 0
    # Tumor must be a focal lesion (< 15 cm3 on our standard sample), not the entire brain
    assert data["volumes_cm3"]["Whole Tumor (WT)"] < 15.0
    assert data["meshes"]["wt"]["vertex_count"] > 0

def test_segment_upload_single():
    with open("sample_data/sample_patient_flair.nii.gz", "rb") as f:
        files = [("files", ("sample_patient_flair.nii.gz", f, "application/gzip"))]
        response = client.post("/api/segment/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["volumes_cm3"]["Whole Tumor (WT)"] < 15.0
    assert data["meshes"]["wt"]["vertex_count"] > 0

def test_spa_index_serving():
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

