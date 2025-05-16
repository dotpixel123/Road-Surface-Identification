## Road Surface Condition Classification with YOLOv9

A lightweight Modal‑powered pipeline that fine‑tunes a YOLO model on the RSCD (Road Surface Condition Database) and performs frame‑by‑frame inference on video data to detect conditions like wet asphalt, snow, etc.

---

### 📦 Tech Stack
- **Python 3.12**  
- **Modal.app** for serverless compute & volumes  
- **Ultralytics YOLOv9** for object detection / classification  
- **OpenCV** for image handling & video creation  
- **Seaborn & Matplotlib** for visualization  
- **RoboFlow** for annotation & dataset management  
- **NVIDIA A100 / T4** GPUs  

---

### 🗂️ Dataset
- **RSCD**: originally ~1 million images  
- **Sampling**: randomly selected **30,000** images for training/validation/testing  
- **Annotation**: bounding‑box labels created in RoboFlow  
- **YAML config**: `data/dataset.yaml`  

---

### 📊 Dataset Distribution (Train Set)
<p align="center">
  <img src="Output\Figure_1.png" width="600" alt="Distribution of RSCD test images">
</p>
*Distribution of sampled classes in the RSCD test set (30 k images).*

---

### 🗂️ What I Did
1. Mounted a persistent volume (`yolo-finetune`) under `/root/data`  
2. Fine‑tuned YOLOv9 on the sampled RSCD images using Modal’s A100 instances (80 epochs by default)  
3. Ran inference on a separate video frame set—saving annotated frames to `predictions/{model_id}`  
4. Compiled those frames into an MP4 via OpenCV’s `VideoWriter`  

---

### 📸 Sample Outputs
<p align="center">
  <img src="Output/image.png" width="200" alt="Wet Asphalt">
  <img src="Output/image2.png" width="200" alt="Snowy Road">
</p>

<details>
<summary>▶️ Sample Inference Video</summary>

[![Inference Video]](Output/output_video.mp4)
</details>

---

### 🔖 Results
<p align="center">
  <img src="Output\loss.png" width="200" alt="Wet Asphalt">
  <img src="Output\confusion.png" width="200" height="130" alt="Wet Asphalt">
</p>

