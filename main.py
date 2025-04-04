from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics~=8.3.93", "opencv-python~=4.10.0"])
)

# Create or retrieve the volume
volume = modal.Volume.from_name("yolo-finetune")

# Define the volume path inside the container
volume_path = Path("/root") / "data"

# Create the Modal app with the image and volume mounted
app = modal.App("yolo-finetune", image=image, volumes={volume_path: volume})

MINUTES = 60

TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"A100-80GB:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 4


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout= 240 * MINUTES,
)
def train(
    model_id: str,
    resume=True,
    quick_check=False,
):
    from ultralytics import YOLO

    volume.reload()  # Ensure volume is synced

    model_path = volume_path / "runs" / model_id
    model_path.mkdir(parents=True, exist_ok=True)

    data_path = volume_path / "dataset" / "dataset.yaml"
    best_weights = model_path / "weights" / "last.pt"

    # if resume and best_weights.exists():
        # If the checkpoint is resumable, load it.
    model = YOLO(str(best_weights))
    # else:
    #     # Otherwise, start from the base model.
    #     model = YOLO("yolov9c.pt")

    # If best.pt training is finished, you should force resume to False:
    # For example, you might force it here:

    model.train(
        data=data_path,
        fraction=0.04 if quick_check else 1.0,
        device=list(range(TRAIN_GPU_COUNT)),
        epochs=1 if quick_check else 80,  # set total epochs higher as desired
        batch=64,
        imgsz=320 if quick_check else 640,
        seed=117,
        workers=max(TRAIN_CPU_COUNT // TRAIN_GPU_COUNT, 1),
        cache=True,
        project=f"{volume_path}/runs",
        name=model_id,
        verbose=True,
        resume=resume,
    )


@app.cls(gpu="T4")
class Inference:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)

    @modal.method()
    def stream(self, model_id: str, image_files: list | None = None):
        """Counts the number of objects in a list of image files.
        Intended as a demonstration of high-throughput streaming inference."""
        import time

        completed, start = 0, time.monotonic_ns()
        for image_path in image_files:
            # completed += 1
            # if completed >= 200: 
            #     break
            results = self.model.predict(  # noqa: F841
                image_path,      # pass the path string
                half=True,       # use fp16
                save=True,  
                exist_ok=True, 
                verbose=False,
                project=f"{volume_path}/predictions/{model_id}",
                conf=0.4,
            )
            completed += 1

        elapsed_seconds = (time.monotonic_ns() - start) / 1e9
        print("Inferences per second:", round(completed / elapsed_seconds, 2))
        print(f"TOTAL INFERENCES: {completed}")


@app.function()
def infer(model_id: str):
    import os

    # Instantiate Inference using the best.pt weights
    inference = Inference(volume_path / "runs" / model_id / "weights" / "last.pt")
    
    test_dir = volume_path / "dataset" / "inference_snow"
    # Build a list of full paths for test images (filtering out non-image files)
    test_images_path = [str(test_dir / f) for f in os.listdir(str(test_dir)) if f.lower().endswith(".jpg")]

    print(f"{model_id}: Running streaming inferences on all images in the test set...")
    # Use .call() to run the remote method synchronously
    inference.stream.remote(model_id, test_images_path)

@app.function()
def create_video(model_id: str, fps: int = 20):
    import cv2
    
    # Convert the predictions_dir from string to Path
    predictions_dir = volume_path / "predictions" / model_id / "predict"
    # List all jpg files in the predictions directory and sort them
    image_files = sorted(list(predictions_dir.glob("*.jpg")))
    
    if not image_files:
        print("No predicted images found for video creation.")
        return f"{model_id}: No video created, no predicted images found."
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print("Unable to read the first image for video creation.")
        return f"{model_id}: Error reading first image for video creation."
    else: 
        print(f"Creating video from {len(image_files)} predicted images...")

    height, width, _ = first_frame.shape
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = str(volume_path / "output_video.mp4")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        if frame is None:
            print(f"Warning: Unable to read {img_file}")
            continue
        
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_video}")
    return f"{model_id}: Video saved to {output_video}"

