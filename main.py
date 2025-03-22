from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "ultralytics~=8.3.93",
        "opencv-python~=4.10.0",
        "term-image==0.7.1"
    ])
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
    timeout=60 * MINUTES,
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
    best_weights = model_path / "weights" / "best.pt"
    
    if resume and best_weights.exists():
        # If the checkpoint is resumable, load it.
        model = YOLO(str(best_weights))
    else:
        # Otherwise, start from the base model.
        model = YOLO("yolov9c.pt")
    
    # If best.pt training is finished, you should force resume to False:
    # For example, you might force it here:
    resume = False

    model.train(
        data=data_path,
        fraction=0.04 if quick_check else 1.0,
        device=list(range(TRAIN_GPU_COUNT)),
        epochs=1 if quick_check else 20,  # set total epochs higher as desired
        batch=32,
        imgsz=320 if quick_check else 640,
        seed=117,
        workers=max(TRAIN_CPU_COUNT // TRAIN_GPU_COUNT, 1),
        cache=False,
        project=f"{volume_path}/runs",
        name=model_id,
        verbose=True,
        resume=resume
    )


@app.cls()
class Inference:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)
    '''
    @modal.method()
    def predict(self, model_id: str, image_path: str, display: bool = False):
        """A simple method for running inference on one image at a time."""
        results = self.model.predict(
            image_path,
            half=True,  # use fp16
            save=True,
            exist_ok=True,
            project=f"{volume_path}/predictions/{model_id}",
        )
        if display:
            from term_image.image import from_file

            terminal_image = from_file(results[0].path)
            terminal_image.draw()
        # you can view the output file via the Volumes UI in the Modal dashboard -- https://modal.com/storage
    '''


@app.function()
def infer(model_id: str):  
    import os
    inference = Inference(
        volume_path / "runs" / model_id / "weights" / "best.pt"
    )
    # List only image files in the test folder
    test_dir = volume_path / "dataset" / "test"
    test_images = [f for f in os.listdir(str(test_dir)) if f.lower().endswith(".jpg")]

    for ii, img_name in enumerate(test_images):
        print(f"{model_id}: Single image prediction on image {img_name}")
        # Construct the full path to the image file
        full_image_path = str(test_dir / img_name)
        inference.predict.remote(
            model_id=model_id,
            image_path=full_image_path,
            display=(ii == 0)
        )
        if ii >= 4:
            break