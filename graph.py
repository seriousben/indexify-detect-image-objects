import subprocess, os, platform, sys
from indexify import RemoteGraph, Graph, Image
from indexify.functions_sdk.data_objects import File
from indexify.functions_sdk.indexify_functions import (
    indexify_function,
)
from pydantic import BaseModel
import time

class ObjectDetectionResult(BaseModel):
    image: File

@indexify_function()
def object_detector(img: File) -> ObjectDetectionResult:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    nparr = np.frombuffer(img.data, np.uint8)
    image_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Load a model
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

    results = model(image_arr)

    # Process results list
    result = results[0]

    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    success, encoded_image = cv2.imencode('.webp', result.plot())
    output_img = File(data=encoded_image.tobytes(), mime_type="image/webp")

    return ObjectDetectionResult(image=output_img)

if __name__ == "__main__":
    g = Graph(name="object_detection_workflow", start_node=object_detector)

    with open('input-image.webp', 'rb') as reader:
        img = File(data=reader.read(), mime_type="image/webp")
        # Pass server_url="http://..." to point to indexify server. default is
        # http://localhost:8900

        if len(sys.argv) > 1 and sys.argv[1] == "deploy":
            g = RemoteGraph.deploy(g)
            print("Running graph remotely")
        else:
            print("Running graph locally run `python graph.py deploy` to run remotely")

        invocation_id = g.run(block_until_done=True, img=img)
        output = g.output(invocation_id, "object_detector")

        output_filepath = f"output-{time.time_ns()}-{invocation_id}.webp"
        with open(output_filepath, "wb") as binary_file:
            binary_file.write(output[0].image.data)

        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', output_filepath))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(output_filepath)
        else:                                   # linux variants
            subprocess.call(('xdg-open', output_filepath))