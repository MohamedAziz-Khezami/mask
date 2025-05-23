from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import os
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from groundingdino.util.inference import Model
import supervision as sv
import torch
import io
import traceback

# ----------------------------- Initialisation -----------------------------
app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to models
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATHS = {
    "new": os.path.join("weights", "checkpoint0014.pth"),
    "old": os.path.join("weights", "groundingdino_swint_ogc.pth"),
}
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")

# Download weights if missing
for version, path in GROUNDING_DINO_CHECKPOINT_PATHS.items():
    if not os.path.exists(path):
        url = (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            if version == "old"
            else None
        )
        if url:
            os.system(f"wget -q -P {os.path.join('weights')} {url}")

if not os.path.exists(SAM_CHECKPOINT_PATH):
    os.system(
        f"wget -q -P {os.path.join('weights')} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )

# Load models
models = {}
for version, checkpoint_path in GROUNDING_DINO_CHECKPOINT_PATHS.items():
    models[version] = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=checkpoint_path,
        device=DEVICE,
    )

SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
sam_predictor = SamPredictor(sam)

# Define the classes; assumed zero-indexed mapping:
CLASSES = ["ceiling", "window", "door", "roof"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# ----------------------------- Helper Functions -----------------------------
def enhance_class_name(class_names):
    # In this example we don't modify the names.
    return class_names

def segment_with_sam(sam_predictor, image, xyxy):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def create_binary_mask(bounding_boxes, H, W):
    """
    Creates a binary mask with detected objects in black on a white background.
    """
    binary_mask = np.ones((H, W), dtype=np.uint8) * 255
    for x_min, y_min, x_max, y_max in bounding_boxes:
        binary_mask[y_min:y_max, x_min:x_max] = 0
    binary_mask_pil = Image.fromarray(binary_mask)
    mask_binary_buffer = BytesIO()
    binary_mask_pil.save(mask_binary_buffer, format="PNG")
    return mask_binary_buffer.getvalue()
def creating_mask_with_dino(image, model_version):
    try:
        grounding_dino_model = models[model_version]
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(CLASSES),
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        
        detections.xyxy = detections.xyxy.astype(int)
        
        # Generate segmentation masks using SAM
        detections.mask = segment_with_sam(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        # Fallback if no mask is detected
        if len(detections.mask) == 0:
            white_image = np.ones_like(image, dtype=np.uint8) * 255
            white_buffer = BytesIO()
            Image.fromarray(white_image).save(white_buffer, format="PNG")

            original_buffer = BytesIO()
            Image.fromarray(image).save(original_buffer, format="PNG")
            return white_buffer.getvalue(), original_buffer.getvalue()
        
        # Create a blank overlay for annotation
        mask_annotator = sv.MaskAnnotator()
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        
        # Define a mapping from class names to colors (BGR)
        class_colors = {
            "wall": None,      
            "window": (202, 163, 96),     #ORION color
            "door": (202, 163, 96),       
            "ceiling": None,  
            "unknown": (202, 163, 96)   # None for transparency
        }
        
        # Iterate over each detection mask and apply the corresponding color.
        for mask, class_id in zip(detections.mask, detections.class_id):
            if class_id is None or class_id >= len(CLASSES):
                class_label = "unknown"
            else:
                class_label = CLASSES[class_id]
            
            color = class_colors.get(class_label)
            if color is not None:
                mask_indices = mask.astype(bool)
                mask_overlay[mask_indices] = color
        
        # Blend the colored overlay with the original image.
        alpha = 0.5  # transparency factor
        annotated_image = image.copy()
        overlay_indices = np.any(mask_overlay != 0, axis=-1)
        annotated_image[overlay_indices] = (
            alpha * mask_overlay[overlay_indices] + (1 - alpha) * image[overlay_indices]
        ).astype(np.uint8)
                
        annotated_image_pil = Image.fromarray(annotated_image)
        annotated_buffer = BytesIO()
        annotated_image_pil.save(annotated_buffer, format="PNG")
        
        H, W = image.shape[:2]
        bounding_boxes = detections.xyxy.tolist()
        
        binary_mask_data = create_binary_mask(bounding_boxes, H, W)
        
        return binary_mask_data, annotated_buffer.getvalue()
    
    except Exception as e:
        print("Erreur lors de l'exécution de creating_mask_with_dino:")
        print(traceback.format_exc())
        return None, None

# ----------------------------- FastAPI Routes -----------------------------
@app.post("/mask/png")
async def get_mask_as_png(file: UploadFile = File(...), model_version: str = Query("new", enum=["new", "old"])):
    try:
        image = Image.open(file.file)
        image_np = np.array(image)
        mask_png, _ = creating_mask_with_dino(image_np, model_version)
        return StreamingResponse(BytesIO(mask_png), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des masques : {str(e)}")

@app.post("/mask/json")
async def get_mask_as_json(file: UploadFile = File(...), model_version: str = Query("new", enum=["new", "old"])):
    try:
        image = Image.open(file.file)
        image_np = np.array(image)
        mask_png, annotated_png = creating_mask_with_dino(image_np, model_version)
        return JSONResponse(
            content={
                "mask": base64.b64encode(mask_png).decode("utf-8"),
                "result": base64.b64encode(annotated_png).decode("utf-8"),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des masques : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)