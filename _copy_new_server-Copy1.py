from fastapi import FastAPI, File, UploadFile, HTTPException
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

# ----------------------------- Initialisation -----------------------------
app = FastAPI()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement des poids et modèles
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights", "checkpoint0014.pth")
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")

if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
    raise FileNotFoundError("Le modèle GroundingDINO n'a pas été trouvé.")

if not os.path.exists(SAM_CHECKPOINT_PATH):
    os.system(f"wget -q -P {'weights'} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    device=DEVICE
)
SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
sam_predictor = SamPredictor(sam)

CLASSES = ['window', 'door']
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# ----------------------------- Fonctions principales -----------------------------
def enhance_class_name(class_names):
    return [f"all {class_name}s" for class_name in class_names]

def segment_with_sam(sam_predictor, image, xyxy, image_shape):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        mask_resized = cv2.resize(masks[index].astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        result_masks.append(mask_resized)
    return np.array(result_masks)

def creating_mask_with_dino(image):
    height, width, _ = image.shape
    
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(CLASSES),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    if not hasattr(detections, 'xyxy') or detections.xyxy is None or len(detections.xyxy) == 0:
        white_image = np.ones_like(image, dtype=np.uint8) * 255  # Image blanche
        white_buffer = BytesIO()
        Image.fromarray(white_image).save(white_buffer, format="PNG")
        return white_buffer.getvalue(), white_buffer.getvalue()
    
    # Correction de la taille et des positions
    detections.xyxy = np.clip(detections.xyxy, 0, [width - 1, height - 1, width - 1, height - 1]).astype(int)
    detections.mask = segment_with_sam(sam_predictor, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy, (height, width))
    
    mask_overlay = np.zeros((height, width), dtype=np.uint8)
    for mask in detections.mask:
        mask_overlay = np.maximum(mask_overlay, mask)
    
    mask_overlay_colored = np.stack([mask_overlay * 255] * 3, axis=-1)
    
    alpha = 0.5
    annotated_image = cv2.addWeighted(image, 1 - alpha, mask_overlay_colored, alpha, 0)
    
    mask_buffer = BytesIO()
    Image.fromarray(mask_overlay_colored).save(mask_buffer, format="PNG")
    annotated_buffer = BytesIO()
    Image.fromarray(annotated_image).save(annotated_buffer, format="PNG")
    
    return mask_buffer.getvalue(), annotated_buffer.getvalue()

# ----------------------------- Routes FastAPI -----------------------------
@app.post("/mask/png/new")
async def get_mask_as_png(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)
        mask_png, _ = creating_mask_with_dino(image_np)
        return StreamingResponse(BytesIO(mask_png), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des masques : {str(e)}")

@app.post("/mask/json/new")
async def get_mask_as_json(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)
        mask_png, annotated_png = creating_mask_with_dino(image_np)

        mask_base64 = base64.b64encode(mask_png).decode('utf-8')
        annotated_base64 = base64.b64encode(annotated_png).decode('utf-8')

        return JSONResponse(content={
            "mask": mask_base64,
            "result": annotated_base64
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des masques : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
