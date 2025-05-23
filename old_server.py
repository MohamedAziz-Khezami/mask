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
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")

if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
    os.system(f"wget -q -P {os.path.join('weights')} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")

if not os.path.exists(SAM_CHECKPOINT_PATH):
    os.system(f"wget -q -P {os.path.join('weights')} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

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

def segment_with_sam(sam_predictor, image, xyxy):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def creating_mask_with_dino(image):
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(CLASSES),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    detections.xyxy = detections.xyxy.astype(int)
    detections.mask = segment_with_sam(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    if len(detections.mask) == 0:
        white_image = np.ones_like(image) * 255  # Image blanche
        white_buffer = BytesIO()
        Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB)).save(white_buffer, format="PNG")
        return white_buffer.getvalue(), white_buffer.getvalue()


    custom_cmap = LinearSegmentedColormap.from_list("white_black", ["white", "black"], N=256)

    fig, axes = plt.subplots(
        nrows=int(np.ceil(np.sqrt(len(detections.mask)))),
        ncols=int(np.ceil(np.sqrt(len(detections.mask)))),
        figsize=(16, 16)
    )
    axes = np.array(axes).flatten()
    for idx, ax in enumerate(axes):
        if idx < len(detections.mask):
            ax.imshow(detections.mask[idx], cmap=custom_cmap)
            ax.axis('off')
        else:
            ax.axis('off')

    fig.patch.set_facecolor("white")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor="white")
    buf.seek(0)

    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    annotated_buffer = BytesIO()
    annotated_image_pil.save(annotated_buffer, format="PNG")

    plt.close(fig)

    return buf.getvalue(), annotated_buffer.getvalue()

# ----------------------------- Routes FastAPI -----------------------------
@app.post("/mask/png/old")
async def get_mask_as_png(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image_np = np.array(image)
        mask_png, _ = creating_mask_with_dino(image_np)
        return StreamingResponse(BytesIO(mask_png), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération des masques : {str(e)}")

@app.post("/mask/json/old")
async def get_mask_as_json(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
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


@app.post("/mask/json/old")
async def get_mask_as_json(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
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
    uvicorn.run(app, host="0.0.0.0", port=5000)
