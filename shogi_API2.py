import os
import logging
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from io import BytesIO
from main import create_csa, create_board_instance, create_svg

# CUDAを無効にする
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# PyTorchのCUDA無効化を明示的に設定
import torch
torch.backends.cudnn.enabled = False

# ロギングの設定
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "将棋OCRです"}

@app.post("/process_shogi_image/")
async def process_shogi_image(file: UploadFile = File(...), process_type: str = Form(...)):
    logging.debug("Received image for processing")
    contents = await file.read()
    
    try:
        image = Image.open(BytesIO(contents))
        logging.debug("Image opened successfully")
    except Exception as e:
        logging.error(f"Failed to open image: {e}")
        return {"error": "Failed to open image"}
    
    try:
        if process_type == "board_instance":
            logging.debug("Processing type: board_instance")
            serialized_board = create_board_instance(image)
            logging.debug("Board instance created successfully")
            return {"board_instance": serialized_board}
        
        elif process_type == "svg":
            logging.debug("Processing type: svg")
            svg_data = create_svg(image)
            logging.debug("SVG data created successfully")
            return {"svg_data": svg_data}
        else:
            logging.debug("Processing type: csa")
            csa_data = create_csa(image)
            logging.debug("CSA data created successfully")
            return {"csa_data": csa_data}
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return {"error": f"Processing failed: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
