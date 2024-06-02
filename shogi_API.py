# 起動　uvicorn shogi_API:app --reload
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from io import BytesIO
from main import create_csa, create_board_instance, create_svg
#from app.main import create_csa, create_board_instance, create_svg


app = FastAPI()

@app.post("/process_shogi_image/")
async def process_shogi_image(file: UploadFile = File(...), process_type: str = Form(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    if process_type == "csa":
        csa_data = create_csa(image)
        return {"csa_data": csa_data}
    
    elif process_type == "board_instance":
        serialized_board = create_board_instance(image)
        return {"board_instance": serialized_board}
    
    elif process_type == "svg":
        svg_data = create_svg(image)
        return {"svg_data": svg_data}
    else:
        return {"error": "Invalid process type"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)