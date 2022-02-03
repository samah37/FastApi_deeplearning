from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    response = {"success": False}
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    return response

