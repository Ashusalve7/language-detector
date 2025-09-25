from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, os
from language_identifier import LanguageIdentifier

app = FastAPI()
identifier = LanguageIdentifier()

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = identifier.process_file(temp_path)
    os.remove(temp_path)

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
