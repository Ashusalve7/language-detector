from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import base64, requests

from language_identifier import LanguageIdentifier

app = FastAPI()
identifier = LanguageIdentifier()

@app.post("/identify")
async def identify_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()
    result = identifier.process_file(file_bytes, ext)
    return JSONResponse(content=result)

@app.post("/identify/base64")
async def identify_base64(filename: str = Form(...), content: str = Form(...)):
    file_bytes = base64.b64decode(content)
    ext = filename.split(".")[-1].lower()
    result = identifier.process_file(file_bytes, ext)
    return JSONResponse(content=result)

@app.post("/identify/url")
async def identify_from_url(url: str = Form(...)):
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to download file"}
    file_bytes = response.content
    ext = url.split(".")[-1].lower().split("?")[0]
    result = identifier.process_file(file_bytes, ext)
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
