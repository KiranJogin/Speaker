from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.services.transcribe_service import transcribe_audio
import uvicorn

app = FastAPI(title="Speaker Diarization + Transcription API")

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup HTML templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the frontend HTML page
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Upload an audio file — get transcript with speaker labels
    """
    try:
        output = await transcribe_audio(file)
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
