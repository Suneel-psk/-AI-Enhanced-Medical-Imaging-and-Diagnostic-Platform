from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import pyaudio
import wave
import pandas as pd
from p1f import (identify_disease,identify_disease_from_image, get_recorded_audio_input, process_image_input)
from fastapi.staticfiles import StaticFiles



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configuration for file upload
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/identify_disease_text", response_class=HTMLResponse)
async def identify_disease_text(request: Request, symptoms_text: str = Form(...)):
    if symptoms_text:
        disease_info = identify_disease(symptoms_text)
        return templates.TemplateResponse("results.html", {"request": request, "predicted_disease": disease_info})
    return RedirectResponse(url="/", status_code=303)

@app.post("/record_audio", response_class=HTMLResponse)
async def record_audio(request: Request, audio_file: UploadFile = File(...)):
    if audio_file.filename == '':
        return RedirectResponse(url="/", status_code=303)

    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await audio_file.read())

    text = get_recorded_audio_input(file_path)
    if text:
        disease_info = identify_disease(text)
        return templates.TemplateResponse("results.html", {"request": request, "predicted_disease": disease_info})

    return RedirectResponse(url="/", status_code=303)

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image(request: Request, image_file: UploadFile = File(...)):
    if image_file.filename == '':
        return RedirectResponse(url="/", status_code=303)

    file_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await image_file.read())

    try:
        # Process the image
        disease_info = process_image_input(file_path)
        logging.debug(f"Disease info returned: {disease_info}")

        if isinstance(disease_info, dict):
            # If disease_info contains detailed information, display it
            if 'message' in disease_info:
                # Message for normal cases
                return templates.TemplateResponse("results.html", {"request": request, "message": disease_info['message']})
            else:
                # Detailed disease information
                return templates.TemplateResponse("results.html", {"request": request, "predicted_disease": disease_info})
        else:
            # Handle the case where disease_info is not in expected format
            return templates.TemplateResponse("results.html", {"request": request, "message": "An error occurred while processing the image."})

    except Exception as e:
        logging.error(f"Exception in upload_image endpoint: {e}")
        return templates.TemplateResponse("results.html", {"request": request, "message": "An error occurred while processing the image."})

@app.post("/record_live_speech", response_class=HTMLResponse)
async def record_live_speech(request: Request, duration: int = Form(9)):  # Default to 9 seconds
    # Set up PyAudio for live recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
    frames = []
    for _ in range(int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a file
    temp_filename = 'live_speech.wav'
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

    # Recognize the audio
    text = get_recorded_audio_input(temp_filename)
    if text:
        disease_info = identify_disease(text)
        if isinstance(disease_info, dict):
            return templates.TemplateResponse("results.html", {"request": request, "predicted_disease": disease_info})
        else:
            # In case of no disease found or an error in identifying the disease
            error_message = "We couldn't identify any disease from your input. Please try again."
            return templates.TemplateResponse("results.html", {"request": request, "message": error_message})
    
    # In case of no text recognized
    error_message = "An error occurred while processing the audio input. Please try again."
    return templates.TemplateResponse("results.html", {"request": request, "message": error_message})
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, debug=True)
    #uvicorn hmapp:app --host 0.0.0.0 --port 5001 --reload
    #http://localhost:5001/ to access url in browser
