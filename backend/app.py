from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
import torchaudio
import subprocess
import tempfile
import os
import shutil
from pathlib import Path
import uuid
import socket
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="SeamlessExpressive Video Translator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001", # Add this line
        "http://127.0.0.1:3001"  # Good practice to add this too
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
seamless_model = None
processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_seamless_model():
    """Load SeamlessExpressive model - try multiple approaches"""
    global seamless_model, processor
    
    print("🚀 Starting model loading process...")
    
    # Approach 1: Try SeamlessExpressive CLI interface
    try:
        print("🔄 Attempting to load SeamlessExpressive with CLI interface...")
        
        # Check if seamless_communication is available
        import seamless_communication
        print(f"✅ seamless_communication found at: {seamless_communication.__file__}")
        
        # Try to import the CLI
        try:
            from seamless_communication.cli.expressivity import predict as expressivity_predict
            print("✅ SeamlessExpressive CLI interface loaded!")
            seamless_model = "seamless_expressivity"  # Flag that we have the CLI
            return
        except ImportError as e:
            print(f"⚠️  CLI import failed: {e}")
            
        # Alternative: try the general inference module
        try:
            from seamless_communication.inference import Translator
            model_dir = os.path.expanduser("~/Downloads/SeamlessExpressive")
            if os.path.exists(model_dir):
                seamless_model = Translator(
                    model_name_or_card=os.path.join(model_dir, "m2m_expressive_unity.pt"),
                    vocoder_name_or_card=os.path.join(model_dir, "pretssel_melhifigan_wm-16khz.pt"),
                    device=device,
                )
                print(f"✅ SeamlessExpressive loaded with your model files on {device}")
                return
            else:
                print("⚠️  Model directory not found...")
        except ImportError as e:
            print(f"⚠️  General inference import failed: {e}")
            
    except ImportError:
        print("⚠️  seamless_communication not available")
    except Exception as e:
        print(f"⚠️  Error loading SeamlessExpressive: {e}")
    
    # Approach 2: Fallback to HuggingFace SeamlessM4Tv2
    try:
        print("🔄 Falling back to HuggingFace SeamlessM4Tv2...")
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        seamless_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
        seamless_model.to(device)
        
        print(f"✅ SeamlessM4Tv2 loaded from HuggingFace on {device}")
        print("ℹ️  Note: Using SeamlessM4Tv2 instead of SeamlessExpressive")
        return
        
    except Exception as e:
        print(f"❌ All loading methods failed: {e}")
        seamless_model = None

def extract_audio_from_video(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", 
            audio_path, "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False

def translate_audio_with_cli(audio_path: str, output_path: str, source_lang: str, target_lang: str):
    """Translate audio using SeamlessExpressive CLI"""
    try:
        print(f"🎵 Translating with SeamlessExpressive CLI: {source_lang} -> {target_lang}...")
        
        # Language code mapping for CLI
        lang_map = {
            "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", 
            "pt": "por", "zh": "cmn", "ja": "jpn", "ko": "kor", "ar": "arb", 
            "hi": "hin", "ru": "rus"
        }
        
        src_lang = lang_map.get(source_lang, source_lang)
        tgt_lang = lang_map.get(target_lang, target_lang)
        
        # Try multiple CLI approaches
        model_dir = os.path.expanduser("~/Downloads/SeamlessExpressive")
        
        # Approach 1: Try the direct expressivity_predict command
        try:
            cmd = [
                "expressivity_predict",
                audio_path,
                "--tgt_lang", tgt_lang,
                "--model_name", "seamless_expressivity",
                "--vocoder_name", "vocoder_pretssel", 
                "--output_path", output_path
            ]
            
            if os.path.exists(model_dir):
                cmd.extend(["--gated-model-dir", model_dir])
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ SeamlessExpressive CLI translation saved (method 1)")
            return True
            
        except subprocess.CalledProcessError as e1:
            print(f"⚠️  Method 1 failed: {e1}")
            print(f"⚠️  Method 1 stderr: {e1.stderr}")
            print(f"⚠️  Method 1 stdout: {e1.stdout}")
            
        # Approach 2: Try the pretssel_inference.py script
        try:
            # Create a temporary TSV file for the inference script
            temp_tsv = audio_path.replace('.wav', '.tsv')
            with open(temp_tsv, 'w') as f:
                f.write(f"id\taudio\n")
                f.write(f"test\t{audio_path}\n")
            
            cmd = [
                "python", "-m", "seamless_communication.cli.expressivity.evaluate.pretssel_inference",
                temp_tsv,
                "--task", "s2st",
                "--tgt_lang", tgt_lang,
                "--model_name", "seamless_expressivity",
                "--vocoder_name", "vocoder_pretssel",
                "--output_path", os.path.dirname(output_path),
                "--audio_root_dir", ""
            ]
            
            if os.path.exists(model_dir):
                cmd.extend(["--gated-model-dir", model_dir])
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # The output might be in a different location, try to find it
            output_dir = os.path.dirname(output_path)
            for file in os.listdir(output_dir):
                if file.endswith('.wav') and 'hypo' in file:
                    shutil.move(os.path.join(output_dir, file), output_path)
                    break
            
            print(f"✅ SeamlessExpressive CLI translation saved (method 2)")
            return True
            
        except subprocess.CalledProcessError as e2:
            print(f"⚠️  Method 2 failed: {e2}")
            print(f"⚠️  Method 2 stderr: {e2.stderr}")
            print(f"⚠️  Method 2 stdout: {e2.stdout}")
            
        # Approach 3: Try with python -m seamless_communication.cli.expressivity
        try:
            cmd = [
                "python", "-m", "seamless_communication.cli.expressivity",
                "predict",
                audio_path,
                "--tgt_lang", tgt_lang,
                "--model_name", "seamless_expressivity",
                "--vocoder_name", "vocoder_pretssel", 
                "--output_path", output_path
            ]
            
            if os.path.exists(model_dir):
                cmd.extend(["--gated-model-dir", model_dir])
                
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ SeamlessExpressive CLI translation saved (method 3)")
            return True
            
        except subprocess.CalledProcessError as e3:
            print(f"⚠️  Method 3 failed: {e3}")
            print(f"⚠️  Method 3 stderr: {e3.stderr}")
            print(f"⚠️  Method 3 stdout: {e3.stdout}")
            print(f"❌ All CLI methods failed")
            return False
        
    except Exception as e:
        print(f"❌ Error in CLI translation: {e}")
        return False

def translate_audio(audio_path: str, output_path: str, source_lang: str, target_lang: str):
    """Translate audio using available method"""
    try:
        if seamless_model is None:
            raise Exception("No translation model loaded")
        
        print(f"🎵 Translating {source_lang} -> {target_lang}...")
        
        # Check if we're using SeamlessExpressive CLI
        if isinstance(seamless_model, str) and seamless_model == "seamless_expressivity":
            return translate_audio_with_cli(audio_path, output_path, source_lang, target_lang)
        
        # Load audio for other methods
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Language code mapping
        lang_map = {
            "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", 
            "pt": "por", "zh": "cmn", "ja": "jpn", "ko": "kor", "ar": "arb", 
            "hi": "hin", "ru": "rus"
        }
        
        src_lang = lang_map.get(source_lang, source_lang)
        tgt_lang = lang_map.get(target_lang, target_lang)
        
        # Check if we're using SeamlessExpressive Translator
        if hasattr(seamless_model, 'predict'):
            # SeamlessExpressive approach
            print(f"🔄 Using SeamlessExpressive S2ST: {src_lang} -> {tgt_lang}")
            translated_text, translated_speech, _ = seamless_model.predict(
                input=waveform.squeeze(),
                task_str="S2ST",
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            
            if translated_speech is not None:
                if len(translated_speech.shape) == 1:
                    translated_speech = translated_speech.unsqueeze(0)
                torchaudio.save(output_path, translated_speech.cpu(), 16000)
                print(f"✅ SeamlessExpressive translation saved")
                return True
                
        elif processor is not None:
            # HuggingFace approach
            print(f"🔄 Using HuggingFace SeamlessM4Tv2: {src_lang} -> {tgt_lang}")
            audio_inputs = processor(audios=waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            audio_inputs = audio_inputs.to(device)
            
            audio_array = seamless_model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
            
            if len(audio_array.shape) == 1:
                audio_array = torch.from_numpy(audio_array).unsqueeze(0)
            else:
                audio_array = torch.from_numpy(audio_array)
                
            torchaudio.save(output_path, audio_array, 16000)
            print(f"✅ HuggingFace translation saved")
            return True
        
        print("❌ No compatible translation method found")
        return False
        
    except Exception as e:
        print(f"❌ Error in translation: {e}")
        return False

def combine_video_audio(video_path: str, audio_path: str, output_path: str):
    """Combine original video with translated audio"""
    try:
        cmd = [
            "ffmpeg", "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
            output_path, "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e}")
        return False

def cleanup_temp_dir(temp_dir: str):
    """Clean up temporary directory"""
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_seamless_model()

@app.get("/")
async def root():
    return {"message": "SeamlessExpressive Video Translator API"}

@app.get("/health")
async def health_check():
    model_type = None
    if seamless_model is not None:
        if isinstance(seamless_model, str) and seamless_model == "seamless_expressivity":
            model_type = "SeamlessExpressive CLI"
        elif hasattr(seamless_model, 'predict'):
            model_type = "SeamlessExpressive"
        else:
            model_type = "SeamlessM4Tv2"
    
    return {
        "status": "healthy",
        "model_loaded": seamless_model is not None,
        "model_type": model_type,
        "processor_loaded": processor is not None,
        "device": str(device)
    }

@app.post("/translate")
async def translate_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...)
):
    """Translate video using available model"""
    
    if seamless_model is None:
        return {"error": "No translation model loaded"}
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    session_id = str(uuid.uuid4())
    
    try:
        # Save uploaded video
        video_path = os.path.join(temp_dir, f"{session_id}_input.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        # Extract audio from video
        audio_path = os.path.join(temp_dir, f"{session_id}_audio.wav")
        if not extract_audio_from_video(video_path, audio_path):
            return {"error": "Failed to extract audio from video"}
        
        # Translate audio
        translated_audio_path = os.path.join(temp_dir, f"{session_id}_translated.wav")
        if not translate_audio(audio_path, translated_audio_path, source_lang, target_lang):
            return {"error": "Failed to translate audio"}
        
        # Combine video with translated audio
        output_path = os.path.join(temp_dir, f"{session_id}_output.mp4")
        if not combine_video_audio(video_path, translated_audio_path, output_path):
            return {"error": "Failed to combine video with translated audio"}
        
        # Return the translated video
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"translated_{video.filename}"
        )
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"error": f"Translation failed: {str(e)}"}

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "zh", "name": "Chinese (Mandarin)"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ru", "name": "Russian"},
        ]
    }

def find_free_port(start_port=8002):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + 20):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue
    return None

# At the end of your app.py file
if __name__ == "__main__":
    import uvicorn
    
    # Define a static port
    port = 8004
    
    print(f"🚀 Starting server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)