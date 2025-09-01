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
import whisper
from speechbrain.pretrained import SpeakerRecognition
import numpy as np
import librosa
import cv2   
import time
from scipy.stats import pearsonr
 

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
_speaker_model = None
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

def translate_audio_with_cli(audio_path: str, output_path: str, source_lang: str, target_lang: str, duration_factor: float = 1.0):
    """Translate audio using SeamlessExpressive CLI"""
    try:
        print(f"🎵 Translating with SeamlessExpressive CLI: {source_lang} -> {target_lang} at {duration_factor}x speed...")

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
                "--output_path", output_path,
                "--duration_factor", str(duration_factor) # ADDED DURATION FACTOR
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
                "--audio_root_dir", "",
                "--duration_factor", str(duration_factor) # ADDED DURATION FACTOR
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
                "--output_path", output_path,
                "--duration_factor", str(duration_factor) # ADDED DURATION FACTOR
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

def translate_audio(audio_path: str, output_path: str, source_lang: str, target_lang: str, preserve_style: bool, duration_factor: float):
    """Translate audio by dispatching to the correct model based on user choice."""
    
    if preserve_style:
        # User wants to preserve vocal style, use the expressive CLI model
        print("▶️ Dispatching to SeamlessExpressive model...")
        return translate_audio_with_cli(audio_path, output_path, source_lang, target_lang, duration_factor)
    
    else:
        # User wants a generic voice, use the standard M4Tv2 model
        print("▶️ Dispatching to standard SeamlessM4Tv2 model...")
        try:
            if processor is None or not hasattr(seamless_model, 'generate'):
                 raise Exception("Standard M4Tv2 model/processor not loaded or incompatible.")

            print(f"🔄 Using HuggingFace SeamlessM4Tv2 for generic voice: {source_lang} -> {target_lang}")
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            audio_inputs = processor(audios=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
            
            lang_map = { "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por", "zh": "cmn", "ja": "jpn", "ko": "kor", "ar": "arb", "hi": "hin", "ru": "rus"}
            tgt_lang_code = lang_map.get(target_lang, target_lang)

            audio_array = seamless_model.generate(**audio_inputs, tgt_lang=tgt_lang_code)[0].cpu().numpy().squeeze()
            
            torchaudio.save(output_path, torch.from_numpy(audio_array).unsqueeze(0), 16000)
            print(f"✅ HuggingFace translation saved")
            return True

        except Exception as e:
            print(f"❌ Error during M4Tv2 (generic) translation: {e}")
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

def get_speaker_model(device="cpu"):
    """Lazy load speaker recognition model"""
    global _speaker_model
    if _speaker_model is None:
        print("Loading SpeechBrain speaker recognition model...")
        _speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": device}
        )
    return _speaker_model

def calculate_speaker_similarity(source_audio_path: str, translated_audio_path: str, device="cpu") -> float:
    """Calculate speaker similarity using SpeechBrain"""
    print(f"Calculating Speaker Similarity...")
    try:
        spkrec_model = get_speaker_model(device)
        score, _ = spkrec_model.verify_files(source_audio_path, translated_audio_path)
        return float(score.squeeze())
    except Exception as e:
        print(f"Speaker similarity calculation failed: {e}")
        return 0.0

def calculate_acoustic_features(audio_path: str) -> dict:
    """Extract acoustic features using librosa"""
    print(f"Calculating Acoustic Features...")
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate F0 (pitch)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0) if np.any(f0) else 0.0
        f0_std = np.nanstd(f0) if np.any(f0) else 0.0

        # Calculate RMS energy and intensity
        rms_energy = librosa.feature.rms(y=y)
        intensity_mean = np.mean(rms_energy)
        intensity_std = np.std(rms_energy)
        
        # Calculate RMS and Peak amplitudes
        rms_mean = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))

        return {
            "f0_mean": float(f0_mean),
            "f0_std": float(f0_std),
            "intensity_mean": float(intensity_mean),
            "intensity_std": float(intensity_std),
            "rms_mean": float(rms_mean),
            "peak_amplitude": float(peak_amplitude)
        }
    except Exception as e:
        print(f"Failed to calculate acoustic features: {e}")
        return {"f0_mean": 0, "f0_std": 0, "intensity_mean": 0, "intensity_std": 0, "rms_mean": 0, "peak_amplitude": 0}

def get_audio_emotion(audio_path: str) -> str:
    """Analyze emotion from audio using acoustic features"""
    try:
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        if len(y) == 0:
            return "neutral"
        
        # Calculate energy and pitch variation as emotion indicators
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)
        
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_f0 = f0[~np.isnan(f0)]
        pitch_variation = np.std(voiced_f0) if len(voiced_f0) > 0 else 0.0
        
        # Simple classification based on energy and pitch variation
        if energy > 0.02 and pitch_variation > 20:
            emotion = "excited"
        elif energy < 0.01:
            emotion = "calm"
        elif pitch_variation > 15:
            emotion = "expressive"
        else:
            emotion = "neutral"
        
        return emotion
        
    except Exception as e:
        print(f"Error analyzing audio emotion: {e}")
        return "neutral"

def calculate_lip_sync_quality(video_path: str, audio_path: str) -> float:
    """Calculate lip-sync quality using MediaPipe landmarks and audio correlation"""
    print("Calculating lip-sync quality...")
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Extract mouth openings from video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return 0.0
            
        mouth_openings = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                upper_lip = landmarks.landmark[13]  # Upper lip center
                lower_lip = landmarks.landmark[14]  # Lower lip center
                
                mouth_opening = np.sqrt(
                    (upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2
                )
                mouth_openings.append(mouth_opening)
            else:
                mouth_openings.append(0.0)
                
        cap.release()
        
        # Extract audio envelope
        y, sr = librosa.load(audio_path, sr=16000)
        frame_length = int(sr / fps)
        
        audio_envelope = []
        for i in range(0, len(y), frame_length):
            chunk = y[i:i + frame_length]
            if len(chunk) > 0:
                audio_envelope.append(np.mean(np.abs(chunk)))
                
        # Calculate correlation
        min_len = min(len(mouth_openings), len(audio_envelope))
        if min_len > 1:
            correlation, _ = pearsonr(mouth_openings[:min_len], audio_envelope[:min_len])
            return float(correlation) if not np.isnan(correlation) else 0.0
        else:
            return 0.0
            
    except Exception as e:
        print(f"Lip-sync calculation failed: {e}")
        return 0.0

def get_visual_emotion(video_path: str) -> str:
    """Get visual emotion using DeepFace"""
    print("Calculating visual emotion...")
    try:
        from deepface import DeepFace
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_index = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return "error_or_no_face"

        # Save temporary frame
        temp_frame_path = f"temp_frame_{int(time.time())}.jpg"
        cv2.imwrite(temp_frame_path, frame)

        analysis = DeepFace.analyze(
            img_path=temp_frame_path, 
            actions=['emotion'], 
            enforce_detection=True,
            silent=True
        )
        
        import os
        os.remove(temp_frame_path)  # Cleanup
        
        if analysis and isinstance(analysis, list):
            return analysis[0]['dominant_emotion']
        return "no_face_detected"
        
    except Exception as e:
        print(f"Visual emotion classification failed: {e}")
        return "error_or_no_face"

def calculate_visual_identity_similarity(original_video_path: str, translated_video_path: str) -> float:
    """Calculate facial identity similarity using DeepFace"""
    print("Calculating visual identity similarity...")
    try:
        from deepface import DeepFace
        
        # Extract middle frames from both videos
        cap1 = cv2.VideoCapture(original_video_path)
        frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_count1 // 2)
        ret1, frame1 = cap1.read()
        cap1.release()
        
        cap2 = cv2.VideoCapture(translated_video_path)
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count2 // 2)
        ret2, frame2 = cap2.read()
        cap2.release()
        
        if not ret1 or not ret2:
            return 0.0
            
        # Save temporary frames
        frame1_path = f"temp_original_{int(time.time())}.jpg"
        frame2_path = f"temp_translated_{int(time.time())}.jpg"
        cv2.imwrite(frame1_path, frame1)
        cv2.imwrite(frame2_path, frame2)
        
        try:
            result = DeepFace.verify(
                frame1_path, 
                frame2_path,
                model_name="ArcFace",
                enforce_detection=False
            )
            distance = result['distance']
            similarity = 1.0 / (1.0 + distance)
            
            import os
            os.remove(frame1_path)
            os.remove(frame2_path)
            
            return float(similarity)
        except:
            import os
            os.remove(frame1_path)
            os.remove(frame2_path)
            return 0.0
            
    except Exception as e:
        print(f"Visual identity similarity calculation failed: {e}")
        return 0.0

def calculate_visual_quality_lpips(original_video_path: str, translated_video_path: str) -> float:
    """Calculate visual quality using LPIPS if available, otherwise SSIM"""
    print("Calculating visual quality...")
    try:
        # Try LPIPS first
        try:
            import lpips
            import torch
            
            lpips_model = lpips.LPIPS(net='alex')
            
            # Extract frames
            cap1 = cv2.VideoCapture(original_video_path)
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_count1 // 2)
            ret1, frame1 = cap1.read()
            cap1.release()
            
            cap2 = cv2.VideoCapture(translated_video_path)
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count2 // 2)
            ret2, frame2 = cap2.read()
            cap2.release()
            
            if not ret1 or not ret2:
                return 0.0
                
            def preprocess_frame(frame):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
                tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
                tensor = (tensor / 255.0) * 2.0 - 1.0
                return tensor.unsqueeze(0)
            
            tensor1 = preprocess_frame(frame1)
            tensor2 = preprocess_frame(frame2)
            
            with torch.no_grad():
                lpips_distance = lpips_model(tensor1, tensor2)
            
            return float(lpips_distance.item())
            
        except:
            # Fallback to SSIM
            from skimage.metrics import structural_similarity as ssim
            
            cap1 = cv2.VideoCapture(original_video_path)
            frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_count1 // 2)
            ret1, frame1 = cap1.read()
            cap1.release()
            
            cap2 = cv2.VideoCapture(translated_video_path)
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_count2 // 2)
            ret2, frame2 = cap2.read()
            cap2.release()
            
            if not ret1 or not ret2:
                return 0.0
                
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            height, width = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (width, height))
            gray2 = cv2.resize(gray2, (width, height))
            
            ssim_score = ssim(gray1, gray2)
            return float(ssim_score)
            
    except Exception as e:
        print(f"Visual quality calculation failed: {e}")
        return 0.0

def get_deepfake_score(video_path: str) -> float:
    """Basic deepfake detection using video quality heuristics"""
    print("Running basic deepfake detection...")
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return 0.5
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            score = 0.7
        elif laplacian_var < 200:
            score = 0.4
        else:
            score = 0.2
            
        return float(score)
        
    except Exception as e:
        print(f"Basic deepfake detection failed: {e}")
        return 0.5

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
    target_lang: str = Form(...),
    preserve_style: bool = Form(...),
    duration_factor: float = Form(...)
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
        
        # Get original video duration
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        original_video_duration = frame_count / fps
        cap.release()
        
        # Get original audio duration
        import librosa
        y_orig, sr = librosa.load(audio_path, sr=None)
        original_audio_duration = len(y_orig) / sr
        
        print(f"ORIGINAL VIDEO DURATION: {original_video_duration:.2f}s")
        print(f"ORIGINAL AUDIO DURATION: {original_audio_duration:.2f}s")
        
        # Translate audio with timing
        import time
        translation_start = time.time()
        translated_audio_path = os.path.join(temp_dir, f"{session_id}_translated.wav")
        if not translate_audio(audio_path, translated_audio_path, source_lang, target_lang, preserve_style, duration_factor):
            return {"error": "Failed to translate audio"}
        translation_end = time.time()
        
        # Get translated audio duration
        y_trans, sr = librosa.load(translated_audio_path, sr=None)
        translated_audio_duration = len(y_trans) / sr
        
        processing_time = translation_end - translation_start
        duration_ratio = translated_audio_duration / original_audio_duration
        
        print(f"TRANSLATION PROCESSING TIME: {processing_time:.2f}s")
        print(f"TRANSLATED AUDIO DURATION: {translated_audio_duration:.2f}s")
        print(f"DURATION RATIO: {duration_ratio:.3f}")
        print(f"AUDIO-VIDEO SYNC GAP: {abs(original_video_duration - translated_audio_duration):.2f}s")
        
        if translated_audio_duration < original_video_duration:
            print(f"WARNING: Audio finishes {original_video_duration - translated_audio_duration:.2f}s before video ends")
        
        # Generate transcript from translated audio
        try:
            model = whisper.load_model("base")
            result = model.transcribe(translated_audio_path)
            transcript = result["text"]
            print(f"TRANSCRIPT: {transcript}")
        except Exception as e:
            print(f"Transcription failed: {e}")
        
        # Calculate speaker similarity
        speaker_sim = calculate_speaker_similarity(audio_path, translated_audio_path, device)
        print(f"SPEAKER SIMILARITY: {speaker_sim}")
        
        # Calculate acoustic features for both audios
        original_features = calculate_acoustic_features(audio_path)
        translated_features = calculate_acoustic_features(translated_audio_path)
        
        # Calculate differences
        f0_diff = abs(original_features["f0_mean"] - translated_features["f0_mean"])
        intensity_diff = abs(original_features["intensity_mean"] - translated_features["intensity_mean"])
        
        print(f"ORIGINAL RMS: {original_features['rms_mean']:.4f}, Peak: {original_features['peak_amplitude']:.4f}")
        print(f"TRANSLATED RMS: {translated_features['rms_mean']:.4f}, Peak: {translated_features['peak_amplitude']:.4f}")
        print(f"F0 DIFFERENCE: {f0_diff:.2f}")
        print(f"INTENSITY DIFFERENCE: {intensity_diff:.4f}")
        
        # Get audio emotions
        original_emotion = get_audio_emotion(audio_path)
        translated_emotion = get_audio_emotion(translated_audio_path)
        emotion_consistent = original_emotion == translated_emotion
        
        print(f"AUDIO EMOTION - Original: {original_emotion}, Translated: {translated_emotion}, Consistent: {emotion_consistent}")
        
        # Combine video with translated audio
        output_path = os.path.join(temp_dir, f"{session_id}_output.mp4")
        if not combine_video_audio(video_path, translated_audio_path, output_path):
            return {"error": "Failed to combine video with translated audio"}
        
        # Comprehensive Visual and Performance Analysis
        try:
            # Visual Emotions
            original_visual_emotion = get_visual_emotion(video_path)
            translated_visual_emotion = get_visual_emotion(output_path)
            visual_emotion_consistent = original_visual_emotion == translated_visual_emotion
            
            print(f"VISUAL EMOTION - Original: {original_visual_emotion}, Translated: {translated_visual_emotion}, Consistent: {visual_emotion_consistent}")
            
            # Lip Sync Quality
            lip_sync_score = calculate_lip_sync_quality(output_path, translated_audio_path)
            print(f"LIP SYNC SCORE: {lip_sync_score:.4f}")
            
            # Visual Identity Similarity
            visual_identity = calculate_visual_identity_similarity(video_path, output_path)
            print(f"VISUAL IDENTITY SIMILARITY: {visual_identity:.4f}")
            
            # Visual Quality (LPIPS)
            visual_quality = calculate_visual_quality_lpips(video_path, output_path)
            print(f"VISUAL QUALITY (LPIPS): {visual_quality:.4f}")
            
            # Deepfake Score
            deepfake_score = get_deepfake_score(output_path)
            print(f"DEEPFAKE SCORE: {deepfake_score:.1f}")
            
            # Temporal Analysis
            temporal_compression_ratio = translated_audio_duration / original_audio_duration
            real_time_factor = processing_time / original_video_duration
            pause_count = 0  # SeamlessM4T doesn't analyze pauses like your system
            
            print(f"TEMPORAL COMPRESSION RATIO: {temporal_compression_ratio:.3f}")
            print(f"REAL-TIME FACTOR: {real_time_factor:.2f}x")
            print(f"PAUSE COUNT: {pause_count}")
            print(f"ORIGINAL DURATION: {original_video_duration:.2f}s")
            print(f"MAPPED DURATION: {translated_audio_duration:.2f}s (SeamlessM4T doesn't remap)")
            
        except Exception as e:
            print(f"Comprehensive analysis failed: {e}")
        
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

if __name__ == "__main__":
    import uvicorn
    
    # Define a static port to match the frontend and avoid conflicts
    port = 8004
    
    print(f"🚀 Starting server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)