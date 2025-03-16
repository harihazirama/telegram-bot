import os
import json
import wave
import vosk
import subprocess
import logging
from pydub import AudioSegment

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def convert_audio(audio_path):    
    """Convert non-WAV files to WAV and ensure correct format."""
    temp_output = "output.wav"
    try:
        # Convert non-WAV files to WAV
        if not audio_path.endswith(".wav"):
            logger.info(f"Converting {audio_path} to WAV...")
            audio = AudioSegment.from_file(audio_path)
            audio_path = audio_path.rsplit(".", 1)[0] + ".wav"
            audio.export(audio_path, format="wav")

        # Normalize volume
        audio = AudioSegment.from_wav(audio_path)
        audio = audio + 10  # Increase volume by 10 dB
        audio.export(audio_path, format="wav")
        logger.info("Audio volume normalized.")

        # Convert to mono and 16kHz if needed
        if audio.channels != 1 or audio.frame_rate != 16000:
            logger.info("Converting audio to mono and 16kHz sample rate...")
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", temp_output
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio_path = temp_output

        logger.info(f"Audio conversion completed: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Error in audio conversion: {e}", exc_info=True)
        return None

def audio_to_text(audio_path, model_path):
    """Convert speech from audio file to text using Vosk."""
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return "Error: Model not found"

    audio_path = convert_audio(audio_path)
    if not audio_path:
        return "Error: Audio conversion failed"

    results = []
    try:
        logger.info(f"Loading audio file: {audio_path}")
        audio_binary_file = wave.open(audio_path, 'rb')

        model = vosk.Model(model_path)
        recognizer = vosk.KaldiRecognizer(model, audio_binary_file.getframerate())
        recognizer.SetWords(True)
        logger.info("Vosk model loaded successfully.")

        while True:
            data = audio_binary_file.readframes(4000)
            if not data:
                break
            logger.info(f"Processing {len(data)} bytes of audio data...")

            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                logger.info(f"Intermediate Recognition: {result}")
                results.append(json.loads(result).get("text", ""))
        
        final_results = json.loads(recognizer.FinalResult())
        logger.info(f"Final Recognition Output: {final_results}")
        results.append(final_results.get("text", ""))

        return " ".join(results)
    
    except Exception as e:
        logger.error(f"Error in speech recognition: {e}", exc_info=True)
        return "Error in speech recognition"

    finally:
        audio_binary_file.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Deleted temporary audio file: {audio_path}")




# def main():
#     audio_path = "audio.ogg"
#     VOSK_MODEL_PATH = "/home/hariii/Documents/Model/Vosk/vosk_model"

#     transcribed_text = audio_to_text(audio_path, VOSK_MODEL_PATH)
#     print("Final Transcription:", transcribed_text)


# if __name__ == "__main__":
#     main()
