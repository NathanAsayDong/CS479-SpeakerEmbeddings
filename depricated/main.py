import os
import sounddevice as sd
import soundfile as sf
from asr_service import ASRService
from translation_service import TranslationService
from tts_service import TTSService
from enums import Language

def play_audio(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, fs)
    sd.wait()

def main():
    print("Initializing services...")
    
    # Initialize services
    try:
        asr = ASRService()
        print("ASR Service ready.")
        
        translator = TranslationService()
        print("Translation Service ready.")
        
        tts = TTSService()
        print("TTS Service ready.")
        
    except Exception as e:
        print(f"Error initializing services: {e}")
        return

    print("\n--- Speech-to-Speech Translation Pipeline ---")
    print("This pipeline will:")
    print("1. Listen to your microphone")
    print("2. Transcribe the English speech")
    print("3. Translate to Spanish")
    print("4. Synthesize Spanish speech in your voice style")
    
    # Enrollment Phase
    speaker_audio_path = None
    if input("\nDo you want to enroll your voice for better cloning? (y/n): ").lower().strip() == 'y':
        input("Press Enter to start recording your enrollment (speak clearly for 10s)...")
        speaker_audio_path = asr.record_audio(duration=10, file_path="my_voice_enrollment.wav")
        print("Enrollment complete! This audio will be used for cloning.")

    while True:
        input("\nPress Enter to start recording (or Ctrl+C to exit)...")
        
        try:
            # Step 1: Listen and Transcribe
            print("\nRecording... (speak now)")
            english_text, current_audio_path = asr.listen_transcribe(duration=5)
            print(f"Transcribed: {english_text}")
            
            if not english_text:
                print("No speech detected. Try again.")
                continue

            # Step 2: Translate
            print("Translating...")
            spanish_text = translator.translate(english_text, target_language=Language.PORTUGUESE)
            print(f"Translated: {spanish_text}")
            
            # Step 3: Synthesize
            print("Synthesizing speech...")
            output_path = "output_spanish.wav"
            
            # Use enrollment audio if available, otherwise use the current clip
            reference_audio = speaker_audio_path if speaker_audio_path else current_audio_path
            
            tts.synthesize(spanish_text, output_path, reference_audio)
            print(f"Output saved to {output_path}")
            
            # Step 4: Playback
            print("Playing output...")
            play_audio(output_path)
            
            # Cleanup source audio if desired
            # os.remove(source_audio_path)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

