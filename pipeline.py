
import os
import csv
import soundfile as sf
 
from app import generate_audio
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
 
ESD_ROOT = r"C:\Users\JIYA KHANNA\Downloads\archive (3)\Emotion Speech Dataset"
OUTPUT_ROOT = r"C:\Users\JIYA KHANNA\OneDrive\Desktop\tts_emotion\synthetic_audio"
 
SPEAKER_IDS = ["0011"]
NUM_TEXTS = 350  # generates all 350, skips existing 200 automatically
 
# ─────────────────────────────────────────────
# EMOTION DESCRIPTIONS
# ─────────────────────────────────────────────
 
EMOTION_DESCRIPTIONS = {
    "Neutral":  "The speaker speaks in a neutral tone. The recording is high quality.",
    "Angry":    "The speaker speaks in an angry tone. The recording is high quality.",
    "Happy":    "The speaker speaks in a happy tone. The recording is high quality.",
    "Sad":      "The speaker speaks in a sad tone. The recording is high quality.",
    "Surprise": "The speaker speaks in a shocked tone. The recording is high quality.",
}
 
# ─────────────────────────────────────────────
# TRANSCRIPT PARSER
# ─────────────────────────────────────────────
 
def parse_transcript(txt_path, num_texts=350):
    emotion_texts = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            file_id, text, emotion = parts
            if emotion not in emotion_texts:
                emotion_texts[emotion] = []
            if len(emotion_texts[emotion]) < num_texts:
                emotion_texts[emotion].append((file_id, text))
    return emotion_texts
 
# ─────────────────────────────────────────────
# CSV SETUP
# ─────────────────────────────────────────────
 
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_path = os.path.join(OUTPUT_ROOT, "generated_audio.csv")
csv_file = open(csv_path, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
 
if os.path.getsize(csv_path) == 0:
    csv_writer.writerow(["speaker", "emotion", "file_id", "text", "wav_path"])
 
# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
 
try:
    for speaker_id in SPEAKER_IDS:
        txt_path = os.path.join(ESD_ROOT, speaker_id, f"{speaker_id}.txt")
 
        if not os.path.exists(txt_path):
            print(f"[SKIP] Transcript not found: {txt_path}")
            continue
 
        print("\n" + "=" * 60)
        print(f"Processing Speaker: {speaker_id}")
        print("=" * 60)
 
        emotion_texts = parse_transcript(txt_path, NUM_TEXTS)
 
        for emotion, entries in emotion_texts.items():
            description = EMOTION_DESCRIPTIONS.get(emotion)
            if description is None:
                continue
 
            out_dir = os.path.join(OUTPUT_ROOT, speaker_id, emotion)
            os.makedirs(out_dir, exist_ok=True)
 
            print(f"\n[{emotion}] -> {len(entries)} files")
 
            for idx, (file_id, text) in enumerate(entries):
                wav_path = os.path.join(out_dir, f"{file_id}.wav")
 
                # Skip if already generated
                if os.path.exists(wav_path):
                    print(f"  [{idx+1}/{len(entries)}] SKIP: {file_id}")
                    continue
 
                try:
                    print(f"  [{idx+1}/{len(entries)}] {file_id} | {text[:50]}")
                    audio_arr, sr = generate_audio(text, description)
                    sf.write(wav_path, audio_arr, sr)
                    csv_writer.writerow([speaker_id, emotion, file_id, text, wav_path])
                    csv_file.flush()
 
                except Exception as e:
                    print(f"  [ERROR] {file_id}: {e}")
                    continue
 
finally:
    csv_file.close()
 
print("\nDone! Audio saved to:", OUTPUT_ROOT)
print("CSV saved to:", csv_path)