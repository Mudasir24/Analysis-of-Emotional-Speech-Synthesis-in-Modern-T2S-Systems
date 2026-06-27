import os
import torch
import scipy.io.wavfile as wavfile
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RAVDESS_ROOT = r"C:\Users\JIYA KHANNA\OneDrive\Desktop\tts_emotion\ravdess"

OUTPUT_ROOT = r"C:\Users\JIYA KHANNA\OneDrive\Desktop\tts_emotion\synthetic_audio_ravdess"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
)

# ---------------------------------------------------------------------------
# EMOTION DESCRIPTIONS
# ---------------------------------------------------------------------------

EMOTION_DESCRIPTIONS = {
    "Neutral": "A calm, neutral voice speaking in a flat, even tone with no strong emotion.",
    "Happy": "Speaker speaks in a happy, cheerful tone.",
    "Sad": "Speaker speaks in a sad tone.",
    "Angry": "Speaker speaks in an angry tone.",
    "Fearful": "Speaker speaks in a fearful anxious tone.",
    "Disgust": "Speaker speaks in a disgusted tone.",
}

# ---------------------------------------------------------------------------
# FIND FILES
# ---------------------------------------------------------------------------

def find_ravdess_files(root):
    files = []

    emotions = [
        "neutral",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust"
    ]

    for emotion in emotions:
        emotion_dir = os.path.join(root, emotion)

        if not os.path.exists(emotion_dir):
            print(f"Warning: Missing folder: {emotion_dir}")
            continue

        for filename in sorted(os.listdir(emotion_dir)):
            if filename.lower().endswith(".wav"):
                files.append(
                    (
                        os.path.join(emotion_dir, filename),
                        emotion
                    )
                )

    return files

# ---------------------------------------------------------------------------
# AUDIO GENERATION
# ---------------------------------------------------------------------------

def generate_audio(prompt, description):

    input_ids = tokenizer(
        description,
        return_tensors="pt"
    ).input_ids.to(device)

    prompt_input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )

    audio_arr = generation.cpu().numpy().squeeze()

    return audio_arr, model.config.sampling_rate

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    files = find_ravdess_files(RAVDESS_ROOT)

    print(f"Found {len(files)} RAVDESS files")

    if len(files) == 0:
        print("No WAV files found.")
        return



    log_rows = [
        "filename,output_filename,emotion,prompt,output_path"
    ]

    for i, (filepath, emotion_folder) in enumerate(files, start=1):

        
        filename = os.path.basename(filepath)

        if filename == "001.wav":
            prompt_text = "Kids are talking by the door"

        elif filename == "002.wav":
            prompt_text = "Dogs are sitting by the door"

        else:
            print(f"Unknown filename: {filename}")
            continue

        emotion = emotion_folder.capitalize()

        description = EMOTION_DESCRIPTIONS[emotion]

        

        out_dir = os.path.join(
            OUTPUT_ROOT,
            emotion_folder
        )

        os.makedirs(out_dir, exist_ok=True)

        output_filename = filename

        out_path = os.path.join(
            out_dir,
            output_filename
        )

        if os.path.exists(out_path):
            print(
                f"[{i}/{len(files)}] Already exists, skipping: {output_filename}"
            )
            continue

        print(
            f"[{i}/{len(files)}] Generating {emotion_folder}/{output_filename}"
        )

        try:

            audio_arr, sampling_rate = generate_audio(
                prompt_text,
                description
            )

            wavfile.write(
                out_path,
                sampling_rate,
                audio_arr
            )

            log_rows.append(
                f"{filename},{output_filename},{emotion},{prompt_text},{out_path}"
            )

        except Exception as e:
            print(
                f"Error generating {output_filename}: {e}"
            )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    log_path = os.path.join(
        OUTPUT_ROOT,
        "generated_audio_log.csv"
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_rows))

    print("\nDone.")
    print(f"Log saved to: {log_path}")

if __name__ == "__main__":
    main()