import os

root = r"C:\Users\JIYA KHANNA\OneDrive\Desktop\tts_emotion\ravdess"

for emotion in os.listdir(root):
    emotion_path = os.path.join(root, emotion)

    if os.path.isdir(emotion_path):
        files = os.listdir(emotion_path)

        print(f"\n{emotion}")
        print(f"Found {len(files)} files")

        for f in files[:5]:
            print("  ", f)