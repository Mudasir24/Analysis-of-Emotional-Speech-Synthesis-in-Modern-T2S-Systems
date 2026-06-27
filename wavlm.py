import torch
import torchaudio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

model = WavLMModel.from_pretrained("microsoft/wavlm-base")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base"
)

model.eval()


def extract_embedding(audio_arr, sr):

    waveform = torch.tensor(
        audio_arr,
        dtype=torch.float32
    ).unsqueeze(0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=16000
        )
        waveform = resampler(waveform)

    waveform_1d = waveform.squeeze(0).numpy()

    inputs = feature_extractor(
        waveform_1d,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return embedding