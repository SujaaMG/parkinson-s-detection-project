import librosa
import numpy as np
import pandas as pd
import os

def extract_features_from_audio(file_path):
    """Extract MFCCs, jitter, shimmer, spectral, and energy-based features as numeric floats."""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y, _ = librosa.effects.trim(y)
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return None

    features = {}

    # Core spectral features (convert all to float)
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features["centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features["rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features["chroma_stft"] = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    features["rms"] = float(np.mean(librosa.feature.rms(y=y)))

    # MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, coeff in enumerate(mfcc):
        features[f"mfcc_{i+1}"] = float(np.mean(coeff))

    # Pitch-based jitter/shimmer proxies
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[pitches > 0]
    if len(pitch) > 1:
        diff = np.diff(pitch)
        features["jitter"] = float(np.mean(np.abs(diff / pitch[:-1])))
        features["shimmer"] = float(np.std(pitch) / np.mean(pitch))
    else:
        features["jitter"] = 0.0
        features["shimmer"] = 0.0

    # Optional placeholders to match model input dimensions (if needed)
    features["dummy_1"] = 0.0
    features["dummy_2"] = 0.0

    # Return as a single-row DataFrame
    return pd.DataFrame([features])


def process_dataset(audio_folder="data/raw_audio/"):
    """Iterate through both HC and PD folders and extract features."""
    all_data = []
    for label_dir, label in [("HC", 0), ("PD", 1)]:
        folder = os.path.join(audio_folder, label_dir)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)
                feats_df = extract_features_from_audio(file_path)
                if feats_df is not None and not feats_df.empty:
                    feats_df["label"] = label
                    feats_df["file_name"] = file
                    all_data.append(feats_df)
                    print(f"✅ Processed {file}")

    if not all_data:
        print("⚠️ No audio files processed. Please check your folder paths.")
        return None

    df = pd.concat(all_data, ignore_index=True)
    os.makedirs("data/features", exist_ok=True)
    df.to_csv("data/features/features.csv", index=False)
    print(f"🎯 Features saved to data/features/features.csv with {df.shape[0]} samples.")
    return df


if __name__ == "__main__":
    process_dataset()
