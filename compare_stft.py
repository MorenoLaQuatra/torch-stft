import torch
from torch_stft import STFT
import torch.nn.functional as F
import numpy as np
import librosa

# Load the example audio from librosa
audio, sr = librosa.load(librosa.example('brahms'))
device = 'cpu'

# Define STFT parameters
filter_length = 1024
hop_length = 256
win_length = 1024
window = 'hann'

# Convert audio to PyTorch tensor
audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
print(f"Audio Tensor Shape: {audio_tensor.shape}")

# STFT using librosa
print("\n=== Computing STFT with Librosa ===")
stft_librosa = librosa.stft(audio, n_fft=filter_length, hop_length=hop_length, win_length=win_length, window=window)
print(f"Librosa STFT Shape: {stft_librosa.shape}")

# STFT using torch_stft
print("\n=== Computing STFT with torch_stft ===")
stft = STFT(
    filter_length=filter_length, 
    hop_length=hop_length, 
    win_length=win_length,
    window=window
).to(device)
magnitude, phase = stft.transform(audio_tensor)
print(f"Torch STFT Magnitude Shape: {magnitude.shape}, Phase Shape: {phase.shape}")

# Compare STFT Spectrograms between librosa and torch_stft
magnitude_librosa = np.abs(stft_librosa)
magnitude_torch_stft = magnitude.cpu().data.numpy()[0]
print(f"\n=== Comparing STFT Spectrograms (Librosa vs torch_stft) ===")
print(f"Magnitude Shape (librosa): {magnitude_librosa.shape}, Magnitude Shape (torch_stft): {magnitude_torch_stft.shape}")

# Pad or trim magnitude spectrograms to match dimensions in case of minor differences
if magnitude_librosa.shape[1] > magnitude_torch_stft.shape[1]:
    magnitude_librosa = magnitude_librosa[:, :magnitude_torch_stft.shape[1]]
elif magnitude_torch_stft.shape[1] > magnitude_librosa.shape[1]:
    magnitude_torch_stft = magnitude_torch_stft[:, :magnitude_librosa.shape[1]]

# Calculate MSE between spectrograms
mse_spectrogram_torch_stft = np.mean((magnitude_librosa - magnitude_torch_stft) ** 2)
print(f"MSE for STFT spectrogram magnitude comparison (Librosa vs torch_stft): {mse_spectrogram_torch_stft:.4e}")
print(f"Are spectrograms approximately equal? {np.allclose(magnitude_librosa, magnitude_torch_stft, atol=1e-5)}")

# STFT using torch's built-in function
print("\n=== Computing STFT with Torch Built-in ===")
stft_torch = torch.stft(
    audio_tensor.squeeze(0),
    n_fft=filter_length,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length).to(device),
    return_complex=True
)
print(f"Torch Built-in STFT Shape: {stft_torch.shape}")

# Compare STFT Spectrograms between librosa and torch's built-in STFT
magnitude_torch_builtin = torch.abs(stft_torch).cpu().data.numpy()
print(f"\n=== Comparing STFT Spectrograms (Librosa vs Torch Built-in) ===")
print(f"Magnitude Shape (librosa): {magnitude_librosa.shape}, Magnitude Shape (torch built-in): {magnitude_torch_builtin.shape}")

# Pad or trim magnitude spectrograms to match dimensions in case of minor differences
if magnitude_librosa.shape[1] > magnitude_torch_builtin.shape[1]:
    magnitude_librosa = magnitude_librosa[:, :magnitude_torch_builtin.shape[1]]
elif magnitude_torch_builtin.shape[1] > magnitude_librosa.shape[1]:
    magnitude_torch_builtin = magnitude_torch_builtin[:, :magnitude_librosa.shape[1]]

# Calculate MSE between spectrograms
mse_spectrogram_torch_builtin = np.mean((magnitude_librosa - magnitude_torch_builtin) ** 2)
print(f"MSE for STFT spectrogram magnitude comparison (Librosa vs Torch Built-in): {mse_spectrogram_torch_builtin:.4e}")
print(f"Are spectrograms approximately equal? {np.allclose(magnitude_librosa, magnitude_torch_builtin, atol=1e-5)}")

# ISTFT and reconstruction comparison
print("\n=== Reconstruction and Comparison ===")

# Inverse STFT using torch_stft
output_tensor = stft.inverse(magnitude, phase)
output_torch_stft = output_tensor.cpu().data.numpy()[0]
print(f"Reconstructed Audio Shape (torch_stft): {output_torch_stft.shape}")

# Inverse STFT using librosa
audio_reconstructed_librosa = librosa.istft(stft_librosa, hop_length=hop_length, win_length=win_length, window=window)
print(f"Reconstructed Audio Shape (librosa): {audio_reconstructed_librosa.shape}")

# Inverse STFT using torch's built-in ISTFT
audio_reconstructed_torch = torch.istft(
    stft_torch,
    n_fft=filter_length,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length).to(device)
)
output_torch = audio_reconstructed_torch.cpu().data.numpy()
print(f"Reconstructed Audio Shape (torch built-in): {output_torch.shape}")

# Pad or trim reconstructed audios to match original length in case of minor differences
if len(audio_reconstructed_librosa) > len(audio):
    audio_reconstructed_librosa = audio_reconstructed_librosa[:len(audio)]
else:
    audio = audio[:len(audio_reconstructed_librosa)]

if len(output_torch_stft) > len(audio):
    output_torch_stft = output_torch_stft[:len(audio)]
else:
    audio = audio[:len(output_torch_stft)]

if len(output_torch) > len(audio):
    output_torch = output_torch[:len(audio)]
else:
    audio = audio[:len(output_torch)]

# Calculate MSE between original audio and reconstructed audio
mse_torch_stft = np.mean((output_torch_stft - audio) ** 2)
print(f"MSE for torch_stft reconstruction: {mse_torch_stft:.4e}")

mse_librosa = np.mean((audio_reconstructed_librosa - audio) ** 2)
print(f"MSE for librosa reconstruction: {mse_librosa:.4e}")

mse_torch_builtin = np.mean((output_torch - audio) ** 2)
print(f"MSE for torch's built-in STFT and ISTFT reconstruction: {mse_torch_builtin:.4e}")

# Compare reconstructions
print(f"\n=== Comparing Reconstructions ===")
print(f"Difference in MSE between torch_stft and librosa: {abs(mse_torch_stft - mse_librosa):.4e}")
print(f"Difference in MSE between torch_stft and torch's built-in STFT: {abs(mse_torch_stft - mse_torch_builtin):.4e}")
print(f"Difference in MSE between librosa and torch's built-in STFT: {abs(mse_librosa - mse_torch_builtin):.4e}")
