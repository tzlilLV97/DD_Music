from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
from Decoder import our_decoder
from Encoder import our_encoder

# Encode audio file to tokens and Decode it back

audio_path = r"C:\Projects\WavTokenizer\TrainTokens\AudioData\1073150.low.mp3"
save_path = r"C:\Projects\WavTokenizer\TrainTokens\tokenTest"

our_encoder(audio_path, save_path)

token_path = save_path
audio_outpath = "./after_tokenizing"

our_decoder(token_path, audio_outpath)




