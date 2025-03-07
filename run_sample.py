import torch
import argparse
import sys
import os
sys.path.append('./OurWavTokenizer')
from load_model import load_model, load_model_local
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
from tqdm import tqdm

def gen_sample(audio_path, steps, model_path, batch_size=1, start_save=0):
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model_local(
        model_path, device)
    #model, graph, noise = load_model(args.model_path, device)
    #tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    wavtokenizer = WavTokenizer.from_pretrained0802(
        "./OurWavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "./OurWavTokenizer/models/wavtokenizer_medium_music_audio_320_24k.ckpt")
    wavtokenizer = wavtokenizer.to('cpu')
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (batch_size, 300), 'analytic', steps, device=device
    )

    token = sampling_fn(model)[None]

    features = wavtokenizer.codes_to_features(token.cpu())
    bandwidth_id = torch.tensor([0])
    audio_out = wavtokenizer.decode(features.cpu(), bandwidth_id=bandwidth_id)
    for i in tqdm(range(batch_size)):
        audio_path_file = os.path.join(audio_path,f'{int(start_save+i)}.wav')
        torchaudio.save(audio_path_file, audio_out[i][None], sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

if __name__=="__main__":
    # gen_sample(f"./gen_samples/", 1024,
    #                '/gpfs0/bgu-haimp/users/romhi/PycharmProjects/SEDD/exp_local/openwebtext/2025.01.13/122837', 10, start_save=500)
    # gen_sample(f"./gen_samples/", 1024,
    #                '/gpfs0/bgu-haimp/users/romhi/PycharmProjects/SEDD/exp_local/openwebtext/2025.01.13/122837', 100, start_save=100)
    # gen_sample(f"./gen1/", 1024,
    #                '/gpfs0/bgu-haimp/users/romhi/PycharmProjects/SEDD/exp_local/openwebtext/2025.01.15/222301', 100, start_save=0)
    # gen_sample(f"./gen1/", 1024,
    #                '/gpfs0/bgu-haimp/users/romhi/PycharmProjects/SEDD/exp_local/openwebtext/2025.01.15/222301', 100, start_save=100)
    # gen_sample(f"./gen1/", 1024,
    #                '/gpfs0/bgu-haimp/users/romhi/PycharmProjects/SEDD/exp_local/openwebtext/2025.01.15/222301', 100, start_save=200)
    gen_sample(f"./new-data-gen/", 1024,
                   '/gpfs0/bgu-haimp/users/romhi/PycharmProjects/SEDD/exp_local/openwebtext/2025.01.16/144128', 100, start_save=0)