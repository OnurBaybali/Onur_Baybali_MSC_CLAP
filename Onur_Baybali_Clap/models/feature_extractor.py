#!/usr/bin/env python3
# coding: utf-8
# below codes are based and adapted from https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/models/feature_extractor.py


import torch
import torch.nn as nn
from torchlibrosa import LogmelFilterBank, Spectrogram
import dac
from pathlib import Path


class AudioFeature(nn.Module):

    def __init__(self, audio_config):
        super().__init__()
        self.mel_trans = Spectrogram(n_fft=audio_config["n_fft"],
                                     hop_length=audio_config["hop_length"],
                                     win_length=audio_config["n_fft"],
                                     window='hann',
                                     center=True,
                                     pad_mode='reflect',
                                     freeze_parameters=True)

        self.log_trans = LogmelFilterBank(sr=audio_config["sr"],
                                          n_fft=audio_config["n_fft"],
                                          n_mels=audio_config["n_mels"],
                                          fmin=audio_config["f_min"],
                                          fmax=audio_config["f_max"],
                                          ref=1.0,
                                          amin=1e-10,
                                          top_db=None,
                                          freeze_parameters=True)

    def forward(self, input):
        # input: waveform [bs, wav_length]
        # print('input.shape: ', input.shape)
        # print(input)
        mel_feats = self.mel_trans(input)
        # print('mel_feats.shape: ', mel_feats.shape)
        # print(mel_feats)
        log_mel = self.log_trans(mel_feats)
        # print('log_mel.shape: ', log_mel.shape)
        # print(log_mel)
        return log_mel
    
##################################################################
# IMPLEMENTATION OF DAC CODES FEATURES
##################################################################

class DACodes(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        # self.config = config   
        self.sr = config["audio_args"]["sr"]
        self.NumCodes = config["audio_encoder_args"]["N_CODEBOOKS"]
        ### Add the implementation of DAC
        model_path = Path(config["audio_encoder_args"]["dac_path"])
        # model_path = Path('/home/dhuang/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth')
        self.audio_enc = dac.DAC.load(model_path)
        # self.audio_enc = self.audio_enc.to('cuda')
        # self.audio_enc.eval()
        if "audio_width" in config["audio_encoder_args"]:
            self.audio_width = config["audio_encoder_args"]["audio_width"]              
            
        if config["audio_encoder_args"]["freeze"]:
            for name, param in self.audio_enc.named_parameters():
                param.requires_grad = False

    def forward(self, inputs):
        """
        :param inputs: audio features
        :return: encoded audio embeddings
        """
        # self.audio_enc.eval()
        # signals = AudioSignal(inputs)
        x = self.audio_enc.preprocess(inputs, self.sr)
        # x = torch.tensor(x[:,None,:])
        x = x[:,None,:].detach().clone()
        # print("inputs wav hz is: ", len(inputs[0]) / self.config["audio_args"]["max_length"])
        # print("x shape is: ", x.shape)
        # time.sleep(10)
        # print(f"self.training in eval mode: {not self.training}")
        # print(f"self.training.audio_enc in eval mode: {not self.audio_enc.training}")
        with torch.no_grad():
            # print("N_CODEBOOKS: ", self.config["audio_encoder_args"]["N_CODEBOOKS"])
            z, codes, latents, _, _ = self.audio_enc.encode(x, self.NumCodes)
            # print("codes shape is: ", codes.shape)
            # print("torch.flatten(codes, start_dim=1): ", torch.flatten(codes, start_dim=1).shape)
        return torch.flatten(codes, start_dim=1)#.to(torch.float)
    
##################################################################
# IMPLEMENTATION OF DAC LATENTS FEATURES
##################################################################

class DACLatents(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        # self.config = config   
        self.sr = config["audio_args"]["sr"]
        self.NumCodes = config["audio_encoder_args"]["N_CODEBOOKS"]
        ### Add the implementation of DAC
        # device = torch.device('cuda')            
        model_path = Path(config["audio_encoder_args"]["dac_path"])
        # model_path = Path('/home/dhuang/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth')
        self.audio_enc = dac.DAC.load(model_path)
        # self.audio_enc = self.audio_enc.to(device)
        # self.audio_enc.eval()
        if "audio_width" in config["audio_encoder_args"]:
            self.audio_width = config["audio_encoder_args"]["audio_width"]            
            
        if config["audio_encoder_args"]["freeze"]:
            for name, param in self.audio_enc.named_parameters():
                param.requires_grad = False

    def forward(self, inputs):
        """
        :param inputs: audio features
        :return: encoded audio embeddings
        """
        # self.audio_enc.eval()
        # signals = AudioSignal(inputs)
        x = self.audio_enc.preprocess(inputs, self.sr)
        # x = torch.tensor(x[:,None,:])
        x = x[:,None,:].detach().clone()
        # print("inputs wav hz is: ", len(inputs[0]) / self.config["audio_args"]["max_length"])
        # print("x shape is: ", x.shape)
        # time.sleep(10)
        # print(f"self.training in eval mode: {not self.training}")
        # print(f"self.training.audio_enc in eval mode: {not self.audio_enc.training}")
        with torch.no_grad():
            # print("N_CODEBOOKS: ", self.config["audio_encoder_args"]["N_CODEBOOKS"])
            z, codes, latents, _, _ = self.audio_enc.encode(x, self.NumCodes)
            # print("codes shape is: ", codes.shape)
            # print("torch.flatten(codes, start_dim=1): ", torch.flatten(codes, start_dim=1).shape)
        # return torch.flatten(latents, start_dim=1)#.to(torch.float)
        return latents

##################################################################
# IMPLEMENTATION chunk func for VAMP
##################################################################
import math
def chunk(z: torch.Tensor,
          mask: torch.Tensor,
          n_codebooks: int,
          max_seq_len: int,
):
    # coarse z
    cz = z[:, : n_codebooks, :].clone()
    mask = mask[:, : n_codebooks, :]

    seq_len = cz.shape[-1]

    # we need to split the sequence into chunks by max seq length
    # we need to split so that the sequence length is less than the max_seq_len
    n_chunks = math.ceil(seq_len / max_seq_len)
    chunk_len = math.ceil(seq_len / n_chunks)
    # print(f"will process {n_chunks} chunks of length {chunk_len}")

    z_chunks = torch.split(cz, chunk_len, dim=-1)
    mask_chunks = torch.split(mask, chunk_len, dim=-1)
 
    return zip(z_chunks, mask_chunks)