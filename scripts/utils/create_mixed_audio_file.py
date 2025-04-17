# -*- coding: utf-8 -*-
# script taken from https://github.com/Sato-Kunihiko/audio-SNR.git
import argparse
import array
import math
import numpy as np
import random
import wave
import ipdb
import sys
import os
import soundfile as sf


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_file', type=str, default="/data/users/achingacham/Corpora/quora-question-pairs/qqp_audio_test/clean_audio/utt_6997.wav.wav")
    parser.add_argument('--noise_file', type=str, default="/projects/SFB_A4/AudioRepo/noises_2/babble")
    parser.add_argument('--output_mixed_file', type=str, default="/tmp/sample_noisy_speech.wav")
    parser.add_argument('--output_clean_file', type=str, default='/tmp/sample_clean.wav')
    parser.add_argument('--snr', type=float, default="-5")
    parser.add_argument('-sT', '--startTime', type=float, default=0)
    parser.add_argument('-eT', '--endTime', type=float, default=100)
    parser.add_argument('-mmL', '--max_mix_length', help="when the noise snippet needs to same across multiple mixing instances. eg: paraphrase pairs find the maximum length",
                        type=int, default=None)

    #boolean variables as args: mention only required:
    #eg: python3 create_mixed_audio_file.py --clean_file --noise_file -sO -mix [noise needs to mixed and output to console]
    parser.add_argument('-sO', '--stdOut', default=False, action="store_true", help="choose either std output (sox) or save file")
    parser.add_argument('-mix', '--mixNoise', default=False, action="store_true", help="choose to mix noise or just pick a snippet of audio")

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    # The dtype depends on the value of pulse-code modulation. The int16 is set for 16-bit PCM.
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    amp = np.trim_zeros(amp)  #remove leading and trailing zeros (representing silence)
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def save_waveform(output_path, params, amp):
    #pdb.set_trace()
    output_file = wave.Wave_write(output_path)
    output_file.setparams(params) #nchannels, sampwidth, framerate, nframes, comptype, compname
    output_file.writeframes(array.array('h', amp.astype(np.int16)).tobytes() )
    output_file.close()

def time_to_frame(waveObj, time_ms):
    #input is wavFile and the time in milliseconds
    #ipdb.set_trace()
    total_s = waveObj.getnframes()/waveObj.getframerate() #total time in seconds
    n_frames_ms = waveObj.getframerate()/1000 #number of frames in milli second

    if time_ms:

        starting_frame_no_at_time_ms = (time_ms - 1) * n_frames_ms
        ending_frame_no_at_time_ms = time_ms * n_frames_ms - 1  # ends just before the next milli second.

    else:
        starting_frame_no_at_time_ms = 0  # starts with index 0
        ending_frame_no_at_time_ms = n_frames_ms - 1  # ends just before the next milli second.

    return int(starting_frame_no_at_time_ms), int(ending_frame_no_at_time_ms)

def get_audio_length(wav_file_path):

    wav_data = wave.open(wav_file_path, "r")
    wav_amp = cal_amp(wav_data)

    return len(wav_amp)

def perform_additive_noise_mixing(**kwargs):

    # ipdb.set_trace()
    # print(kwargs)

    clean_file = kwargs['clean_file']

    # NO noise mixing if there exists NO clean file
    if not os.path.exists(clean_file):
        print(clean_file, " does not exist!")
        return 1

    clean_wav = wave.open(clean_file, "r")

    startFrame_nos = time_to_frame(clean_wav, kwargs['startTime'] * 1000)
    endFrame_nos = time_to_frame(clean_wav, kwargs['endTime'] * 1000)

    clean_amp_org = cal_amp(clean_wav)

    if len(clean_amp_org) > endFrame_nos[1]:
        clean_amp = clean_amp_org[startFrame_nos[0]:endFrame_nos[1]]
    else:
        clean_amp = clean_amp_org[startFrame_nos[0]:]

    if not kwargs['mixNoise']:
        # just pick the required portion of audio
        save_waveform(kwargs['output_clean_file'], clean_wav.getparams(), clean_amp)
        os.system(
            "/data/users/achingacham/anaconda3/envs/pysari/bin/sox -t wav " + kwargs['output_clean_file'] + " -t wav -")
        os.system("rm " + kwargs['output_clean_file'])

    else:
        # perform noise mixing
        noise_file = kwargs['noise_file']
        noise_wav = wave.open(noise_file, "r")
        noise_amp_org = cal_amp(noise_wav)

        mix_length = len(clean_amp)

        # to make the noise as random as possible; but also perform synchronized noise mixing for paraphrases.
        if kwargs['max_mix_length'] is None:
            random_start = np.random.randint(0, len(noise_amp_org) - mix_length)
        else:
            max_mix_length = kwargs['max_mix_length']
            np.random.seed(int(max_mix_length))  # to make sure the given mix-length is used to pick the same noise snippet.
            #random_start = np.random.randint(0, len(noise_amp_org) - max_mix_length)
            random_start = np.random.randint(0, int(len(noise_amp_org)/2))

        #print("Random start:", random_start, "mix-length:", mix_length)
        noise_amp = noise_amp_org[random_start: random_start + mix_length]
        clean_rms = cal_rms(clean_amp)
        noise_rms = cal_rms(noise_amp)

        snr = kwargs['snr']
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = (clean_amp + adjusted_noise_amp)

        # prepend few second of noise (for accoustom, for listening exp)
        # print("Before mixing : ", len(mixed_amp))
        # temp_1 = divided_noise_amp[:5000]
        # temp_2 = np.append(temp_1, divided_noise_amp)
        # temp_noise = np.append(temp_2, temp_1)
        # temp_clean = np.append(np.zeros(5000), clean_amp)
        # temp_clean = np.append(temp_clean, np.zeros(5000))
        # mixed_amp = temp_clean + ( temp_noise * (adjusted_noise_rms / noise_rms)  )
        # print("After mixing:", len(mixed_amp))

        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
            if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
                reduction_rate = max_int16 / mixed_amp.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed_amp.min(axis=0)

            mixed_amp = mixed_amp * (reduction_rate)
            clean_amp = clean_amp * (reduction_rate)
            # temp_clean = temp_clean * (reduction_rate)

        if kwargs['stdOut']:

            save_waveform(kwargs['output_mixed_file'], clean_wav.getparams(), mixed_amp)
            os.system(
                "/data/users/achingacham/anaconda3/envs/pysari/bin/sox -t wav " + kwargs['output_mixed_file'] + " -t wav -")
            os.system("rm " + kwargs['output_mixed_file'])

        else:

            save_waveform(kwargs['output_mixed_file'], clean_wav.getparams(), mixed_amp)
            save_waveform(kwargs['output_clean_file'], clean_wav.getparams(), clean_amp)  # for pySTOI


if __name__ == '__main__':

    dict_args = get_args() # either std input arguments
    # args = Args() # or using class parameters
    perform_additive_noise_mixing(**dict_args)