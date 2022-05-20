#!/usr/bin/python3
# -*- coding: utf-8 -*-
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#       OS : GNU/Linux Ubuntu 16.04 or later
# LANGUAGE : Python 3.5.2 or later
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
'''
Предназначен для подавления шума в wav аудиозаписи с помощью библиотеки RNNoise (https://github.com/xiph/rnnoise).

Содержит класс RNNoise. Подробнее в https://github.com/Desklop/RNNoise_Wrapper.

Зависимости: pydub, numpy.
'''
import sys
import os
import platform
import subprocess
import time
import ctypes
import pkg_resources
import numpy as np
from pydub import AudioSegment

__version__ = 1.1


class RNNoise(object):
    sample_width = 2
    channels = 1
    sample_rate = 48000
    frame_duration_ms = 10

    def __init__(self, f_name_lib=None):

        f_name_lib = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  'libs', 'librnnoise_default.so.0.4.1')
        if not os.path.exists(f_name_lib):
            print('You must first compile RNNoise library.')
            sys.exit(1)
        #f_name_lib = self.__get_f_name_lib(f_name_lib)
        self.rnnoise_lib = ctypes.cdll.LoadLibrary(f_name_lib)

        self.rnnoise_lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.rnnoise_lib.rnnoise_process_frame.restype = ctypes.c_float
        self.rnnoise_lib.rnnoise_create.restype = ctypes.c_void_p
        self.rnnoise_lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

        self.rnnoise_obj = self.rnnoise_lib.rnnoise_create(None)

    def __get_f_name_lib(self, f_name_lib=None):
        package_name = __file__
        package_name = package_name[package_name.rfind('/') +
                                    1:package_name.rfind('.py')]

        if not f_name_lib:
            subname = 'librnnoise'
            system = platform.system()
            if system == 'Linux' or system == 'Darwin':
                found_f_name_lib = pkg_resources.resource_filename(
                    package_name, 'libs/{}_5h_b_500k.so.0.4.1'.format(subname))
                if not os.path.exists(found_f_name_lib):
                    found_f_name_lib = self.__find_lib(subname)
            else:
                found_f_name_lib = self.__find_lib(subname)

            if not found_f_name_lib:
                raise NameError(
                    "could not find RNNoise library with subname '{}'".format(
                        subname))

        else:
            f_names_available_libs = pkg_resources.resource_listdir(
                package_name, 'libs/')
            for available_lib in f_names_available_libs:
                if available_lib.find(f_name_lib) != -1:
                    f_name_lib = pkg_resources.resource_filename(
                        package_name, 'libs/{}'.format(available_lib))

            found_f_name_lib = self.__find_lib(f_name_lib)
            if not found_f_name_lib:
                raise NameError(
                    "could not find RNNoise library with name/subname '{}'".
                    format(f_name_lib))

        return found_f_name_lib

    def __find_lib(self, f_name_lib, root_folder='.'):
        f_name_lib_full = os.path.abspath(f_name_lib)
        if os.path.isfile(f_name_lib_full) and os.path.exists(f_name_lib_full):
            return f_name_lib_full

        for path, folder_names, f_names in os.walk(root_folder):
            for f_name in f_names:
                if f_name.rfind(f_name_lib) != -1:
                    return os.path.join(path, f_name)

    def reset(self):
        self.rnnoise_lib.rnnoise_destroy(self.rnnoise_obj)
        self.rnnoise_obj = self.rnnoise_lib.rnnoise_create(None)

    def filter_frame(self, frame):
        # 480 = len(frame)/2, len(frame) всегда должна быть 960 значений (т.к. ширина фрейма 2 байта (16 бит))
        # (т.е. длина фрейма 10 мс (0.01 сек) при частоте дискретизации 48000 Гц, 48000*0.01*2=960).
        # Если len(frame) != 960, будет ошибка сегментирования либо сильные искажения на итоговой аудиозаписи.

        # Если вынести np.ndarray((480,), 'h', frame).astype(ctypes.c_float) в __get_frames(), то прирост в скорости работы составит
        # не более 5-7% на аудиозаписях, длиной от 60 секунд. На более коротких аудиозаписях прирост скорости менее заметен и несущественен.

        frame_buf = np.ndarray((480, ), 'h', frame).astype(ctypes.c_float)
        frame_buf_ptr = frame_buf.ctypes.data_as(ctypes.POINTER(
            ctypes.c_float))

        vad_probability = self.rnnoise_lib.rnnoise_process_frame(
            self.rnnoise_obj, frame_buf_ptr, frame_buf_ptr)
        return vad_probability, frame_buf.astype(ctypes.c_short).tobytes()

    def filter(self,
               audio,
               sample_rate=None,
               voice_prob_threshold=0.0,
               save_source_sample_rate=True):
        frames, source_sample_rate = self.__get_frames(audio, sample_rate)
        if not save_source_sample_rate:
            source_sample_rate = None

        denoised_audio = self.__filter_frames(frames, voice_prob_threshold,
                                              source_sample_rate)

        if isinstance(audio, AudioSegment):
            return denoised_audio
        else:
            return denoised_audio.raw_data

    def __filter_frames(self,
                        frames,
                        voice_prob_threshold=0.0,
                        sample_rate=None):

        denoised_frames_with_probability = [
            self.filter_frame(frame) for frame in frames
        ]
        denoised_frames = [
            frame_with_prob[1]
            for frame_with_prob in denoised_frames_with_probability
            if frame_with_prob[0] >= voice_prob_threshold
        ]
        denoised_audio_bytes = b''.join(denoised_frames)

        denoised_audio = AudioSegment(data=denoised_audio_bytes,
                                      sample_width=self.sample_width,
                                      frame_rate=self.sample_rate,
                                      channels=self.channels)

        if sample_rate:
            denoised_audio = denoised_audio.set_frame_rate(sample_rate)
        return denoised_audio

    def __get_frames(self, audio, sample_rate=None):
        if isinstance(audio, AudioSegment):
            sample_rate = source_sample_rate = audio.frame_rate
            if sample_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            audio_bytes = audio.raw_data
        elif isinstance(audio, bytes):
            if not sample_rate:
                raise ValueError(
                    "when type(audio) = 'bytes', 'sample_rate' can not be None"
                )
            audio_bytes = audio
            source_sample_rate = sample_rate
            if sample_rate != self.sample_rate:
                audio = AudioSegment(data=audio_bytes,
                                     sample_width=self.sample_width,
                                     frame_rate=sample_rate,
                                     channels=self.channels)
                audio = audio.set_frame_rate(self.sample_rate)
                audio_bytes = audio.raw_data
        else:
            raise TypeError("'audio' can only be AudioSegment or bytes")

        frame_width = int(self.sample_rate *
                          (self.frame_duration_ms / 1000.0) * 2)
        if len(audio_bytes) % frame_width != 0:
            silence_duration = frame_width - len(audio_bytes) % frame_width
            audio_bytes += b'\x00' * silence_duration

        offset = 0
        frames = []
        while offset + frame_width <= len(audio_bytes):
            frames.append(audio_bytes[offset:offset + frame_width])
            offset += frame_width
        return frames, source_sample_rate

    def read_wav(self, f_name_wav, sample_rate=None):
        if isinstance(f_name_wav, str) and f_name_wav.rfind('.wav') == -1:
            raise ValueError(
                "'f_name_wav' must contain the name .wav audio recording")

        audio = AudioSegment.from_wav(f_name_wav)

        if sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        if audio.sample_width != self.sample_width:
            audio = audio.set_sample_width(self.sample_width)
        if audio.channels != self.channels:
            audio = audio.set_channels(self.channels)
        return audio

    def write_wav(self, f_name_wav, audio_data, sample_rate=None):
        if isinstance(audio_data, AudioSegment):
            self.write_wav_from_audiosegment(f_name_wav, audio_data,
                                             sample_rate)
        elif isinstance(audio_data, bytes):
            if not sample_rate:
                raise ValueError(
                    "when type(audio_data) = 'bytes', 'sample_rate' can not be None"
                )
            self.write_wav_from_bytes(f_name_wav, audio_data, sample_rate)
        else:
            raise TypeError("'audio_data' is of an unsupported type. Supported:\n" + \
                            "\t- pydub.AudioSegment with audio\n" + \
                            "\t- byte string with audio data (without wav header)")

    def write_wav_from_audiosegment(self,
                                    f_name_wav,
                                    audio,
                                    desired_sample_rate=None):
        if desired_sample_rate:
            audio = audio.set_frame_rate(desired_sample_rate)
        audio.export(f_name_wav, format='wav')

    def write_wav_from_bytes(self,
                             f_name_wav,
                             audio_bytes,
                             sample_rate,
                             desired_sample_rate=None):
        audio = AudioSegment(data=audio_bytes,
                             sample_width=self.sample_width,
                             frame_rate=sample_rate,
                             channels=self.channels)
        if desired_sample_rate and desired_sample_rate != sample_rate:
            audio = audio.set_frame_rate(desired_sample_rate)

        audio.export(f_name_wav, format='wav')
