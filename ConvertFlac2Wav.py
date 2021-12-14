# Convert Flac files back to .wav files
import fileutils
from fileutils import audioset
import numpy
import struct
import subprocess
import librosa
from scipy.io import wavfile
from fileutils import smart_open
from scipy.io.wavfile import write
import os
import numpy as np


def writeAudioSetWAV(filename, destFolder):
    """
    Reads audio and labels from a Google Audio Set file (disguised as .flac).
    Returns two variables:
      * wav -- a 2-D numpy float32 array, where each row is a waveform
        (10 seconds @ 16 kHz, mono);
      * labels -- a 2-D numpy int32 array of zeros and ones, where each row
        indicates the sound events active in the corresponding waveform.
    """
    wav, sr = librosa.core.load(filename, sr = 16000, dtype = "float32")
#     print(wav, sr)
    with smart_open(filename, "rb") as f:
        f.seek(-16, 2)
        nClips, nSamples, nLabels, labelOffset = struct.unpack("<4i", f.read(16))
#         print('nClips,', nClips, 'nSamples',nSamples, 'nLabels', nLabels, 'labelOffset', labelOffset)
        wav = wav.reshape(nClips, nSamples)
        nBytes = (nLabels - 1) // 8 + 1
#         f.seek(-16 - nClips * nBytes, 2)
        f.seek(labelOffset, 0)
        data = struct.unpack("<%dB" % (nClips * nBytes), f.read(nClips * nBytes))
        bytes = numpy.array(data).reshape(nClips, nBytes)
        labels = numpy.zeros((nClips, nLabels), dtype = "int32")
#         print(nLabels)
        for i in range(nLabels):
            labels[:,i] = (bytes[:, i // 8] >> (i % 8)) & 1
        hashes = struct.unpack("12p" * nClips, f.read(12 * nClips))
        print(wav.shape, labels.shape, len(hashes))
        for row, hash_i in zip(wav, hashes):
            print('row:', row.shape, 'label', hash_i)
            writeWAV(row, destFolder, hash_i.decode()+'.wav', sr)
#         for l_row in labels:
#             print(l_row.shape)
        return wav, labels

def writeWAV(data, destfolder, wav_file_name, sr):
    print(os.path.join(destfolder, wav_file_name))
    write(os.path.join(destfolder, wav_file_name), sr, data.astype(np.float32))

if __name__ == '__main__':
    rootdir = '/local/slurm-5415408/local/audio/yunwang/GoogleAudioSet/binary/'
    unbalanced_dest = '/local/slurm-5415408/local/audio/yunwang/GoogleAudioSet/unbalanced_wav/'
    balanced_dest = '/local/slurm-5415408/local/audio/yunwang/GoogleAudioSet/balance_wav/'
    valid_dest = '/local/slurm-5415408/local/audio/yunwang/GoogleAudioSet/valid_wav/'
    eval_dest = '/local/slurm-5415408/local/audio/yunwang/GoogleAudioSet/eval_wav/'
#     dcase_eval = '/local/slurm-5415408/local/audio/yunwang/dcase/eval_wav/'
#     dcase_valid = '/local/slurm-5415408/local/audio/yunwang/dcase/valid_wav/'
#     dcase_train = '/local/slurm-5415408/local/audio/yunwang/dcase/train_wav/'
    for subdir, dirs, files in os.walk(rootdir):
        ub_count = 0
        balanced_count = 0
        valid_count = 0
        eval_count = 0
        for file in files:
            if file[-4::] == 'flac':
                abs_path = os.path.join(subdir, file)
                if (file[:20] == 'GAS_train_unbalanced'):
                    writeAudioSetWAV(abs_path, unbalanced_dest)
                    print('unbalanced:', ub_count, abs_path)
                    ub_count+=1
                elif (file[:9] == 'GAS_valid'):
                    writeAudioSetWAV(abs_path, valid_dest)
                    print('valid:', valid_count, abs_path)
                    valid_count+=1
                elif (file[:8] == 'GAS_eval'):
                    writeAudioSetWAV(abs_path, eval_dest)
                    print('eval:', eval_count, abs_path)
                    eval_count+=1
                elif (file[:18] == 'GAS_train_balanced'):
                    writeAudioSetWAV(abs_path, balanced_dest)
                    print('balanced:', balanced_count, abs_path)
                    balanced_count+=1
    print('ub, bal, valid, eval', ub_count, balanced_count, valid_count, eval_count)
