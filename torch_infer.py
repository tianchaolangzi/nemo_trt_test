import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime
import tensorrt as trt
import numpy as np

import get_feature
import sys, os
import scipy.io.wavfile as wave
import torch
import ctc_decoder


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
vocabulary_size = len(vocab)

@profile
def torch_infer(trt_path, wav_path):
    stream = cuda.Stream()
    f = open(trt_path, "rb") 
    runtime = trt.Runtime(TRT_LOGGER) 
    trt_engine = runtime.deserialize_cuda_engine(f.read())
    trt_ctx = trt_engine.create_execution_context() # 创建context用来执行推断
    
    profile_shape = trt_engine.get_profile_shape(profile_index=0, binding=0)
    max_input_shape = profile_shape[2]
    max_output_shape = [
            max_input_shape[0], vocabulary_size,
            (max_input_shape[-1] + 1) // 2
        ]
    output_nbytes = trt.volume(max_output_shape) * trt.float32.itemsize
    d_output = cuda.mem_alloc(output_nbytes)

    # 读取数据
    sample_freq, signal = wave.read(wav_path)
    params = {
        "input_type":"logfbank", 
        "backend":"librosa", 
        "num_audio_features": 64, 
        "window_size": 20e-3, 
        "window_stride":10e-3, 
        "window": "hanning", 
        "dither": 1e-5, 
        "norm_per_feature": True
    }
    features, duration = get_feature.get_speech_features(signal, sample_freq, params)
    x = features.T.astype(trt.nptype(trt.float32))
    x = np.expand_dims(x, 0)
    

    greedy_predictions = torch.tensor(h_output).argmax(dim=-1, keepdim=False)
    return greedy_predictions

@profile
def test():
    onnx_path = sys.argv[1]
    wav_path = sys.argv[2]
    mode = 'fp16'
    trt_path = build_trt_engine(onnx_path, mode, asr_model=None)
    predictions = trt_infer(trt_path, wav_path)
    hypotheses = ctc_decoder.post_process_predictions([predictions], vocab)
    print("hypotheses: ", hypotheses)


if __name__ == "__main__":
    test()
