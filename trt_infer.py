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


def build_trt_engine(onnx_path, mode, asr_model=None):
    trt_engine_path = "{}.trt".format(onnx_path + "."+ mode)
    if os.path.exists(trt_engine_path):
       return trt_engine_path
    min_input_shape=(1, 64, 64)
    max_input_shape=(256, 64, 3600)
    workspace_size = 4 * 1024
    with trt.Builder(TRT_LOGGER) as builder:
        network_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(
                flags=network_flags) as network, trt.OnnxParser(
                    network, TRT_LOGGER
                ) as parser, builder.create_builder_config() as builder_config:
            parser.parse_from_file(onnx_path)
            builder_config.max_workspace_size = workspace_size * (1024 * 1024)

            if mode == 'int8':
                builder.int8_mode = True
                batchstream = AudioStream(asr_model, max_input_shape)
                Int8_calibrator = PythonEntropyCalibrator(batchstream)
                builder_config.set_flag(trt.BuilderFlag.INT8)
                builder_config.int8_calibrator = Int8_calibrator
            
            elif mode == 'fp16':
                builder.fp16_mode = True
                builder_config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            profile.set_shape("audio_signal",
                              min=min_input_shape,
                              opt=max_input_shape,
                              max=max_input_shape)
                              
            builder_config.add_optimization_profile(profile)

            engine = builder.build_engine(network, builder_config)
            serialized_engine = engine.serialize()
            with open(trt_engine_path, "wb") as fout:
                fout.write(serialized_engine)
    print("Convert onnx to trt_engine successfully")
    return trt_engine_path

# @profile
def trt_infer(trt_path, wav_path):
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
    input_nbytes = trt.volume(x.shape) * trt.float32.itemsize
    d_input = cuda.mem_alloc(input_nbytes)

    trt_ctx.set_binding_shape(0, x.shape)
    assert trt_ctx.all_binding_shapes_specified

    h_output = cuda.pagelocked_empty(tuple(trt_ctx.get_binding_shape(1)),
                                     dtype=np.float32)

    h_input_signal = cuda.register_host_memory(
        np.ascontiguousarray(x.ravel()))
    cuda.memcpy_htod_async(d_input, h_input_signal, stream)
    trt_ctx.execute_async_v2(bindings=[int(d_input),
                                       int(d_output)],
                             stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    greedy_predictions = torch.tensor(h_output).argmax(dim=-1, keepdim=False)
    return greedy_predictions

# @profile
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
