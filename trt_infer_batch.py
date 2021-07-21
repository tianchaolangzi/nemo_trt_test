from argparse import ArgumentParser
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import nemo
import nemo.collections.asr as nemo_asr
import yaml
import torch
import numpy as np
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




def get_parser():
    # Usage and Command line arguments
    parser = ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size to use for evaluation")
    parser.add_argument("--wer_target", type=float, default=None, help="used by test")
    parser.add_argument("--trim_silence", default=True, type=bool, help="trim audio from silence or not")
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--amp_opt_level", default="O1", type=str)
    parser.add_argument("--lm_path", default=None, type=str)
    parser.add_argument('--alpha', default=1.0, type=float, help='value of LM weight', required=False)
    parser.add_argument('--alpha_max', type=float,
                        help='maximum value of LM weight (for a grid search in \'eval\' mode)', required=False, )
    parser.add_argument('--alpha_step', type=float,
                        help='step for LM weight\'s tuning in \'eval\' mode', required=False, default=0.1)
    parser.add_argument('--beta', default=1.5, type=float, help='value of word count weight', required=False)
    parser.add_argument('--beta_max', type=float,
                        help='maximum value of word count weight (for a grid search in \'eval\' mode', required=False, )
    parser.add_argument('--beta_step', type=float,
                        help='step for word count weight\'s tuning in \'eval\' mode', required=False, default=0.1, )
    parser.add_argument("--beam_width", default=80, type=int)
    return parser


class ModelASR():
    def __init__(self, args):

        self.load_dir = args.load_dir
        if args.local_rank is not None:
            if args.lm_path:
                raise NotImplementedError(
                    "Beam search decoder with LM does not currently support evaluation on multi-gpu."
                )
            device = nemo.core.DeviceType.AllGpu
        else:
            device = nemo.core.DeviceType.GPU

        with open(args.model_config) as f:
            jasper_params = yaml.load(f)
        self.vocab = jasper_params['labels']
        self.sample_rate = jasper_params['sample_rate']

        # Setup NeuralModuleFactory to control training
        # instantiate Neural Factory with supported backend
        self.neural_factory = nemo.core.NeuralModuleFactory(
            local_rank=args.local_rank,
            optimization_level=args.amp_opt_level,
            placement=device,
        )

        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            sample_rate=self.sample_rate, **jasper_params["AudioToMelSpectrogramPreprocessor"]
        )
        # self.jasper_encoder = nemo_asr.JasperEncoder(
        #     feat_in=jasper_params["AudioToMelSpectrogramPreprocessor"]["features"], **jasper_params["JasperEncoder"]
        # )
        # self.jasper_decoder = nemo_asr.JasperDecoderForCTC(
        #     feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"], num_classes=len(self.vocab)
        # )
        # self.greedy_decoder = nemo_asr.GreedyCTCDecoder()

        # logging.info('================================')
        # logging.info(f"Number of parameters in encoder: {self.jasper_encoder.num_weights}")
        # logging.info(f"Number of parameters in decoder: {self.jasper_decoder.num_weights}")
        # logging.info(f"Total number of parameters in model: " f"{self.jasper_decoder.num_weights + self.jasper_encoder.num_weights}")
        # logging.info('================================')

        # self.language_model = BeamSearchDecoderWithLM(
        #     vocab=self.vocab,
        #     beam_width=args.beam_width,
        #     alpha=args.alpha,
        #     beta=args.beta,
        #     lm_path=args.lm_path,
        #     num_cpus=max(os.cpu_count(), 1),
        #     input_tensor=False,
        # )

    @profile
    def get_features(self, json_file, wav_path=None):
        eval_data_layer = nemo_asr.AudioToTextDataLayer(
            manifest_filepath=json_file,
            labels=self.vocab,
            batch_size=64,
            sample_rate=self.sample_rate,
            shuffle=False
        )
        audio_signal_e1, a_sig_length_e1, transcript_e1, transcript_len_e1 = eval_data_layer()
        processed_signal_e1, p_length_e1 = self.data_preprocessor(
            input_signal=audio_signal_e1, length=a_sig_length_e1)
        feature_tensors = self.neural_factory.infer(
            tensors=[processed_signal_e1])[0]
        
        for i in range(len(feature_tensors)):
            tmp_tensor = torch.zeros((feature_tensors[i].shape[0], 64, 1600))
            tmp_tensor[:, :, :feature_tensors[i].shape[2]] += feature_tensors[i]
            feature_tensors[i] = tmp_tensor
        feature_tensors = torch.cat(feature_tensors, dim=0)
        return feature_tensors


@profile
def trt_infer(trt_path, feature_tensors, batch_size=64):
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

    predictions = []
    for step in range(feature_tensors.shape[0] // batch_size + 1):
        if (step+1)*batch_size < feature_tensors.shape[0]:
            input_features = feature_tensors[step*batch_size:(step+1)*batch_size]
        else:
            input_features = feature_tensors[step*batch_size:]

        input_nbytes = trt.volume(input_features.shape) * trt.float32.itemsize
        d_input = cuda.mem_alloc(input_nbytes)

        trt_ctx.set_binding_shape(0, input_features.shape)
        assert trt_ctx.all_binding_shapes_specified

        h_output = cuda.pagelocked_empty(tuple(trt_ctx.get_binding_shape(1)),
                                        dtype=np.float32)

        h_input_signal = cuda.register_host_memory(
            np.ascontiguousarray(input_features.ravel()))
        cuda.memcpy_htod_async(d_input, h_input_signal, stream)
        trt_ctx.execute_async_v2(bindings=[int(d_input),
                                        int(d_output)],
                                stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        predictions += torch.tensor(h_output).argmax(dim=-1, keepdim=False)
    
    predictions = torch.stack(predictions)
    hypotheses = ctc_decoder.post_process_predictions([predictions], vocab)
    return hypotheses




@profile
def main():
    onnx_path = sys.argv[1]
    mode = 'fp16'
    trt_path = build_trt_engine(onnx_path, mode)
    model_args = [
        '--model_config', 'examples/asr/configs/quartznet15x5.yaml',
        '--load_dir', 'model/english/',
        '--lm_path', './language-model/english_4gram_1.2.binary',
        '--eval_batch_size', '1',
        '--beam_width', '64',
        '--amp_opt_level', 'O1'
    ]
    parser = get_parser()
    args = parser.parse_args(model_args)
    asr_model = ModelASR(args)
    # trt_path = sys.argv[1]
    json_file = sys.argv[2]
    predict_file = sys.argv[3]
    feature_tensors = asr_model.get_features(json_file)
    # print(feature_tensors.shape)
    # print(feature_tensors)
    transcripts = trt_infer(trt_path, feature_tensors)
    with open(predict_file, 'w') as f:
        for transcript in transcripts:
            f.write(transcript + '\n')


if __name__ == "__main__":
    main()
