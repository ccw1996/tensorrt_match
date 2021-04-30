import os
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import common


TRT_LOGGER = trt.Logger()




def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder,\
             builder.create_network(common.EXPLICIT_BATCH) as network,\
             builder.create_builder_config() as config,\
             trt.OnnxParser(network, TRT_LOGGER) as parser:
            config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 874,800]
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, (1,3, 400, 400), (1,3, 800, 800), (1,3, 4000, 4000))
            config.add_optimization_profile(profile)
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def resize_im(w, h, scale=800, max_scale=4000):
    f = float(scale) / min(h, w)
    if max_scale is not None:
        if f * max(h, w) > max_scale:
            f = float(max_scale) / max(h, w)
    newW, newH = int(w * f), int(h * f)

    return newW - (newW % 32), newH - (newH % 32)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers2(engine,h_,w_):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    print('engine.get_binding_format_desc',engine.get_binding_format_desc(0))
    for count,binding in enumerate(engine):
        print('binding:',binding)
        if engine.binding_is_input(binding):
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size*(int)(h_)*(int)(w_)
            #size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            #dtype=np.float16
            print('dtype:',dtype)
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size *28*25
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            #dtype=np.float16
            print('dtype:',dtype)
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))

        print('size:',size)
        print('input:',inputs)
        print('output:',outputs)
        print('------------------')
    return inputs, outputs, bindings, stream

def main():
    onnx_file_path = 'test3.onnx'
    engine_file_path = "model_engine.trt"
    input_image_path="../yoloF_test/YOLOF/datasets/coco/val2017/000000000285.jpg"
    image_raw=Image.open(input_image_path)
    w, h = image_raw.size
    w_, h_ = resize_im(w, h, scale=800, max_scale=4000)
    print(w_,h_)
    image_resized=image_raw.resize((w_,h_),resample=Image.BICUBIC)
    image_resized = np.array(image_resized, dtype=np.int32, order='C')
    
    output_shapes = [(1, 512, 28, 25)]
    trt_outputs = []
    inputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs,outputs, bindings, stream = allocate_buffers2(engine,w_, h_)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image_resized
        context.set_binding_shape(0, (1, 3, h_, w_))
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    print(trt_outputs[0])


main()
# engine = get_engine(onnx_file_path, engine_file_path)
