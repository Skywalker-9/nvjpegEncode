/*
 *
 * To Generate rgb stream
 * gst-launch-1.0 filesrc location= sample_720p.yuv ! videoparse width=1280 height=720 format=i420 ! nvvideoconvert ! "video/x-raw, format=GBR" ! filesink location= sample_720p.rgb
 *
 * To Compile this app
 * g++ -g -m64 simple_nvjpeg_encode.cpp -lnvjpeg -I/usr/local/cuda-11.8/targets/x86_64-linux/include -ldl -lrt -pthread -lcudart -L/usr/local/cuda-11.8/lib64  -I./
 *
 * */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <string.h>
#include "nvjpeg.h"
#include <cuda_runtime_api.h>

typedef enum _input_pixel_format
{
    YUV = 1,
    RGB
}input_pixel_format;

//input_pixel_format ipixfmt = YUV;
input_pixel_format ipixfmt = RGB;

static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
  switch (error) {
    case NVJPEG_STATUS_SUCCESS:
      return "NVJPEG_STATUS_SUCCESS";

    case NVJPEG_STATUS_NOT_INITIALIZED:
      return "NVJPEG_STATUS_NOT_INITIALIZED";

    case NVJPEG_STATUS_INVALID_PARAMETER:
      return "NVJPEG_STATUS_INVALID_PARAMETER";

    case NVJPEG_STATUS_BAD_JPEG:
      return "NVJPEG_STATUS_BAD_JPEG";

    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
      return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

    case NVJPEG_STATUS_EXECUTION_FAILED:
      return "NVJPEG_STATUS_EXECUTION_FAILED";

    case NVJPEG_STATUS_ARCH_MISMATCH:
      return "NVJPEG_STATUS_ARCH_MISMATCH";

    case NVJPEG_STATUS_INTERNAL_ERROR:
      return "NVJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

struct encode_params_t {
  std::string input_dir;
  std::string output_dir;
  std::string format;
  std::string subsampling;
  int quality;
  int huf;
  int dev;
};

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }

nvjpegEncoderParams_t encode_params;
nvjpegHandle_t nvjpeg_handle;
nvjpegJpegState_t jpeg_state;
nvjpegEncoderState_t encoder_state;

int main(int argc, const char *argv[])
{
    encode_params_t params;
    params.huf = 0;
    params.subsampling = "420";
    params.format = "yuv";
    params.dev = 0;
    params.quality = 100;

#if 0
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));
#endif

    if (!strcmp(argv[1], "YUV"))
        ipixfmt = YUV;
    else
        ipixfmt = RGB;

    std::string input_file_name;
    if (ipixfmt == YUV)
        input_file_name.assign("sample_720p.yuv");
    if (ipixfmt == RGB)
        input_file_name.assign("sample_720p.rgb");

    std::ifstream file(input_file_name.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
    {
        /* worked! */
    }
    else
        std::cout << "couldn't read input yuv file" << std::endl;

    cudaDeviceProp props;
    //checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));
    cudaGetDeviceProperties(&props, params.dev);

    unsigned char * pBuffer = NULL; 

    nvjpegChromaSubsampling_t subsampling;
    if (ipixfmt == YUV)
        subsampling = NVJPEG_CSS_420;
    if (ipixfmt == RGB)
        subsampling = NVJPEG_CSS_444;

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    widths[0] = 1280;
    heights[0] = 720;

    cudaError_t eCopy = cudaMalloc((void **)&pBuffer, widths[0] * heights[0] * NVJPEG_MAX_COMPONENT);
    if(cudaSuccess != eCopy) 
    {
        std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(eCopy) << std::endl;
        return 1;
    }

    cudaMemcpy (pBuffer, buffer.data(), size, cudaMemcpyHostToDevice);


    nvjpegImage_t imgdesc;
    if (ipixfmt == YUV)
    {
        imgdesc =
        {
            {
                pBuffer,
                pBuffer + widths[0]*heights[0],
                pBuffer + widths[0]*heights[0] + (widths[0]*heights[0] / 4)/*,
                pBuffer + widths[0]*heights[0]*3*/
            },
            {
                (unsigned int)widths[0],
                (unsigned int)widths[0]/2,
                (unsigned int)widths[0]/2/*,
                (unsigned int)widths[0]*/
            }
        };
    }

    if (ipixfmt == RGB)
    {
        imgdesc = 
        {
            {
                pBuffer,
                pBuffer + widths[0]*heights[0],
                pBuffer + widths[0]*heights[0]*2/*,
                pBuffer + 3*/
            },
            {
                (unsigned int)widths[0],
                (unsigned int)widths[0],
                (unsigned int)widths[0]/*,
                (unsigned int)widths[0]*/
            }
        };
    }

    int nReturnCode = 0;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));
    checkCudaErrors(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL));
    checkCudaErrors(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL));

    // sample input parameters
    checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, params.quality, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, params.huf, NULL));

    nvjpegOutputFormat_t oformat;
    nvjpegInputFormat_t input_format;
    if (ipixfmt == YUV)
         oformat = NVJPEG_OUTPUT_YUV;
    if (ipixfmt == RGB)
    {
        oformat = NVJPEG_OUTPUT_RGB;
        input_format = NVJPEG_INPUT_RGB;
    }

    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, subsampling, NULL));

    if (ipixfmt == YUV)
    {
        checkCudaErrors(nvjpegEncodeYUV(nvjpeg_handle,
                    encoder_state,
                    encode_params,
                    &imgdesc,
                    subsampling,
                    widths[0],
                    heights[0],
                    NULL));
    }

    if (ipixfmt == RGB)
    {
        checkCudaErrors(nvjpegEncodeImage(nvjpeg_handle,
                    encoder_state,
                    encode_params,
                    &imgdesc,
                    input_format,
                    widths[0],
                    heights[0],
                    NULL));
    }

    std::vector<unsigned char> obuffer;
    size_t length;
    checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                nvjpeg_handle,
                encoder_state,
                NULL,
                &length,
                NULL));

    obuffer.resize(length);
    checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                nvjpeg_handle,
                encoder_state,
                obuffer.data(),
                &length,
                NULL));

    std::cout << "Writing JPEG file: sample.jpg"  << std::endl;
    std::ofstream outputFile("sample.jpg", std::ios::out | std::ios::binary);
    outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));

    //checkCudaErrors(cudaFree(pBuffer));
    cudaFree(pBuffer);

#if 0
    checkCudaErrors(cudaEventRecord(stopEvent, NULL));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
#endif

    checkCudaErrors(nvjpegEncoderParamsDestroy(encode_params));
    checkCudaErrors(nvjpegEncoderStateDestroy(encoder_state));
    checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
    checkCudaErrors(nvjpegDestroy(nvjpeg_handle));
}
