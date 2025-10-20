# Model Inference Profiler

A PyTorch-based tool for profiling deep learning model inference performance, analyzing computational bottlenecks, and visualizing resource utilization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamfaham/model-inference-profiler/blob/main/model_inference_profiler.ipynb)

## Overview

This project provides a comprehensive profiling solution for PyTorch models, enabling you to:
- Analyze inference performance on GPU (CUDA)
- Identify computational bottlenecks
- Visualize layer-wise execution time
- Monitor memory usage patterns
- Profile any pretrained torchvision model

## Features

- **Performance Profiling**: Track CPU and CUDA execution times for each operation
- **Memory Analysis**: Monitor memory allocation and usage across layers
- **Visual Analytics**: Generate bar charts for top time-consuming and memory-intensive operations
- **Model Summary**: Display detailed architecture information with parameter counts
- **Easy Integration**: Works with any PyTorch model from torchvision or custom models

## Requirements

```bash
torch
torchvision
torchinfo
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/iamfaham/model-inference-profiler.git
cd model-inference-profiler
```

2. Install dependencies:
```bash
pip install torch torchvision torchinfo matplotlib
```

## Usage

### Running in Google Colab

Click the "Open in Colab" badge at the top to run the notebook directly in Google Colab with free GPU access.

### Local Execution

1. Open the Jupyter notebook:
```bash
jupyter notebook model_inference_profiler.ipynb
```

2. Ensure GPU is available:
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

3. The notebook will guide you through:
   - Loading a pretrained model (default: ViT-B/16)
   - Running warm-up iterations
   - Profiling inference
   - Visualizing results

### Customizing the Model

To profile a different model, simply change the model loading line:

```python
# Vision Transformer (default)
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# Or try other models:
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
# model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
```

## Output Examples

The profiler generates:

1. **Model Summary**: Detailed architecture breakdown with parameter counts and memory estimates
2. **Performance Table**: Top operations sorted by CUDA execution time
3. **CUDA Time Visualization**: Bar chart showing the most time-consuming layers
4. **Memory Usage Visualization**: Bar chart displaying memory allocation per layer

## How It Works

1. **Model Loading**: Loads a pretrained model and moves it to GPU
2. **Warm-up**: Runs multiple inference passes to stabilize GPU performance
3. **Profiling**: Uses PyTorch's built-in profiler to capture:
   - CPU and CUDA activities
   - Operation shapes
   - Memory allocations
4. **Analysis**: Extracts and visualizes performance metrics

## Use Cases

- **Model Optimization**: Identify bottlenecks before deployment
- **Hardware Selection**: Understand resource requirements
- **Comparative Analysis**: Compare different architectures
- **Educational**: Learn about model internals and performance characteristics

## Example Output

For ViT-B/16, you'll see:
- Total parameters: ~86.5M
- Top operations: Matrix multiplications (addmm, sgemm)
- Memory-intensive layers: Attention mechanisms and linear layers

## Contributing

Contributions are welcome! Feel free to:
- Add support for more profiling metrics
- Implement additional visualization options
- Extend to other frameworks
- Improve documentation

## License

This project is open source and available under the MIT License.

## Author

Created by [@iamfaham](https://github.com/iamfaham)

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Model summaries powered by [torchinfo](https://github.com/TylerYep/torchinfo)
- Pretrained models from [torchvision](https://pytorch.org/vision/stable/index.html)