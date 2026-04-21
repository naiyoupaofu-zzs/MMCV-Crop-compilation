# MMCV 裁剪编译 `ModulatedDeformConv2d` 指南
**适用环境：Windows + Conda + PyTorch + CUDA**
由于较新版本的pytorch与mmcv不兼容导致无法正常编译，但是又需要其中的一部分内容，因此尝试裁剪编译部分以供使用
> 目标  
> 不安装完整 `mmcv`，只保留 `deform_conv / modulated_deform_conv`，让：
>
> - `from mmcv.ops import ModulatedDeformConv2d` 可导入
> - `ModulatedDeformConv2d(...).cuda()` 可正常前向

---

## 目录

- [1. 适用场景](#1-适用场景)
- [2. 环境前提](#2-环境前提)
- [3. 源码目录](#3-源码目录)
- [4. 修改 `setup.py`](#4-修改-setuppy)
- [5. 修改 `pybind.cpp`](#5-修改-pybindcpp)
- [6. 修改 `cudabind.cpp`](#6-修改-cudabindcpp)
- [7. 重新编译安装 MMCV](#7-重新编译安装-mmcv)
- [8. 修改 `site-packages` 中的 `__init__.py`](#8-修改-site-packages-中的-__init__py)
- [9. 验证是否真正可用](#9-验证是否真正可用)
- [10. 重要说明](#10-重要说明)
- [11. 建议备份的文件](#11-建议备份的文件)

---

## 1. 适用场景

这套流程适合：

- Windows 环境
- Anaconda 虚拟环境，例如 `yolo26`
- 已安装 PyTorch 和 CUDA 运行时
- 只需要 DyHead / DCN 相关算子
- 不追求完整 `mmcv` 全家桶

最终效果：

- `from mmcv.ops import ModulatedDeformConv2d` 可用
- `ModulatedDeformConv2d(...).cuda()` 可正常前向
- 但以下算子**不保证可用**：
  - `roi_align`
  - `nms`
  - `active_rotated_filter`
  - `ms_deform_attn`

---

## 2. 环境前提

### 2.1 激活环境

```bat
conda activate yolo26
```

### 2.2 需要具备

- Visual Studio 2022 Build Tools
- `cl.exe` 可用
- `nvcc` 可用
- `CUDA_HOME` 指向本机安装的 CUDA Toolkit

### 2.3 建议检查

```bat
where cl
where nvcc
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
```

---

## 3. 源码目录

示例目录：

```text
E:\mmcv_min\mmcv-main
```

下文默认源码目录就是这个。

---

## 4. 修改 `setup.py`

文件路径：

```text
E:\mmcv_min\mmcv-main\setup.py
```

### 4.1 添加过滤函数

在 `get_extensions()` 中加入：

```python
def _keep_minimal_dcn_op(path):
    p = path.replace("\\", "/")
    keep = [
        "/pytorch/deform_conv.cpp",
        "/pytorch/modulated_deform_conv.cpp",
        "/pytorch/pybind.cpp",

        "/pytorch/cpu/deform_conv.cpp",
        "/pytorch/cpu/modulated_deform_conv.cpp",

        "/pytorch/cuda/deform_conv_cuda.cu",
        "/pytorch/cuda/modulated_deform_conv_cuda.cu",
        "/pytorch/cuda/cudabind.cpp",
    ]
    return any(k in p for k in keep)
```

### 4.2 过滤 `op_files`

把 `all_op_files` 和 `op_files` 逻辑改成：

```python
all_op_files = (
    glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') +
    glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') +
    glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') +
    glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp')
)

op_files = [p for p in all_op_files if _keep_minimal_dcn_op(p)]

print("=" * 80)
print("Using minimal MMCV DCN build:")
for p in op_files:
    print(p)
print("=" * 80)
```

### 4.3 作用

只编译 DCN 需要的最小文件集，避免完整 `mmcv` 的其他 CUDA 扩展一起参与编译。

---

## 5. 修改 `pybind.cpp`

文件路径：

```text
E:\mmcv_min\mmcv-main\mmcv\ops\csrc\pytorch\pybind.cpp
```

### 5.1 整个替换为以下内容

```cpp
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// deform conv
void deform_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor output,
    at::Tensor columns,
    at::Tensor ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH,
    int group,
    int deformable_group,
    int im2col_step);

void deform_conv_backward_input(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradInput,
    at::Tensor gradOffset,
    at::Tensor weight,
    at::Tensor columns,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH,
    int group,
    int deformable_group,
    int im2col_step);

void deform_conv_backward_parameters(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor gradOutput,
    at::Tensor gradWeight,
    at::Tensor columns,
    at::Tensor ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH,
    int group,
    int deformable_group,
    float scale,
    int im2col_step);

// modulated deform conv
void modulated_deform_conv_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor output,
    at::Tensor columns,
    int kernel_h, int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias);

void modulated_deform_conv_backward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor ones,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor columns,
    at::Tensor grad_output,
    at::Tensor grad_input,
    at::Tensor grad_weight,
    at::Tensor grad_bias,
    at::Tensor grad_offset,
    at::Tensor grad_mask,
    int kernel_h, int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int group,
    const int deformable_group,
    const bool with_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "deform_conv_forward", &deform_conv_forward,
        py::arg("input"), py::arg("weight"), py::arg("offset"),
        py::arg("output"), py::arg("columns"), py::arg("ones"),
        py::arg("kW"), py::arg("kH"), py::arg("dW"), py::arg("dH"),
        py::arg("padW"), py::arg("padH"),
        py::arg("dilationW"), py::arg("dilationH"),
        py::arg("group"), py::arg("deformable_group"),
        py::arg("im2col_step")
    );

    m.def(
        "deform_conv_backward_input", &deform_conv_backward_input,
        py::arg("input"), py::arg("offset"), py::arg("gradOutput"),
        py::arg("gradInput"), py::arg("gradOffset"),
        py::arg("weight"), py::arg("columns"),
        py::arg("kW"), py::arg("kH"), py::arg("dW"), py::arg("dH"),
        py::arg("padW"), py::arg("padH"),
        py::arg("dilationW"), py::arg("dilationH"),
        py::arg("group"), py::arg("deformable_group"),
        py::arg("im2col_step")
    );

    m.def(
        "deform_conv_backward_parameters", &deform_conv_backward_parameters,
        py::arg("input"), py::arg("offset"), py::arg("gradOutput"),
        py::arg("gradWeight"), py::arg("columns"), py::arg("ones"),
        py::arg("kW"), py::arg("kH"), py::arg("dW"), py::arg("dH"),
        py::arg("padW"), py::arg("padH"),
        py::arg("dilationW"), py::arg("dilationH"),
        py::arg("group"), py::arg("deformable_group"),
        py::arg("scale"), py::arg("im2col_step")
    );

    m.def(
        "modulated_deform_conv_forward", &modulated_deform_conv_forward,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("ones"), py::arg("offset"), py::arg("mask"),
        py::arg("output"), py::arg("columns"),
        py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("pad_h"), py::arg("pad_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("group"), py::arg("deformable_group"),
        py::arg("with_bias")
    );

    m.def(
        "modulated_deform_conv_backward", &modulated_deform_conv_backward,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("ones"), py::arg("offset"), py::arg("mask"),
        py::arg("columns"), py::arg("grad_output"),
        py::arg("grad_input"), py::arg("grad_weight"), py::arg("grad_bias"),
        py::arg("grad_offset"), py::arg("grad_mask"),
        py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"),
        py::arg("pad_h"), py::arg("pad_w"),
        py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("group"), py::arg("deformable_group"),
        py::arg("with_bias")
    );
}
```

---

## 6. 修改 `cudabind.cpp`

文件路径：

```text
E:\mmcv_min\mmcv-main\mmcv\ops\csrc\pytorch\cuda\cudabind.cpp
```

### 6.1 整个替换为以下内容

```cpp
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

// impl symbols
void modulated_deformable_im2col_impl(
    const Tensor data_im,
    const Tensor data_offset,
    const Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    Tensor data_col);

void modulated_deformable_col2im_impl(
    const Tensor data_col,
    const Tensor data_offset,
    const Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    Tensor grad_im);

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col,
    const Tensor data_im,
    const Tensor data_offset,
    const Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    Tensor grad_offset,
    Tensor grad_mask);

// actual CUDA symbols
void modulated_deformable_im2col_cuda(
    const Tensor data_im,
    const Tensor data_offset,
    const Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    Tensor data_col);

void modulated_deformable_col2im_cuda(
    const Tensor data_col,
    const Tensor data_offset,
    const Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(
    const Tensor data_col,
    const Tensor data_im,
    const Tensor data_offset,
    const Tensor data_mask,
    const int batch_size,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col,
    const int width_col,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    Tensor grad_offset,
    Tensor grad_mask);

REGISTER_DEVICE_IMPL(modulated_deformable_im2col_impl, CUDA, modulated_deformable_im2col_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_impl, CUDA, modulated_deformable_col2im_cuda);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, CUDA, modulated_deformable_col2im_coord_cuda);
```

---

## 7. 重新编译安装 MMCV

在 **VS2022 开发者命令行** 中执行：

```bat
conda activate yolo26

set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%
set CL=/Zc:preprocessor
set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler
set MAX_JOBS=1

cd /d E:\mmcv_min\mmcv-main
rmdir /s /q build
rmdir /s /q mmcv.egg-info
pip uninstall -y mmcv
pip install --no-build-isolation -v .
```

---

## 8. 修改 `site-packages` 中的 `__init__.py`

文件路径：

```text
E:\Anaconda\envs\yolo26\Lib\site-packages\mmcv\ops\__init__.py
```

### 8.1 整个替换为以下内容

```python
from .deform_conv import DeformConv2d, DeformConv2dPack, deform_conv2d
from .modulated_deform_conv import ModulatedDeformConv2d, ModulatedDeformConv2dPack, modulated_deform_conv2d

__all__ = [
    "DeformConv2d",
    "DeformConv2dPack",
    "deform_conv2d",
    "ModulatedDeformConv2d",
    "ModulatedDeformConv2dPack",
    "modulated_deform_conv2d",
]
```

### 8.2 原因

你编的是裁剪版 `_ext`，所以 Python 层入口也必须裁剪。  
否则它会去导入没编进去的其他算子，然后报错。

---

## 9. 验证是否真正可用

### 9.1 导入验证

```bat
cd /d E:\
python -c "from mmcv.ops import ModulatedDeformConv2d; print('ModulatedDeformConv2d OK')"
```

### 9.2 CUDA 前向验证

```bat
python -c "import torch; from mmcv.ops import ModulatedDeformConv2d; m=ModulatedDeformConv2d(256,256,3,stride=1,padding=1).cuda(); x=torch.randn(2,256,32,32,device='cuda'); offset=torch.randn(2,18,32,32,device='cuda'); mask=torch.sigmoid(torch.randn(2,9,32,32,device='cuda')); y=m(x,offset,mask); print('forward ok:', y.shape)"
```

### 9.3 正确结果示例

```text
forward ok: torch.Size([2, 256, 32, 32])
```

如果看到这个结果，说明 MMCV 的 DCN 这条链已经真正打通。

---

## 10. 重要说明

这套方案得到的是：

- **裁剪版 mmcv**
- 只保证 `deform_conv / modulated_deform_conv` 可用
- 不保证以下功能可用：
  - `active_rotated_filter`
  - `roi_align`
  - `nms`
  - `ms_deform_attn`
  - 其他未保留的 `mmcv ops`

如果你的目标只是支持 DyHead / DCN，这已经够了。

---

## 11. 建议备份的文件

建议把下面这些文件单独备份：

```text
E:\mmcv_min\mmcv-main\setup.py
E:\mmcv_min\mmcv-main\mmcv\ops\csrc\pytorch\pybind.cpp
E:\mmcv_min\mmcv-main\mmcv\ops\csrc\pytorch\cuda\cudabind.cpp
E:\Anaconda\envs\yolo26\Lib\site-packages\mmcv\ops\__init__.py
```

建议另建一个目录，例如：

```text
E:\backup_mmcv_dcn_fixed
```

把这些文件都复制进去，后续重装环境时会省很多时间。
