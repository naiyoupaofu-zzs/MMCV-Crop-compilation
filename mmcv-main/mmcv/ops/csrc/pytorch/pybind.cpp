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