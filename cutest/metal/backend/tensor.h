#pragma once

#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cutest::metal {

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    static Tensor zeros(Shape shape) { return Tensor(std::move(shape), 0.0f); }
    static Tensor ones(Shape shape) { return Tensor(std::move(shape), 1.0f); }

    const Shape& shape() const { return shape_; }
    id<MTLBuffer> buffer() const { return buffer_; }

private:
    Tensor(Shape shape, float value) : shape_(std::move(shape)) {
        // Zero-length tensors need no allocation; an empty shape is a scalar.
        if (std::find(shape_.begin(), shape_.end(), 0) != shape_.end()) return;

        static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal device unavailable");

        const auto max_elements = device.maxBufferLength / sizeof(float);
        std::size_t elements = 1;
        for (auto dim : shape_) {
            if (elements > max_elements / dim) throw std::length_error("Metal tensor too large");
            elements *= dim;
        }
        buffer_ = [device newBufferWithLength:elements * sizeof(float)
                                       options:MTLResourceStorageModeShared];
        if (!buffer_) throw std::runtime_error("Metal tensor allocation failed");
        std::fill_n(static_cast<float*>(buffer_.contents), elements, value);
    }

    Shape shape_;
    id<MTLBuffer> buffer_ = nil;
};

} // namespace cutest::metal
