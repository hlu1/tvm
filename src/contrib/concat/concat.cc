#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>

namespace tvm {
namespace contrib {

using namespace runtime;
namespace {
void CopyMatrix(
    const size_t itemsize,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb) {
  if (A == nullptr || B == nullptr) {
    return;
  }
  if (lda == N && ldb == N) {
    memcpy(
        static_cast<char*>(B), static_cast<const char*>(A), itemsize * N * M);
  }

  for (int i = 0; i < M; ++i) {
    memcpy(
        static_cast<char*>(B) + ldb * i * itemsize,
        static_cast<const char*>(A) + lda * i * itemsize,
        itemsize * N);
  }
}
} // namespace

TVM_REGISTER_GLOBAL("tvm.contrib.concat")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* A = args[0];
      CHECK(
          TypeMatch(A->dtype, kDLFloat, 32) ||
          TypeMatch(A->dtype, kDLFloat, 64));
      DCHECK_GE(args.size(), 3);
      const int num_inputs = args.size() - 2;
      DLTensor* in0 = args[0];
      DLTensor* out = args[args.size() - 2];
      const int axis = args[args.size() - 1];

      int before = 1, after = 1;
      for (size_t i = 0; i < in0->ndim; i++) {
        if (i == axis) {
          continue;
        } else if (i < axis) {
          before *= in0->shape[i];
        } else {
          after *= in0->shape[i];
        }
      }

      size_t output_channels = out->shape[axis];
      size_t output_offset = 0;
      auto item_size = in0->dtype.bits / 8;
      for (size_t i = 0; i < num_inputs; i++) {
        DLTensor* in = args[i];
        const auto axis_dim = in->shape[axis];
        CopyMatrix(
            item_size,
            before,
            axis_dim * after,
            in->data,
            axis_dim * after,
            static_cast<uint8_t*>(out->data) + output_offset,
            output_channels * after);
        output_offset += axis_dim * after * item_size;
      }
    });
} // namespace contrib
} // namespace tvm
