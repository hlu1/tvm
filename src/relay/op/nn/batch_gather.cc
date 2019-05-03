#include <tvm/expr.h>
#include <tvm/relay/op.h>
#include <tvm/tvm.h>
#include "../op_common.h"
#include "../type_relations.h"
#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

bool BatchGatherRel(
    const Array<Type>& types,
    int /* unused */,
    const Attrs& /* unused */,
    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* indices = types[1].as<TensorTypeNode>();

  if (data == nullptr || indices == nullptr) {
    return false;
  }

  size_t ndim_d = data->shape.size();
  size_t ndim_i = indices->shape.size();
  CHECK_GE(ndim_d, 2) << "data tensor must have at least 2 dimensions";
  CHECK_GE(ndim_i, 1) << "indices tensor must have 1 dimension";

  Array<IndexExpr> oshape;
  oshape.push_back(data->shape[0]);
  for (size_t i = 0; i < ndim_i; i++) {
    oshape.push_back(indices->shape[i]);
  }
  for (size_t i = 2; i < ndim_d; i++) {
    oshape.push_back(data->shape[i]);
  }

  CHECK(!oshape.empty()) << "output tensor cannot be empty";

  reporter->Assign(types[2], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

Expr MakeBatchGather(Expr data, Expr indices) {
  static const Op& op = Op::Get("nn.batch_gather");
  return CallNode::make(op, {data, indices}, Attrs(), {});
}

TVM_REGISTER_API("relay.op.nn._make.batch_gather")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 2>(MakeBatchGather, args, rv);
    });

RELAY_REGISTER_OP("nn.batch_gather")
    .describe(R"code(Batch gather operation.
Given DATA tensor of rank r >= 2, and INDICES tensor of rank q >= 1,
gather entries of the outer-most dimension of DATA indexed by INDICES,
and concatenate them in an output tensor of rank q + r - 1.

- **data**: tensor of rank r >= 2 with the first dimension in DATA being batch size
- **indices**: tensor of rank q >= 1
- **out**: tensor of rank q + r - 1

    )code" TVM_ADD_FILELINE)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("indices", "Tensor", "Indices of data")
    .set_num_inputs(2)
    .set_support_level(3)
    .add_type_rel("BatchGather", BatchGatherRel)
    .set_attr<FInferCorrectLayout>(
        "FInferCorrectLayout",
        ElemwiseArbitraryLayout);

} // namespace relay
} // namespace tvm
