// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "topi/detail/extern.h"
#include "topi/tags.h"
#include "tvm/ir_pass.h"
#include "tvm/tvm.h"

namespace topi {
namespace contrib {
using namespace tvm;
using namespace topi::detail;

inline Tensor concatenate(
    const Array<Tensor>& inputs,
    int axis,
    std::string name = "T_concat",
    std::string tag = "opaque") {
  int ndim = static_cast<int>(inputs[0]->shape.size());
  CHECK(-ndim <= axis && axis < ndim)
      << "concatenate only accepts `axis` in [-ndim, ndim)"
      << ", but got axis = " << axis << ", and ndim = " << ndim;
  if (axis < 0) {
    axis += ndim;
  }
  CHECK_LT(axis, inputs[0]->shape.size()) << "axis out of bounds";

  Array<Expr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }

  Expr join_size = axis_sizes[0];
  for (size_t i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  join_size = tvm::ir::Simplify(join_size);
  Array<Expr> out_shape;
  for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(
        i == static_cast<size_t>(axis) ? join_size : inputs[0]->shape[i]);
  }

  return make_extern(
      {out_shape},
      {inputs[0]->dtype},
      inputs,
      [&](Array<Buffer> ins, Array<Buffer> outs) {
        Array<Expr> args;
        args.push_back(Expr("tvm.contrib.concat"));
        for (auto& in : ins) {
          args.push_back(pack_buffer(in));
        }
        args.push_back(pack_buffer(outs[0]));
        args.push_back(Expr(axis));
        return call_packed(args);
      },
      "C",
      tag,
      {})[0];
}
} // namespace contrib
} // namespace topi
