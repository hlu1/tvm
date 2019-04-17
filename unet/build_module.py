from __future__ import absolute_import, division, print_function, unicode_literals
import tvm
from nnvm import graph as _graph
from nnvm._base import _all_var_init
from nnvm.compiler import graph_attr, graph_util
from nnvm.compiler.build_module import (
    BuildConfig,
    _remove_noref_params,
    _run_graph,
    _update_shape_dtype,
    initialize_variables,
    optimize,
    precompute_prune,
)
from tvm import autotvm


def _get_pregraph(graph, params):
    graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
    graph._set_json_attr("param_name_list", list(params.keys()), "list_str")
    graph = graph.apply("PrecomputePrune")
    pre_graph = graph_attr._move_out_graph(graph, "precompute_graph")
    if pre_graph is None:
        return None, graph, params
    out_names = pre_graph.json_attr("output_names")
    if not pre_graph.symbol.list_output_names():
        return None, graph, params
    with tvm.build_config(auto_unroll_max_step=0):
        out_arrs = _run_graph(pre_graph, params)
    return pre_graph, graph, dict(zip(out_names, out_arrs))


def build(
    graph,
    target=None,
    shape=None,
    dtype="float32",
    params=None,
    target_host=None,
    layout=None,
):
    """Build graph into runtime library.

    The build function will optimize the graph and do the compilation.

    When params is provided, the compiler might split the graph to
    pre-compute certain values, so the final execution graph can
    be different from the original one.

    Parameters
    ----------
    graph : Graph
        The graph to be used in lowering

    target : str or :any:`tvm.target.Target`, optional
        The build target

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for pre-compute
        folding optimization.

    target_host : str or :any:`tvm.target.Target` optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    layout : dict of str to str or str optional
        The input layout

    Returns
    -------
    graph : Graph
        The final execution graph.

    libmod : tvm.Module
        The module that comes with the execution graph

    params : dict of str to NDArray
        The updated parameters of graph if params is passed.
        This can be different from the params passed in.
    """
    target = target if target else tvm.target.current_target()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")
    target = tvm.target.create(target)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(target)
    else:
        tophub_context = autotvm.util.EmptyContext()

    with tophub_context:
        shape = shape if shape else {}
        if not isinstance(shape, dict):
            raise TypeError("require shape to be dict")
        for value in shape.values():
            if not all(isinstance(x, int) for x in value):
                raise TypeError("shape value must be int iterator")

        cfg = BuildConfig.current
        graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
        shape, dtype = _update_shape_dtype(shape, dtype, params)

        # correct layout if necessary
        layout = layout if layout else {}
        graph = graph_attr.set_layout_inputs(graph, layout)
        graph = graph.apply("CorrectLayout")
        index = graph.index
        layouts = graph.json_attr("layout")
        layout = {x: layouts[index.entry_id(x)] for x in index.input_names}

        # Initial pass do shape type inference
        ishape, _ = graph_util.infer_shape(graph, **shape)
        shape.update(zip(graph.index.input_names, ishape))
        if not isinstance(dtype, str):
            idtype, _ = graph_util.infer_dtype(graph, **dtype)
            dtype.update(zip(graph.index.input_names, idtype))
        # Initialize all variables specified in _all_var_init
        init_var = {}
        if _all_var_init:
            init_var = initialize_variables(shape, dtype)
        # Apply optimization
        with target:
            graph = optimize(graph, shape, dtype, layout)

        # Clear extra params without nodes.
        _remove_noref_params(params, graph)

        # Precompute prune
        pre_graph = None
        if params and cfg.pass_enabled("PrecomputePrune"):
            pre_graph, graph, params = _get_pregraph(graph, params)
            shape, dtype = _update_shape_dtype(shape, dtype, params)

        # Operator Fusion and generation
        graph = graph_attr.set_shape_inputs(graph, shape)
        graph = graph.apply("InferShape")
        graph = graph_attr.set_dtype_inputs(graph, dtype)
        graph._set_json_attr("target", str(target), "str")
        if target_host is not None:
            graph._set_json_attr("target_host", str(target_host), "str")
        if cfg.pass_enabled("OpFusion"):
            graph._set_json_attr("opt_level", 1, "int")
        else:
            graph._set_json_attr("opt_level", 0, "int")
        graph = graph.apply("InferShape").apply("InferType")
        graph = graph.apply("GraphFindFusibleGroups")
        graph = graph.apply("GraphFuse")
        with target:
            graph = graph.apply("GraphCompile")
        libmod = graph_attr._move_out_module(graph, "module")
        # Write variable initial values into params
        if init_var:
            if params is None:
                params = {}
            params.update(init_var)
        return pre_graph, graph, libmod, params
