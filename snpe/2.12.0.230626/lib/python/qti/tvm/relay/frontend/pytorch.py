# ==============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, cell-var-from-loop, unnecessary-lambda
# pylint: disable=missing-function-docstring
"""PT: PyTorch frontend."""
import functools
import itertools
import logging
import math
import re
import sys

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from .. import analysis as _analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import qnn, transform
from ..expr_functor import ExprMutator
from ..loops import while_loop
from ..prelude import Prelude, StaticTensorArrayOps
from ..ty import Any, TensorType, TupleType
from . import qnn_torch
from .common import AttrCvt, get_relay_op
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated
from .common import try_infer_value
from .common import set_span
from .pytorch_utils import is_version_greater_than

__all__ = ["from_pytorch"]

# This returns a "subgraph" which puts variables whenever
# the type is known. It also records things to map the input
# nodes to the extracted graph's nodes.
# As Python objects are not round-trippable through C++, and
# our type annotations only live in Python, we need to map
# the we need to map the nodes we get in visiting to the nodes
# we used to construct the graph (they are the same in C++,
# match each other in dictionary lookups, but are not the same
# in Python) by using the hint dictionary filled as
# {node: node for node in nodes} to get the type annotations.
# https://discuss.tvm.apache.org/t/round-tripping-objects-through-the-ffi/8440
class _TypeFinder(ExprMutator):
    def __init__(self, types):
        super().__init__()
        self.counter = 0
        self.vars = {}
        self.types = types
        self.leave = set()  # some variables are not inputs

    def visit_let(self, let):
        self.leave.add(let.var)
        return super().visit_let(let)

    def visit_function(self, fn):
        self.leave.update(fn.params)
        return super().visit_function(fn)

    def visit(self, expr):
        if expr in self.leave:
            return super().visit(expr)
        if expr in self.vars:
            return self.vars[expr]
        if isinstance(expr, tvm.relay.Var):
            self.vars[expr] = expr
            return expr
        if expr in self.types:
            ty = self.types[expr]
            v = tvm.relay.var(f"_{self.counter}", type_annotation=ty)
            self.counter += 1
            self.vars[expr] = v
            return v
        v = super().visit(expr)
        return v


def _should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    def inplace_add_to_add(op_name):
        if op_name == "aten::add_":
            return "aten::add"
        else:
            return op_name

    uses = _get_uses(list_construct_node)

    for loop_use in filter(lambda use: use.user.kind() == "prim::Loop", uses):
        block_input_index = loop_use.offset - 1
        block = list(loop_use.user.blocks())[0]
        list_loop_var = list(block.inputs())[block_input_index]
        uses += _get_uses(list_loop_var.node())

    op_names = map(inplace_add_to_add, set(use.user.kind() for use in uses))

    list_ops = set(["aten::add", "aten::__getitem__"])
    intersect = list_ops.intersection(op_names)

    if len(intersect) > 0 and intersect != set(["aten::add"]):
        return True

    # if add op outputs list, it is dynamic so we need to construct List ADT
    for use in filter(lambda use: use.user.kind() in ["aten::add", "aten::add_"], uses):
        output_type = _get_node_type(use.user)
        if output_type == "ListType":
            return True

    return False


def _is_int_seq(seq):
    # TODO (t-vi): handle non-int constants? (like numpy.intXX)
    return len(seq) > 0 and all([isinstance(i, int) for i in seq])


# operator implementation
class PyTorchOpConverter:
    """A helper class for holding PyTorch op converters."""

    def __init__(self, prelude, default_dtype, use_parser_friendly_name=False):
        self.prelude = prelude
        self.default_dtype = default_dtype
        self.create_convert_map()
        self.types = {}  # map from nodes to (Relay) type annotations
        ### [QTI Change]
        self.source_map = {}  # map from graph node to its source name
        self.op_type_dict = {}  # map from op type to its presenting order
        self.current_op = []  # stack for recording current processing op
        self.use_parser_friendly_name = use_parser_friendly_name
        self.output_name_to_node_name = {}
        self.torch_source_map = {}
        ### [End QTI Change]


    # this incrementally infers the type, see the comments on the type visitor
    # above.
    def infer_type(self, node, mod=None):
        """An incremental method to infer the type of a node in the relay graph."""

        if node in self.types:
            return self.types[node]
        if isinstance(node, tvm.relay.Var):
            return node.type_annotation

        tf = _TypeFinder(types=self.types)
        new_node = tf.visit(node)
        fn = _function.Function(list(tf.vars.values()), new_node)
        new_mod = IRModule({"main": fn})
        if mod is not None:
            new_mod.update(mod)
        new_mod = transform.RemoveUnusedFunctions()(new_mod)
        new_mod = transform.InferType()(new_mod)
        entry = new_mod["main"]
        ty = entry.body.checked_type
        self.types[node] = ty
        return self.types[node]

    def infer_type_with_prelude(self, val):
        body = self.infer_type(val, self.prelude.mod)
        return body

    # list ADT utilities
    def convert_to_list_adt(self, py_lst):
        elem_tys = [self.infer_type_with_prelude(elem) for elem in py_lst]
        msg = "List elements should have identical types"
        assert all(map(lambda ty: ty == elem_tys[0], elem_tys)), msg

        # get_type returns type_name, ctor1, ..., ctorN
        # 1 is nil
        _, cons, nil = self.prelude.mod.get_type("List")
        adt_lst = nil()
        for elem in reversed(py_lst):
            adt_lst = cons(elem, adt_lst)
        return adt_lst

    def map_tensor_array_constructor(self, adt_lst, shape):
        static_tensor_array_ops = StaticTensorArrayOps(self.prelude, "float32", shape)
        static_tensor_array_ops.register()
        tensor_create = self.prelude.get_tensor_ctor_static("tensor_constructor", "float32", shape)
        return self.prelude.map(tensor_create, adt_lst)

    def convert_to_tensor_array(self, adt_lst):
        _, cons, nil = self.prelude.mod.get_type("List")
        if self.prelude.length(adt_lst) == 0:
            return nil()

        checked_type = self.infer_type_with_prelude(self.prelude.hd(adt_lst))
        shape = checked_type.shape
        tensor_array = self.map_tensor_array_constructor(adt_lst, shape)
        return tensor_array, tuple(shape)

    def infer_shape(self, inputs, mod=None):
        """A method to get the output type of an intermediate node in the graph."""
        typ = self.infer_type(inputs, mod=mod)
        if hasattr(typ, "shape"):
            # Regular operator that outputs tensors
            return get_const_tuple(typ.shape)
        # The return type is not a tensor, for example List
        return typ

    def infer_shape_with_prelude(self, inputs):
        return self.infer_shape(inputs, mod=self.prelude.mod)

    def record_output_type(self, output):
        if isinstance(output, tuple):
            cleaned_output = [o for o in output if o is not None]
            types = self.infer_type_with_prelude(_expr.Tuple(cleaned_output))
            for o, t in zip(cleaned_output, types.fields):
                self.types[o] = t
        elif isinstance(output, _expr.Expr):
            self.infer_type_with_prelude(output)
        # it can also happen that the type is int or so

    def pytorch_promote_types(self, inputs, dtypes):
        """This promotes TVM inputs with TVM dtypes passed like PyTorch would"""
        actual_dtypes = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, _expr.Expr):
                idt = self.infer_type(inp).dtype
                actual_dtypes.append(idt)
            else:
                actual_dtypes.append(dtypes[i])
        dtypes = actual_dtypes
        tensor_dtypes = [dt for inp, dt in zip(inputs, dtypes) if not np.isscalar(inp)]
        non_tensor_inputs = [inp for inp in inputs if np.isscalar(inp)]
        result_type = _pytorch_result_type(tensor_dtypes, non_tensor_inputs)
        results = []
        for inp, dt in zip(inputs, dtypes):
            if np.isscalar(inp):
                results.append(_expr.const(inp, dtype=result_type))
            elif dt == result_type:
                results.append(inp)
            else:
                results.append(_op.cast(inp, result_type))
        return results

    def is_quantized_tensor(self, data):
        # If a quantized Torch module is saved and loaded back, dtype will be dropped
        # Since dtypes from Torch tensors are not reliable in such cases, we use
        # Relay's type inference result to decide if an input tensor is quantized
        ty = self.infer_type_with_prelude(data)
        return ty.dtype == "uint8"

    # Operator implementations
    def make_elemwise(self, name):
        def elemwise(inputs, input_types):
            data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])
            return get_relay_op(name)(data0, data1)

        return elemwise

    def min_max_common(self, name_elemwise, name_reduce, inputs, input_types):
        if len(inputs) == 1:
            data = self.pytorch_promote_types(inputs[:1], input_types[:1])
            return get_relay_op(name_reduce)(data[0])
        elif len(inputs) >= 2 and isinstance(inputs[1], int):
            data = self.pytorch_promote_types(inputs[:1], input_types[:1])
            dim = inputs[1]
            keepdims = inputs[2] if len(inputs) > 2 else False
            # also return dummy indices
            return get_relay_op(name_reduce)(data[0], axis=dim, keepdims=keepdims), None
        else:
            data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])
            return get_relay_op(name_elemwise)(data0, data1)

    def max(self, inputs, input_types):
        return self.min_max_common("maximum", "max", inputs, input_types)

    def min(self, inputs, input_types):
        return self.min_max_common("minimum", "min", inputs, input_types)

    def make_unary(self, name):
        def unary(inputs, input_types):
            # this is just to ensure tensor input
            (data,) = self.pytorch_promote_types(inputs[:1], input_types[:1])
            return get_relay_op(name)(data)

        return unary

    def log1p(self, inputs, input_types):
        # 1_plus_log x = log(x + 1)
        (dtype,) = input_types
        one = _expr.const(1, dtype=dtype)
        return _op.log(inputs[0] + one)

    def square(self, inputs, input_types):
        (dtype,) = input_types
        return _op.power(inputs[0], _expr.const(2, dtype))

    def arange(self, inputs, input_types):
        def _get_value(val, dtype):
            # dtype is a tvm dtype
            if isinstance(val, _expr.Expr):
                ### [QTI Change]
                # since "arange" op will fill expr into its attribute
                # invoke set_span here to prevent expr-rewritten occurrs in span-filling stage
                op_node = self.current_op[-1]
                torch_node_name = self._get_node_name(op_node)
                source_name = torch_node_name if torch_node_name else self.source_map[self.current_op[-1]]
                inp = set_span(_op.cast(val, dtype), source_name, op_type=op_node.kind())
                ### [End QTI Change]
                ret, _ = try_infer_value(inp, lambda ret: _expr.const(ret, dtype))
            else:
                ret = _create_typed_const(val, dtype)
            return ret

        def _get_type(val, inp_type):
            if isinstance(val, _expr.Expr):
                dtype = str(self.infer_type(val))
                return dtype
            return inp_type

        # PyTorch arange uses the following type semantics:
        # - if a dtype is given, start, stop, step are converted to that dtype
        # - if no dtype is given and all args are integral, dtype is int64
        # - if no dtype is given and there is a float arg, dtype is float32
        if len(inputs) == 5:
            dtype0 = _get_type(inputs[0], input_types[0])
            if inputs[1] is not None:
                dtype = _convert_dtype_value(inputs[1])
            elif dtype0.startswith("float"):
                dtype = "float32"
            else:
                dtype = "int64"
            start = _expr.const(0, dtype)
            stop = _get_value(inputs[0], dtype)
            step = _expr.const(1, dtype)
        elif len(inputs) == 7:
            types = [_get_type(inputs[i], input_types[i]) for i in range(3)]
            if inputs[3] is not None:
                dtype = _convert_dtype_value(inputs[3])
            elif any([t.startswith("float") for t in types]):
                dtype = "float32"
            else:
                dtype = "int64"
            start = _get_value(inputs[0], dtype)
            stop = _get_value(inputs[1], dtype)
            step = _get_value(inputs[2], dtype)
        else:
            msg = "Unknown number of arguments (%d) to parse." % (len(inputs))
            raise AssertionError(msg)

        return _op.transform.arange(start=start, stop=stop, step=step, dtype=dtype)

    def squeeze(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 1:
            axis = None
        else:
            # TODO (t-vi): why is the cast to int needed? similarly elsewhere
            axis = [int(inputs[1])]

        return _op.transform.squeeze(data, axis)

    def unsqueeze(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        return _op.transform.expand_dims(data, int(axis), 1)

    def concatenate(self, inputs, input_types):
        def tensor_array_concat(lst, axis):
            assert axis == 0, "Tensor array concat supported only for axis 0"
            tensor_array, shape = self.convert_to_tensor_array(lst)
            concat_shape = (Any(),) + shape[1:]
            concat = self.prelude.get_global_var_static("tensor_array_concat", "float32", shape)
            concatenated = concat(tensor_array)

            static_tensor_array_ops = StaticTensorArrayOps(self.prelude, "float32", concat_shape)
            static_tensor_array_ops.register()
            get_tensor = self.prelude.get_global_var_static(
                "tensor_get_data", "float32", concat_shape
            )
            return get_tensor(concatenated)

        data = inputs[0]
        axis = inputs[1]

        if not isinstance(data, list):
            return tensor_array_concat(data, axis)

        if isinstance(data, _expr.Expr):
            data = [data]

        return _op.tensor.concatenate(data, int(axis))

    def slice(self, inputs, input_types):
        axis_dtype = "int64"
        index_size_limit = sys.maxsize
        data = inputs[0]
        dshape = self.infer_shape(data)
        ndim = len(dshape)
        dim = int(inputs[1])
        stride = inputs[4]

        target_begin, is_begin_const = try_infer_value(
            ### [QTI Change] ###
            inputs[2], lambda ret: ret.astype(np.int).item(0)
            ### [End QTI Change] ###
        )
        target_end, is_end_const = try_infer_value(
            ### [QTI Change] ###
            inputs[3], lambda ret: ret.astype(np.int).item(0)
            ### [End QTI Change] ###
        )

        # A fast path when slicing is nop.
        if (
            isinstance(target_begin, int)
            and isinstance(target_end, int)
            and target_begin == 0
            and target_end >= index_size_limit
            and stride == 1
        ):
            return data

        # Process begin
        begin = [0] * ndim
        begin[dim] = target_begin

        if not isinstance(begin[dim], int):
            tmp = []
            for b in begin:
                if isinstance(b, int):
                    tmp.append(_op.expand_dims(_expr.const(b, axis_dtype), axis=0))
                else:
                    tmp.append(_op.cast(_op.expand_dims(b, axis=0), axis_dtype))
            begin = _op.concatenate(tmp, axis=0)
            btype = self.infer_type(begin).dtype
            if str(btype) != axis_dtype:
                begin = _op.cast(begin, axis_dtype)

        # Process end
        if isinstance(target_end, int) and target_end >= index_size_limit:
            target_end = dshape[dim]

        if any([isinstance(d, tvm.tir.Any) for d in dshape]):
            end = _op.shape_of(data)
        else:
            end = dshape

        if isinstance(target_end, int):
            if isinstance(end, list):
                end[dim] = target_end
            else:
                all_static = True
                for i, shape_dim in enumerate(dshape):
                    if i != dim and isinstance(shape_dim, tvm.tir.Any):
                        all_static = False

                if all_static:
                    end = list(get_const_tuple(dshape))
                    end[dim] = target_end
                else:
                    target_end = _expr.const(target_end)
                    end = _op.scatter(
                        end,
                        _op.expand_dims(_expr.const(dim), axis=0),
                        _op.expand_dims(target_end, axis=0),
                        axis=0,
                    )
        else:
            end = _op.cast(_op.shape_of(data), axis_dtype)
            if not isinstance(target_end, tvm.tir.Any):
                ttype = self.infer_type(target_end).dtype
                if str(ttype) != axis_dtype:
                    target_end = _op.cast(target_end, axis_dtype)
                end = _op.scatter(
                    end,
                    _op.expand_dims(_expr.const(dim), axis=0),
                    _op.expand_dims(target_end, axis=0),
                    axis=0,
                )

        if not isinstance(end, list):
            etype = self.infer_type(end).dtype
            if str(etype) != axis_dtype:
                end = _op.cast(end, axis_dtype)

        strides = [1] * ndim
        strides[dim] = stride

        return _op.transform.strided_slice(
            data, begin=begin, end=end, strides=strides, slice_mode="end"
        )

    def narrow(self, inputs, input_types):
        # Inputs are:
        # 0 - the tensor to narrow
        # 1 - the dimension along which to narrow
        # 2 - the starting dimension
        # 3 - the distance to the ending dimension
        # Lets find the ending dimension
        end = self.add(inputs[2:4], input_types[2:4])
        stride = 1
        slice_input = inputs[:3] + [end, stride]
        slice_types = input_types + ["int32"]
        return self.slice(slice_input, slice_types)

    def split(self, inputs, input_types):
        data = inputs[0]
        split_size = int(inputs[1])
        dim = int(inputs[2])

        split_index = split_size
        indices = []
        while split_index < self.infer_shape(data)[dim]:
            indices.append(split_index)
            split_index += split_size

        return _op.split(data, indices, dim)

    def split_with_sizes(self, inputs, input_types):
        data = inputs[0]
        sections = inputs[1]
        dim = int(inputs[2])

        if len(sections) == 1:
            # a special case used in torchvision detection models
            return _expr.TupleWrapper(_expr.Tuple([data]), 1)

        split_index = 0
        indices = []
        for i in range(len(sections) - 1):
            index, _ = try_infer_value(sections[i], lambda ret: int(ret))
            split_index += index
            indices.append(split_index)

        return _op.split(data, indices, dim)

    def tensor_split(self, inputs, input_types):
        # Reference: https://pytorch.org/docs/stable/generated/torch.tensor_split.html
        import torch

        if not isinstance(inputs[1], (int, list, tuple, torch.Tensor)):
            msg = "indices_or_sections type %s could not be parsed in tensor_split op" % (
                type(inputs[1])
            )
            raise AssertionError(msg)

        if isinstance(inputs[1], torch.Tensor) and not (
            list(inputs[1].shape) == [] or list(inputs[1].shape) == 1
        ):
            msg = "indices_or_sections must be a zero-dimensional or one-dimensional long tensor"
            raise AssertionError(msg)

        if isinstance(inputs[1], int) or (
            isinstance(inputs[1], torch.Tensor) and list(inputs[1].shape) == []
        ):
            data = inputs[0]
            n = int(inputs[1])
            dim = int(inputs[2])

            split_size = int(self.infer_shape(data)[dim] / n)
            split_rest = int(self.infer_shape(data)[dim] % n)

            indices = []
            split_index = split_size
            if split_rest == 0:
                for i in range(n - 1):
                    indices.append(split_index)
                    split_index += split_size
            else:
                for i in range(split_rest):
                    indices.append(split_index + 1)
                    split_index = (i + 1) * (split_index + 1)
                for i in range(n - split_rest - 1):
                    split_index += split_size
                    indices.append(split_index)

            return _op.split(data, indices, dim)
        else:
            data = inputs[0]
            sections = inputs[1]
            dim = int(inputs[2])

            if isinstance(sections, tuple):
                sections = list(sections)
            elif isinstance(sections, torch.Tensor):
                sections = sections.cpu().numpy().tolist()

            return _op.split(data, sections, dim)

    def select(self, inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        index = _wrap_const(inputs[2])
        return _op.transform.take(data, index, axis=dim, mode="wrap")

    def take(self, inputs, input_types):
        data = inputs[0]
        indices = _op.cast(inputs[1], "int32")

        return _op.transform.take(data, indices=indices, mode="wrap")

    def topk(self, inputs, input_types):
        data = inputs[0]
        axis = int(inputs[2])
        is_ascend = not bool(inputs[3])
        sort = bool(inputs[4])

        if isinstance(inputs[1], _expr.Expr):
            k, _ = try_infer_value(inputs[1], lambda ret: ret.tolist())
        else:
            k = inputs[1]

        if not sort:
            msg = "Currently supports only sorted output for topk operator."
            raise AssertionError(msg)

        outs = _op.topk(data, k=k, axis=axis, is_ascend=is_ascend, ret_type="both", dtype="int64")

        return outs[0], outs[1]

    def reciprocal(self, inputs, input_types):
        data = inputs[0]
        return _expr.const(1.0, dtype=input_types[0]) / data

    def repeat(self, inputs, input_types):
        data = inputs[0]
        reps = []
        for r in inputs[1]:
            if isinstance(r, int):
                reps.append(r)
            else:
                reps.append(int(_infer_value(r, {}).numpy()))

        return _op.transform.tile(data, reps=reps)

    def repeat_interleave(self, inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], int):
            repeats = inputs[1]
            axis = inputs[2]
        else:
            msg = "Only repeat with one value as repeat is currently supported."
            raise AssertionError(msg)
        if axis is None:  # Flatten the data if no axis is given from torch
            data = _op.transform.reshape(data, [-1])
            axis = 0
        return _op.transform.repeat(data, repeats=repeats, axis=axis)

    def addcdiv(self, inputs, input_types):
        data, t1, t2, c = self.pytorch_promote_types(inputs[:4], input_types[:4])
        return data + (c * (t1 / t2))

    def addcmul(self, inputs, input_types):
        data, t1, t2, c = self.pytorch_promote_types(inputs[:4], input_types[:4])
        return data + (c * (t1 * t2))

    def where(self, inputs, input_types):
        if len(inputs) == 1:
            return self.nonzero([inputs[0], True], input_types)

        cond = inputs[0]
        x, y = self.pytorch_promote_types(inputs[1:3], input_types[1:3])
        return _op.where(cond, x, y)

    def full_impl(self, data, fill_value, dtype):
        size = []
        need_reshape = False
        new_shape = []
        for dim in data:
            if isinstance(dim, _expr.Expr):
                if isinstance(dim, _expr.Constant):
                    dim = int(dim.data.numpy())
                    if isinstance(size, list):
                        size.append(dim)
                    new_shape.append(dim)
                else:
                    dim, success = try_infer_value(dim, lambda ret: int(ret), lambda: 0)
                    new_shape.append(dim)

                    if success:
                        if isinstance(size, list):
                            size.append(dim)
                    else:
                        size = None
                        need_reshape = True
            else:
                if isinstance(size, list):
                    size.append(dim)
                new_shape.append(dim)

        if size is None:
            tmp = []
            for dim in data:
                tmp.append(_op.cast(_op.expand_dims(dim, axis=0), "int64"))
            size = _op.concatenate(tmp, axis=0)

        out = _op.full(_expr.const(fill_value), size, dtype=dtype)
        if need_reshape:
            out = _op.reshape(out, new_shape)
        return out

    def ones(self, inputs, input_types):
        data = inputs[0]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in ones op" % (type(data))
            raise AssertionError(msg)

        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        return self.full_impl(data, 1, dtype)

    def ones_like(self, inputs, input_types):
        data = inputs[0]
        out = _op.ones_like(data)

        # If the input and the output datatype is different, do a cast
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        if input_types[0] != dtype:
            out = _op.cast(out, dtype)

        return out

    def zeros(self, inputs, input_types):
        data = inputs[0]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in zeros op" % (type(data))
            raise AssertionError(msg)

        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        return self.full_impl(data, 0, dtype)

    def zeros_like(self, inputs, input_types):
        data = inputs[0]
        out = _op.zeros_like(data)

        # If the input and the output datatype is different, do a cast
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out

    def full(self, inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in full op" % (type(data))
            raise AssertionError(msg)

        if inputs[2] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[2])
        else:
            # if dtype is None, torch uses a global default set by torch.set_default_tensor_type()
            dtype = self.default_dtype

        return self.full_impl(data, fill_value, dtype)

    def full_like(self, inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        out = _op.full_like(data, _expr.const(fill_value))

        # If the input and the output datatype is different, do a cast
        if inputs[2] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[2])
        else:
            # if dtype is None, torch uses a global default set by torch.set_default_tensor_type()
            dtype = self.default_dtype
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out

    def linspace(self, inputs, input_types):
        start = inputs[0]
        stop = inputs[1]
        step = inputs[2]

        # Find the spacing between values as step
        if step != 1:
            step = (stop - start) / (step - 1)
            stop = stop + step
        else:
            stop = start + step

        dtype = "float32" if inputs[3] is not None else _convert_dtype_value(inputs[3])
        start = _create_typed_const(start, dtype)
        stop = _create_typed_const(stop, dtype)
        step = _create_typed_const(step, dtype)

        return _op.transform.arange(start=start, stop=stop, step=step, dtype=dtype)

    def relu(self, inputs, input_types):
        data = inputs[0]
        if self.is_quantized_tensor(data):
            assert len(inputs) == 3, "Input quant param not found in op inputs"
            input_zero_point = _expr.const(inputs[2], dtype="int32")
            return qnn_torch.quantized_relu(data, input_zero_point)
        return _op.nn.relu(data)

    def prelu(self, inputs, input_types):
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html#torch.nn.PReLU
        data = inputs[0]
        dim = self.get_dims(data)
        ndims = len(dim)
        axis = 0 if ndims == 1 else 1
        alpha = _op.broadcast_to(inputs[1], (dim[axis]))
        return _op.nn.prelu(data, alpha, axis)

    def leaky_relu(self, inputs, input_types):
        data = inputs[0]
        alpha = float(inputs[1])
        return _op.nn.leaky_relu(data, alpha)

    def elu(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        alpha = _expr.const(float(inputs[1]), dtype=dtype)
        return alpha * _op.nn.relu(_expr.const(1, dtype=dtype) - _op.exp(data)) + _op.nn.relu(data)

    def celu(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        alpha = _expr.const(float(inputs[1]), dtype=dtype)
        return alpha * _op.nn.relu(
            _expr.const(1, dtype=dtype) - _op.exp(data / alpha)
        ) + _op.nn.relu(data)

    def gelu(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        # gelu is data  * normcdf(data)
        # normcdf expressed as erf because we don't currently have that intrinsic
        # note that there is also a fastgelu variant approximating normcdf
        # with tanh and third order polynomials, but this is "true" gelu
        return data * (
            _expr.const(0.5, dtype=dtype)
            + _op.erf(data * _expr.const(0.5 ** 0.5, dtype=dtype)) * _expr.const(0.5, dtype=dtype)
        )

    def selu(self, inputs, input_types):
        data = inputs[0]
        # https://pytorch.org/docs/stable/nn.html#selu
        dtype = input_types[0]
        alpha = _expr.const(-1.6732632423543772848170429916717, dtype=dtype)
        gamma = _expr.const(1.0507009873554804934193349852946, dtype=dtype)
        return gamma * (
            alpha * _op.nn.relu(_expr.const(1.0, dtype=dtype) - _op.exp(data)) + _op.nn.relu(data)
        )

    def silu(self, inputs, input_types):
        data = inputs[0]
        return data * _op.tensor.sigmoid(data)

    def glu(self, inputs, input_types):
        """
        Applies the gated linear unit function GLU(a,b)= a * sigmoid(b)
        where a is the first half of the input matrices and b is the second half.
        Link: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
        """
        data = inputs[0]
        dim = inputs[1]
        relay_tup = _op.transform.split(data, 2, dim)
        return relay_tup[0] * _op.tensor.sigmoid(relay_tup[1])

    def log_sigmoid(self, inputs, input_types):
        data = inputs[0]
        return _op.log(_op.tensor.sigmoid(data))

    def hard_sigmoid(self, inputs, input_types):
        def _relu6(x):
            return _op.tensor.clip(x, 0.0, 6.0)

        def func(x):
            return _relu6(x + _expr.const(3.0)) / _expr.const(6.0)

        if self.is_quantized_tensor(inputs[0]):
            input_scale = _expr.const(inputs[1])
            input_zero_point = _expr.const(inputs[2])
            # PyTorch seems to use the following output qparams, but accuracy
            # is broken if we use this.
            # TODO(masahi): Revisit this parameter choice
            #
            # Taken from src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp
            # output_scale = _expr.const(0.00390625)  # 1.0 / 2^8
            # output_zero_point = _expr.const(-128)
            output_scale = input_scale
            output_zero_point = input_zero_point

            data = qnn.op.dequantize(inputs[0], input_scale, input_zero_point, axis=1)
            out = func(data)
            return qnn.op.quantize(out, output_scale, output_zero_point, out_dtype="uint8")

        return func(inputs[0])

    def hard_swish(self, inputs, input_types):
        data = inputs[0]
        return data * self.hard_sigmoid(inputs, input_types)

    def adaptive_avg_pool(self, op, inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]

        def func(x):
            return op(x, output_size=output_size)

        if self.is_quantized_tensor(data):
            return qnn_torch.apply_with_upcast(data, func)

        return func(data)

    def adaptive_max_pool(self, op, inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]
        # returns dummy indices too
        return op(data, output_size=output_size), None

    @staticmethod
    def convert_const_list(data):
        if isinstance(data, list):
            for i, _ in enumerate(data):
                if isinstance(data[i], _expr.Expr):
                    data[i] = int(_infer_value_simulated(data[i], {}).numpy())
        return data

    def maxpool_2d(self, inputs, input_types):
        data = inputs[0]

        pool_size = self.convert_const_list(inputs[1])
        strides = self.convert_const_list(inputs[2] if inputs[2] else pool_size)
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        return _op.nn.max_pool2d(
            data,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            layout="NCHW",
            ceil_mode=ceil_mode,
        )

    def maxpool_2d_with_indices(self, inputs, input_types):
        # returns dummy indices too
        return self.maxpool_2d(inputs, input_types), None

    def maxpool_1d(self, inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        return _op.nn.max_pool1d(
            data,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            layout="NCW",
            ceil_mode=ceil_mode,
        )

    def maxpool_3d(self, inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        return _op.nn.max_pool3d(
            data,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
        )

    def hardtanh(self, inputs, input_types):
        a = inputs[0]
        tanh_min = float(inputs[1])
        tanh_max = float(inputs[2])
        return _op.tensor.clip(a, tanh_min, tanh_max)

    def convolution(self, inputs, input_types):
        # Use transpose or normal
        use_transpose = True if inputs[6] == 1 else False

        data = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        strides = tuple(inputs[3])
        padding = tuple(inputs[4])
        dilation = tuple(inputs[5])

        if isinstance(weight, _expr.Expr):
            inferred_shape = self.infer_shape(weight)
            weight_shape = []
            for infer in inferred_shape:
                weight_shape.append(infer)
        else:
            msg = "Data type %s could not be parsed in conv op" % (type(weight))
            raise AssertionError(msg)

        # Transposed convolutions have IOHW layout.
        if use_transpose:
            weight_shape[0], weight_shape[1] = weight_shape[1], weight_shape[0]

        channels = weight_shape[0]
        groups = int(inputs[8])

        # Check if this is depth wise convolution
        # We need to reshape weight so that Relay could recognize this is depth wise
        # weight_shape[1] is always in_channels // groups
        # For depthwise, in_channels == groups, so weight_shape[1] == 1
        # If groups > 1 but weight_shape[1] != 1, this is group convolution
        if groups > 1 and weight_shape[1] == 1:
            channel_multiplier = channels // groups
            new_weight_shape = (groups, channel_multiplier) + tuple(weight_shape[2:])
            weight = _op.transform.reshape(weight, new_weight_shape)

        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        # We are trying to invoke various relay operations through a single conv_op variable.
        # However the function signatures for some operations have additional attributes so we
        # pass these in along with the standard ones.
        additional_arguments = dict()

        if use_transpose:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d_transpose
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d_transpose
            else:
                conv_op = _op.nn.conv1d_transpose
            output_padding = tuple(inputs[7])
            additional_arguments["output_padding"] = output_padding

        else:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d
            else:
                conv_op = _op.nn.conv1d

        if len(kernel_size) == 3:
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"
        elif len(kernel_size) == 2:
            data_layout = "NCHW"
            kernel_layout = "OIHW"
        else:
            data_layout = "NCW"
            kernel_layout = "OIW"

        # Conv1d does not currently support grouped convolution so we convert it to conv2d
        is_grouped_conv1d = False
        if groups > 1 and len(kernel_size) == 1 and not use_transpose:
            is_grouped_conv1d = True
            conv_op = _op.nn.conv2d
            kernel_size = [1] + kernel_size
            strides = (1,) + strides
            padding = (0,) + padding
            dilation = (1,) + dilation
            data = _op.expand_dims(data, axis=2)
            weight = _op.expand_dims(weight, axis=2)
            data_layout = "NCHW"
            kernel_layout = "OIHW"

        conv_out = conv_op(
            data,
            weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channels=channels,
            kernel_size=kernel_size,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_layout="",
            out_dtype="",
            **additional_arguments,
        )
        if use_bias:
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if is_grouped_conv1d:
            # Because we conducted grouped conv1d convolution through conv2d we must
            # squeeze the output to get the correct result.
            res = _op.squeeze(res, axis=[2])
        return res

    def softmax(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        if isinstance(axis, str):
            axis = int(axis)

        return _op.nn.softmax(data, axis=axis)

    def threshold(self, inputs, input_types):
        data = inputs[0]
        return _op.nn.relu(data)

    def contiguous(self, inputs, input_types):
        return inputs[0]

    def batch_norm(self, inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]

        channels = self.infer_shape(data)

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        moving_mean = inputs[3]
        moving_var = inputs[4]
        epsilon = float(inputs[7])

        return _op.nn.batch_norm(
            data,
            gamma,
            beta,
            moving_mean,
            moving_var,
            axis=1,
            epsilon=epsilon,
            center=center,
            scale=scale,
        )[0]

    def instance_norm(self, inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]
        channels = self.infer_shape(data)

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        epsilon = float(inputs[7])
        return _op.nn.instance_norm(
            data, gamma, beta, axis=1, epsilon=epsilon, center=center, scale=scale
        )

    def get_dims(self, data):
        import torch

        if isinstance(data, _expr.Expr):
            dims = self.infer_shape(data)
        elif isinstance(data, list):
            dims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            dims = data.shape
        else:
            msg = "Data type %s could not be parsed" % type(data)
            raise AssertionError(msg)
        return dims

    def layer_norm(self, inputs, input_types):
        data = inputs[0]
        ndims = len(self.get_dims(inputs[1]))
        assert ndims == 1, "Support only normalization over last one dimension."

        return _op.nn.layer_norm(
            data,
            gamma=inputs[2],
            beta=inputs[3],
            axis=-1,
            epsilon=float(inputs[4]),
            center=True,
            scale=True,
        )

    def group_norm(self, inputs, input_types):
        data = inputs[0]
        gamma = inputs[2]
        beta = inputs[3]
        num_groups = inputs[1]
        epsilon = float(inputs[4])

        return _op.nn.group_norm(
            data,
            gamma=gamma,
            beta=beta,
            num_groups=num_groups,
            axis=1,
            epsilon=epsilon,
            center=True,
            scale=True,
        )

    def transpose(self, inputs, input_types):
        data = inputs[0]

        import torch

        if isinstance(data, _expr.Expr):
            ndims = len(self.infer_shape_with_prelude(data))
        elif isinstance(data, list):
            ndims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            ndims = data.shape
        else:
            msg = "Data type %s could not be parsed in transpose op" % (type(data))
            raise AssertionError(msg)

        if isinstance(data, tvm.runtime.NDArray):
            ndims = len(data.shape)
        axes = list(range(ndims))

        num_inputs = len(inputs)

        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
            if not isinstance(data, _expr.Expr):
                data = _expr.const(data)

        elif num_inputs == 3:
            parse = lambda i: ndims * (i < 0) + i
            src, dst = [parse(int(inputs[i])) for i in [1, 2]]
            axes[src] = dst
            axes[dst] = src
        else:
            axes = inputs[1]
        return _op.transform.transpose(data, axes)

    def flatten(self, inputs, input_types):
        data = inputs[0]
        start = int(inputs[1])
        end = int(inputs[2])
        dshape = get_const_tuple(self.infer_shape_with_prelude(data))
        ndim = len(dshape)
        if end < 0:
            end += ndim
        new_shape = [0] * start

        new_shape.append(-1)
        squeeze_axes = []
        for i in range(start + 1, end + 1):
            new_shape.append(1)
            squeeze_axes.append(i)
        for _ in range(end + 1, ndim):
            new_shape.append(0)
        out = _op.reshape(data, new_shape)
        if squeeze_axes:
            out = _op.squeeze(out, axis=squeeze_axes)
        return out

    def addmm(self, inputs, input_types):
        input_mat = inputs[0]
        mat1 = inputs[1]
        data_type = input_types[1]
        mat2 = inputs[2]

        beta = inputs[3]
        alpha = inputs[4]

        if not isinstance(alpha, _expr.Expr) and alpha != 1:
            alpha = _create_typed_const(alpha, data_type)
            mat1 *= alpha

        if not isinstance(beta, _expr.Expr) and beta != 1:
            beta = _create_typed_const(beta, data_type)
            mat2 *= beta

        transposed_mat2 = _op.transform.transpose(mat2, axes=[1, 0])

        units = self.infer_shape(transposed_mat2)[0]
        dense_out = _op.nn.dense(mat1, transposed_mat2, units=units)

        return dense_out + input_mat

    def size(self, inputs, input_types):
        shape = self.infer_shape_with_prelude(inputs[0])
        axis = None
        if len(inputs) > 1:
            axis = int(inputs[1])

        if any(map(lambda s: isinstance(s, tvm.tir.expr.Any), shape)):
            if axis is None or isinstance(shape[axis], tvm.tir.expr.Any):
                shape_dynamic = _op.shape_of(inputs[0], dtype="int32")
                if axis is not None:
                    return _op.take(shape_dynamic, _expr.const(axis), 0)
                return shape_dynamic

        if axis is not None:
            return _expr.const(shape[axis])
        return _expr.const(shape)

    def numtotensor(self, inputs, input_types):
        val = inputs[0]
        dtype = input_types[0]

        if isinstance(val, _expr.Expr):
            return val

        if isinstance(val, tvm.tir.IntImm):
            val = val.__int__()
            dtype = int

        arr = val * np.ones([]).astype(dtype)
        return arr

    def tensortonum(self, inputs, input_types):
        return inputs[0]

    def view(self, inputs, input_types):
        data = inputs[0]

        if len(inputs) == 3:
            shape_inp = [inputs[1], self.infer_shape(inputs[2])[0]]
        else:
            if isinstance(inputs[1], list):
                shape_inp = inputs[1]
            else:
                shape_inp = self.infer_shape(inputs[1])
        new_shape = shape_inp
        for i, shape in enumerate(shape_inp):
            if isinstance(shape, _expr.Expr):
                val = _infer_value_simulated(shape, {})
                ### [QTI Change] ###
                new_shape[i] = val.numpy().item(0)
                ### [End QTI Change] ###

        return _op.transform.reshape(data, new_shape)

    def reshape(self, inputs, input_types):
        data = inputs[0]
        new_shape = inputs[1]

        tmp_shape = []
        is_dyn = False
        for s in new_shape:
            if isinstance(s, _expr.Constant):
                tmp_shape.append(int(s.data.numpy()))
            elif isinstance(s, _expr.Expr):
                dim, success = try_infer_value(s, lambda ret: int(ret))
                tmp_shape.append(dim)

                if not success:
                    is_dyn = True
            else:
                tmp_shape.append(s)

        if is_dyn:
            new_shape = []
            for i, s in enumerate(tmp_shape):
                if not isinstance(s, _expr.Expr):
                    s = _expr.const(s, "int64")
                else:
                    s = _op.cast(s, "int64")
                new_shape.append(_op.expand_dims(s, axis=0))
            new_shape = _op.concatenate(new_shape, axis=0)
        else:
            new_shape = tmp_shape
        return _op.transform.reshape(data, new_shape)

    def pixel_shuffle(self, inputs, input_types):
        data = inputs[0]
        upscale_factor = inputs[1]
        upscale_squared = upscale_factor * upscale_factor
        b, c, h, w = self.infer_shape(data)
        assert (
            c % upscale_squared == 0
        ), "input channel should be divisible by square of upscale_factor"

        ndims = len(self.infer_shape_with_prelude(data))
        axes = list(range(ndims))
        num_inputs = len(inputs)
        oc = c // upscale_squared
        oh = h * upscale_factor
        ow = w * upscale_factor

        new_shape = [b, oc, upscale_factor, upscale_factor, h, w]
        out_shape = [b, oc, oh, ow]

        data = _op.transform.reshape(data, new_shape)
        # The data will be transposed to
        # [b, oc, h, upscale_factor, w, upscale_factor]
        # for further reshape
        axes = [0, 1, 4, 2, 5, 3]
        data = _op.transform.transpose(data, axes)
        return _op.transform.reshape(data, out_shape)

    def clone(self, inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)

    def log_softmax(self, inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        return _op.nn.log_softmax(data, axis)

    def sigmoid(self, inputs, input_types):
        data = inputs[0]
        return _op.tensor.sigmoid(data)

    def softplus(self, inputs, input_types):
        # softplus(x) is equivalent to relu(x)+log(1+exp(-abs(x)))
        data = inputs[0]
        dtype = input_types[0]
        beta = _expr.const(float(inputs[1]), dtype=dtype)
        neg_abs_data = _op.abs(data) * _expr.const(-1.0, dtype=dtype)
        return _op.nn.relu(data) + _op.log(_op.exp(neg_abs_data * beta) + _expr.const(1.0, dtype=dtype)) / beta

    def make_avg_pool(self, dim):
        def avg_pool(inputs, input_types):
            data = inputs[0]

            pool_size = self.convert_const_list(inputs[1])
            strides = self.convert_const_list(inputs[2] if inputs[2] else pool_size)
            padding = inputs[3]
            ceil_mode = int(inputs[4])
            count_include_pad = int(inputs[5])

            def func(x):
                if dim == 1:
                    return _op.nn.avg_pool1d(
                        x,
                        pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        dilation=(1,),
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                    )
                elif dim == 2:
                    return _op.nn.avg_pool2d(
                        x,
                        pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        dilation=(1, 1),
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                    )
                elif dim == 3:
                    return _op.nn.avg_pool3d(
                        x,
                        pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        dilation=(1, 1, 1),
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                    )
                else:
                    msg = "Average Pooling dimension should be between 1 and 3"
                    raise RuntimeError(msg)

            if self.is_quantized_tensor(data):
                return qnn_torch.apply_with_upcast(data, func)

            return func(data)

        return avg_pool

    def linear(self, inputs, input_types):
        # https://pytorch.org/docs/stable/nn.functional.html#linear
        # 0 - input
        # 1 - weight
        bias = inputs[2]
        a_shape = self.infer_shape_with_prelude(inputs[0])
        b_shape = self.infer_shape_with_prelude(inputs[1])
        if len(a_shape) == 2 and len(b_shape) == 2:
            mm_out = _op.nn.dense(inputs[0], inputs[1])
        elif len(b_shape) == 1:
            mm_out = self.matmul([inputs[0], inputs[1]], input_types[:2])
        else:
            mm_out = self.matmul(
                [inputs[0], _op.transpose(inputs[1], axes=(1, 0))], input_types[:2]
            )
        if isinstance(bias, _expr.Expr):
            bias_ndims = len(self.infer_shape_with_prelude(bias))
            if bias_ndims == 1:
                return _op.nn.bias_add(mm_out, bias, axis=-1)
            mm_dtype = self.infer_type_with_prelude(mm_out).dtype
            return self.add([mm_out, bias], [mm_dtype, input_types[2]])
        return mm_out

    def dropout(self, inputs, input_types):
        data = inputs[0]
        rate = float(inputs[1])

        return _op.nn.dropout(data, rate)

    def make_reduce(self, name):
        def reduce(inputs, input_types):
            data = inputs[0]
            axis = None
            keepdims = False

            if len(inputs) > 2:  # default, torch have only data, axis=None, keepdims=False
                if isinstance(inputs[1], int):
                    axis = int(inputs[1])
                elif _is_int_seq(inputs[1]):
                    axis = inputs[1]
                else:
                    axis = list(self.infer_shape(inputs[1]))
                keepdims = bool(inputs[2])

            return get_relay_op(name)(data, axis=axis, keepdims=keepdims)

        return reduce

    def norm(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        axis = None
        keepdims = False
        if len(inputs) > 3:
            axis = inputs[2]
            keepdims = bool(inputs[3])

        order = inputs[1]
        if order == np.inf:
            return _op.reduce.max(_op.abs(data), axis=axis, keepdims=keepdims)
        elif order == np.NINF:
            return _op.reduce.min(_op.abs(data), axis=axis, keepdims=keepdims)
        else:
            reci_order = _expr.const(1.0 / order, dtype=dtype)
            order = _expr.const(order)
            return _op.power(
                _op.reduce.sum(_op.power(_op.abs(data), order), axis=axis, keepdims=keepdims),
                reci_order,
            )

    def frobenius_norm(self, inputs, input_types):
        data = inputs[0]
        axis = None
        keepdims = False
        if len(inputs) > 2:
            axis = inputs[1] if len(inputs[1]) > 0 else None
            keepdims = bool(inputs[2])

        return _op.sqrt(_op.reduce.sum((data * data), axis=axis, keepdims=keepdims))

    def std(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        return _op.reduce.std(data, axis=axis, keepdims=keepdims, unbiased=unbiased)

    def variance(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        return _op.reduce.variance(data, axis=axis, keepdims=keepdims, unbiased=unbiased)

    def mean(self, inputs, input_types):
        data = inputs[0]

        if inputs[1]:
            axis = inputs[1]
        else:
            axis = None

        if len(inputs) > 2 and inputs[2]:
            keepdims = int(inputs[2])
        else:
            keepdims = False
        if len(inputs) > 3 and inputs[3]:
            exclude = int(inputs[3])
        else:
            exclude = False

        def func(x):
            return _op.mean(x, axis, keepdims, exclude)

        if self.is_quantized_tensor(data):
            assert len(inputs) == 6, "Input quant param not found in op inputs"
            input_scale = _expr.const(inputs[4])
            input_zero_point = _expr.const(inputs[5])
            return qnn_torch.quantized_mean(data, input_scale, input_zero_point, func)

        return func(data)

    def chunk(self, inputs, input_types):
        data = inputs[0]

        num_chunks = int(inputs[1])
        axis = int(inputs[2])

        if isinstance(data, _expr.Expr):
            inferred_shape = self.infer_shape_with_prelude(data)

        shape = []
        for infer in inferred_shape:
            shape.append(infer)

        dim = int(shape[axis])

        if dim % num_chunks:
            unif_size = int(dim / (num_chunks - 1))
        else:
            unif_size = int(dim / num_chunks)

        chunks = []
        for i in range(0, dim, unif_size):
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = i
            end[axis] = i + unif_size
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin=begin, end=end, strides=stride)
            chunks.append(chunk_out)

        if dim % num_chunks:
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = unif_size * (num_chunks - 1)
            end[axis] = dim
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin=begin, end=end, strides=stride)
            chunks.append(chunk_out)

        return chunks

    def matmul(self, inputs, input_types):

        inputs_0 = inputs[0]
        inputs_1 = inputs[1]

        # Need to check input shape as batch matmul must be supported.
        a_shape = self.infer_shape_with_prelude(inputs_0)
        b_shape = self.infer_shape_with_prelude(inputs_1)

        # When performing a batch matmul, we need to properly handle N-dim shapes.
        if len(a_shape) > 2 and len(b_shape) > 2:
            # Convert a into a 3 dimensional tensors.
            need_reshape_output = False
            if len(a_shape) != 3:
                a = _op.reshape(inputs_0, [-1, a_shape[-2], a_shape[-1]])
                need_reshape_output = True
            else:
                a = inputs_0

            # Transpose matrix dimensions of b.
            trans_axes = list(range(len(b_shape)))
            trans_axes[-2], trans_axes[-1] = trans_axes[-1], trans_axes[-2]
            b = _op.transpose(inputs_1, trans_axes)

            # Convert b into a 3 dimensional tensor. Note that the last two dimensions
            # are transposed.
            if len(b_shape) != 3:
                b = _op.reshape(b, [-1, b_shape[-1], b_shape[-2]])

            # Perform a batch matmul.
            output = _op.nn.batch_matmul(a, b)

            # Reshape output to original dimensions.
            if need_reshape_output:
                return _op.reshape(output, [*a_shape[:-2], a_shape[-2], b_shape[-1]])
            return output
        elif len(a_shape) > 2:
            inputs_0 = _op.reshape(inputs_0, [-1, a_shape[-1]])

        if len(b_shape) > 2:
            trans_axes = list(range(len(b_shape)))
            trans_axes[-2], trans_axes[-1] = trans_axes[-1], trans_axes[-2]
            input_1 = _op.reshape(_op.transpose(inputs_1, trans_axes), [-1, b_shape[-2]])
        elif len(b_shape) == 2:
            input_1 = _op.transpose(inputs_1, axes=(1, 0))
        elif len(b_shape) == 1:
            input_1 = _op.expand_dims(inputs_1, 0, 1)

        out = _op.nn.dense(inputs_0, input_1)

        if len(b_shape) == 1:
            out = _op.squeeze(out, axis=[-1])

        # Reshape output into a N dimensional tensor when a or b dim > 2
        if len(a_shape) > 2:
            out = _op.reshape(out, [*a_shape[:-1], b_shape[-1]])
        elif len(b_shape) > 2:
            out = _op.reshape(out, [a_shape[-2], -1, b_shape[-1]])
            out = _op.reshape(
                _op.transpose(out, [1, 0, 2]), [*b_shape[:-2], a_shape[-2], b_shape[-1]]
            )

        return out

    def expand(self, inputs, input_types):
        data_in = inputs[0]
        shape = list(self.infer_shape(data_in))

        ndims = len(shape)
        sizes = inputs[1]
        out = data_in

        out_dims = len(sizes)
        if ndims < out_dims:
            num_newaxis = out_dims - ndims
            out = _op.expand_dims(out, axis=0, num_newaxis=num_newaxis)
            shape = [1] * num_newaxis + shape

        for i in range(out_dims):
            if sizes[i] != -1 and shape[i] == 1:
                if not isinstance(sizes[i], int):
                    sizes[i] = int(_infer_value(sizes[i], {}).numpy())
                out = _op.repeat(out, sizes[i], axis=i)

        return out

    def int(self, inputs, input_types):
        if isinstance(inputs[0], _expr.Expr):
            return inputs[0]
        return int(inputs[0])

    def identity(self, inputs, input_types):
        return inputs[0]

    def none(self, inputs, input_types):
        return None

    def make_pad(self, mode):
        def pad(inputs, input_types):
            data = inputs[0]
            if isinstance(inputs[1], list):
                pad_list = inputs[1]
            else:
                pad_list = list(self.infer_shape(inputs[1]))

            # initialize paddings based on input len
            pad_len = len(self.infer_shape(data)) * 2
            paddings = [0] * pad_len

            if len(pad_list) >= 2:
                paddings[-1] = pad_list[1]
                paddings[-2] = pad_list[0]
            if len(pad_list) >= 4:
                paddings[-3] = pad_list[3]
                paddings[-4] = pad_list[2]
            if len(pad_list) >= 6:
                paddings[-5] = pad_list[5]
                paddings[-6] = pad_list[4]

            # group into tuple of 2 ints
            paddings = [paddings[i : i + 2] for i in range(0, len(paddings), 2)]

            const_paddings = []
            for pad in paddings:
                const_paddings.append([])
                for p in pad:
                    if not isinstance(p, int):
                        p = int(_infer_value(p, {}).numpy())
                    const_paddings[-1].append(p)

            if mode == "constant":
                return _op.nn.pad(data, const_paddings, pad_value=inputs[2], pad_mode=mode)
            else:
                return _op.nn.pad(data, const_paddings, pad_mode=mode)

        return pad

    def clamp(self, inputs, input_types):
        data = inputs[0]

        def get_v(v, default_v):
            if isinstance(v, _expr.Constant):
                return float(v.data.numpy())
            if isinstance(v, _expr.Expr):
                infer_v, success = try_infer_value(v, lambda ret: float(ret))
                if success:
                    return infer_v
            if v is not None:
                return v
            return default_v

        amin = get_v(inputs[1], np.finfo(np.float32).min)
        amax = get_v(inputs[2], np.finfo(np.float32).max)
        return _op.clip(data, amin, amax)

    def to(self, inputs, input_types):
        data = inputs[0]
        dtype = inputs[1] if inputs[1] is not None and not isinstance(inputs[1], str) else inputs[2]
        # special handling for aten::to(data, 6, _, _, _) case
        # 6 means dtype = float
        # this happens when converting upsampling with scale factor
        cast_map = {
            5: "float16",
            6: "float32",
            7: "float64",
            3: "int32",
            4: "int64",
        }

        cast_func = {5: float, 6: float, 7: float, 3: int, 4: int}

        ret = data
        if isinstance(data, _expr.Expr):
            actual_dtype = str(self.infer_type(data).dtype)
            if dtype in cast_map and cast_map[dtype] != actual_dtype:
                ret = _op.cast(data, cast_map[dtype])
        elif dtype in cast_map:
            ret = cast_func[dtype](data)

        return ret

    def get_upsample_out_size(self, inputs, method):
        # This assumes a static shape
        out_size = []
        if inputs[1] is not None:
            for size in inputs[1]:
                if not isinstance(size, int):
                    out_size.append(int(_infer_value(size, {}).numpy()))
                else:
                    out_size.append(size)
        else:
            scale_index = 3 if method in ["bilinear", "trilinear"] else 2
            scales = inputs[scale_index]
            assert scales is not None, "neither out size nor scale provided"
            assert isinstance(scales, list)
            ishape = self.infer_shape(inputs[0])
            for i, scale in enumerate(scales):
                out_size.append(int(math.floor(float(ishape[2 + i]) * scale)))

        return out_size

    def make_upsample(self, method):
        def upsample(inputs, input_types):
            data = inputs[0]
            out_size = self.get_upsample_out_size(inputs, method)

            if len(inputs) > 2 and method == "bilinear":
                align_corners = inputs[2]
            else:
                align_corners = False

            if method == "nearest_neighbor":
                coord_trans = "asymmetric"
            elif align_corners:
                coord_trans = "align_corners"
            else:
                coord_trans = "half_pixel"

            def func(x):
                return _op.image.resize(x, out_size, "NCHW", method, coord_trans)

            if self.is_quantized_tensor(data):
                # input qparams are manually appended by us
                assert isinstance(inputs[-2], float)
                assert isinstance(inputs[-1], int)
                input_scale = _expr.const(inputs[-2])
                input_zero_point = _expr.const(inputs[-1])
                return qnn_torch.quantized_upsample(data, input_scale, input_zero_point, func)

            return func(data)

        return upsample

    def make_upsample3d(self, method):
        def upsample3d(inputs, input_types):
            data = inputs[0]
            out_size = self.get_upsample_out_size(inputs, method)

            if len(inputs) > 2 and method == "trilinear":
                align_corners = inputs[2]
            else:
                align_corners = False

            if method == "nearest_neighbor":
                coord_trans = "asymmetric"
            elif align_corners:
                coord_trans = "align_corners"
            else:
                coord_trans = "half_pixel"

            return _op.image.resize3d(data, out_size, "NCDHW", method, coord_trans)

        return upsample3d

    def expand_as(self, inputs, input_types):
        target = inputs[1]
        t0 = self.infer_type(inputs[0]).dtype
        t1 = self.infer_type(inputs[1]).dtype
        if str(t0) != str(t1):
            target = _op.cast(target, t0)
        return _op.broadcast_to_like(inputs[0], target)

    def broadcast_tensors(self, inputs, input_types):
        tensor_list = inputs[0]
        import torch

        res_shape = list(torch.broadcast_shapes(*[self.infer_shape(t) for t in tensor_list]))
        return [_op.broadcast_to(tensor, res_shape) for tensor in tensor_list]

    def Bool(self, inputs, input_types):
        assert len(inputs) == 1
        return inputs[0]

    def Float(self, inputs, input_types):
        assert len(inputs) == 1
        return _op.cast(inputs[0], "float32")

    def mm(self, inputs, input_types):
        return _op.nn.dense(inputs[0], inputs[1])

    def bitwise_not(self, inputs, input_types):
        data = inputs[0]
        # The input tensor must be of integral or Boolean types.
        # For bool tensors, it computes the logical NOT
        if input_types[0] == "bool":
            out = _op.logical_not(_op.cast(data, "bool"))
        else:
            out = _op.bitwise_not(_op.cast(data, "int"))

        return out

    def bitwise_xor(self, inputs, input_types):
        lhs = inputs[0]
        rhs = inputs[1]
        lhs = _op.cast(lhs, "bool") if input_types[0] == "bool" else _op.cast(lhs, "int")
        rhs = _op.cast(rhs, "bool") if input_types[1] == "bool" else _op.cast(rhs, "int")

        return _op.bitwise_xor(lhs, rhs)

    def logical_not(self, inputs, input_types):
        data = _wrap_const(inputs[0])
        return _op.logical_not(_op.cast(data, "bool"))

    def logical_xor(self, inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")
        rhs = _op.cast(inputs[1], "bool")

        return _op.logical_xor(lhs, rhs)

    def list_getitem(self, inputs, input_types):
        return self.prelude.nth(inputs[0], _wrap_const(inputs[1]))

    def list_len(self, inputs, input_types):
        return self.prelude.length(inputs[0])

    def type_as(self, inputs, input_types):
        assert len(inputs) == 2
        assert len(input_types) == 2
        return _op.cast(inputs[0], input_types[1])

    def gather(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        indices = inputs[2]

        return _op.gather(data, axis, indices)

    def add(self, inputs, input_types):
        # add_ is overloaded for tensor add and list concat
        if input_types[0] == "ListType":
            return self.prelude.concat(inputs[0], inputs[1])
        return self.make_elemwise("add")(inputs, input_types)

    def tensor_array_stack(self, inputs, input_types):
        dim = inputs[1]
        assert dim == 0, "stacking on a dynamic tensor list only supported on a first axis"
        tensor_array, shape = self.convert_to_tensor_array(inputs[0])

        stacked_shape = (Any(),) + shape
        stack = self.prelude.get_global_var_static("tensor_array_stack", "float32", shape)
        stacked = stack(tensor_array)

        static_tensor_array_ops = StaticTensorArrayOps(self.prelude, "float32", stacked_shape)
        static_tensor_array_ops.register()
        get_tensor = self.prelude.get_global_var_static("tensor_get_data", "float32", stacked_shape)
        return get_tensor(stacked)

    def stack(self, inputs, input_types):
        if isinstance(inputs[0], list):
            # a static python list of tensors
            dim = inputs[1]
            return _op.stack(inputs[0], dim)
        else:
            # List ADT case
            assert isinstance(inputs[0], _expr.Expr)
            ty = self.infer_type_with_prelude(inputs[0])
            list_ty = self.prelude.mod.get_global_type_var("List")
            msg = "The input list is expected to be List ADT"
            assert isinstance(ty, tvm.ir.TypeCall) and ty.func == list_ty, msg
            return self.tensor_array_stack(inputs, input_types)

    def rsub(self, inputs, input_types):
        data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])

        # TODO (t-vi): should this also be part of the type promotion?
        alpha = _expr.const(float(inputs[2]))

        # note: rsub means data0 and data1 swap places
        return get_relay_op("subtract")(data1, alpha * data0)

    def embedding(self, inputs, input_types):
        weight = inputs[0]
        indices = inputs[1]

        return _op.take(weight, indices.astype("int32"), axis=0)

    def one_hot(self, inputs, input_types):
        indices = inputs[0].astype("int32")
        num_classes = inputs[1]
        if num_classes == -1:
            msg = "Inferring the number of classes is not yet supported."
            raise NotImplementedError(msg)

        dtype = "int32"
        on_value = tvm.relay.const(1.0, dtype)
        off_value = tvm.relay.const(0.0, dtype)

        return _op.one_hot(indices, on_value, off_value, num_classes, -1, dtype)

    def index(self, inputs, input_types):
        data = inputs[0]
        indices = inputs[1]
        return _op.adv_index([data] + indices)

    def meshgrid(self, inputs, input_types):
        data = inputs[0]
        return _op.meshgrid(data, indexing="ij")

    def nms(self, inputs, input_types):
        boxes = inputs[0]
        scores = inputs[1]
        iou_threshold = inputs[2]

        # TVM NMS assumes score > 0
        ### [QTI Change]
        # - since there exists multi-comsumers for "scores", "num_boxes"
        # - invoke set_span here to prevent expr-rewritten occurrs in span-filling stage
        op_node = self.current_op[-1]
        torch_node_name = self._get_node_name(op_node)
        source_name = torch_node_name if torch_node_name else self.source_map[op_node]
        scores = set_span(scores - _op.min(scores) + _op.const(1.0), source_name, op_type=op_node.kind())

        num_boxes = set_span(_op.shape_of(scores), source_name, op_type=op_node.kind())
        # PyTorch NMS doesn't have score_threshold, so no need to run get_valid_count
        # - since "arange" op will fill expr into its attribute
        # - invoke set_span here to prevent expr-rewritten occurrs in span-filling stage
        indices = _op.transform.arange(set_span(_op.squeeze(num_boxes), source_name, op_type=op_node.kind()), dtype="int32")
        ### [End QTI Change]
        indices = _op.expand_dims(indices, 0, 1)

        # Generate data with shape (1, num_anchors, 5)
        scores = AttrCvt(op_name="expand_dims", extras={"axis": -1, "num_newaxis": 1})([scores], {})
        data = _op.concatenate([scores, boxes], -1)
        data = _op.expand_dims(data, 0, 1)

        # Perform Non-Maximum Suppression,
        # PyTorch NMS doesn't have parameter top_k and max_output_size
        score_index = 0
        top_k = max_out_size = -1
        nms_ret = get_relay_op("non_max_suppression")(
            data=data,
            valid_count=num_boxes,
            indices=indices,
            max_output_size=max_out_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=top_k,
            coord_start=1,
            score_index=score_index,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
        )

        # squeeze the two outputs of nms for strided_slice
        size = get_relay_op("squeeze")(nms_ret[1], axis=[1])
        data_slice = get_relay_op("squeeze")(nms_ret[0], axis=[0])

        # strided slice to get the dynamic result
        ret = get_relay_op("strided_slice")(
            data_slice, begin=_expr.const([0]), end=size, slice_mode="size"
        )
        # in torchvision, indices from nms are int64
        return _op.cast(ret, "int64")

    def logsumexp(self, inputs, input_types):
        data = self.pytorch_promote_types(inputs[:1], input_types[:1])
        dim_list = inputs[1]
        keepdim = inputs[2] if len(inputs) > 2 else False
        # dim is output of prim::ListConstruct, even if it is int in python code
        assert isinstance(dim_list, list), "dim is expected to be a list"
        return _op.logsumexp(data[0], axis=dim_list, keepdims=keepdim)

    def roi_align(self, inputs, input_types):
        data = inputs[0]
        boxes = inputs[1]

        output_size = (inputs[3], inputs[4])
        spatial_scale = inputs[2]
        sample_ratio = inputs[5]
        aligned = False if len(inputs) < 7 else inputs[6]

        if aligned:
            boxes -= _expr.const(0.5 / spatial_scale)

        return _op.vision.roi_align(data, boxes, output_size, spatial_scale, sample_ratio)

    def deform_conv2d(self, inputs, input_types):
        data = inputs[0]
        weight = inputs[1]
        offset = inputs[2]
        strides = (inputs[4], inputs[5])
        padding = (inputs[6], inputs[7])
        dilation = (inputs[8], inputs[9])
        groups = inputs[10]
        deformable_groups = inputs[11]
        weight_shape = self.infer_shape(weight)
        output_channels = weight_shape[0]
        kernel_size = (weight_shape[2], weight_shape[3])

        return _op.nn.deformable_conv2d(
            data,
            offset,
            weight,
            strides,
            padding,
            dilation,
            deformable_groups,
            groups,
            output_channels,
            kernel_size,
        )

    def unbind(self, inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        ishapes = self.infer_shape(data)
        if dim >= len(ishapes):
            msg = "Please check input dim, it shouldn't be greater than or equal to rank."
            raise AttributeError(msg)

        selections = ishapes[dim]
        res_split = _op.split(data, selections, dim)
        # squeeze each split piece to get same shape as aten::unbind
        # TODO (yongwww): add new op to avoid the squeeze overhead
        ret = []
        for i in range(selections):
            ret.append(_op.transform.squeeze(res_split[i], axis=[dim]))
        ret = _expr.TupleWrapper(_expr.Tuple(ret), selections)
        return ret

    def shape_as_tensor(self, inputs, input_types):
        is_symbolic_shape = False
        input_shape = self.infer_shape(inputs[0], self.prelude.mod)
        for axis in input_shape:
            if not isinstance(axis, (int, tvm.tir.IntImm)):
                is_symbolic_shape = True
                break

        if is_symbolic_shape:
            ret = _op.shape_of(inputs[0], dtype="int64")
        else:
            ret = _expr.const(np.array(input_shape), dtype="int64")

        return ret

    def logical_and(self, inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")
        rhs = _op.cast(inputs[1], "bool")

        return _op.logical_and(lhs, rhs)

    def nonzero(self, inputs, input_types, is_numpy_style=False):
        data = inputs[0]
        ret = _op.transform.argwhere(data)
        if is_numpy_style or (len(inputs) > 1 and inputs[1]):
            return self.unbind([ret, 1], None)
        return ret

    def nonzero_numpy(self, inputs, input_types):
        return self.nonzero(inputs, input_types, is_numpy_style=False)

    def scatter(self, inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        index = inputs[2]
        src = inputs[3]
        return _op.transform.scatter(data, index, src, axis)

    def index_put(self, inputs, input_types):
        in_tensor = inputs[0]
        indices = inputs[1]
        values = inputs[2]
        accumulate = inputs[3]
        if not accumulate:
            mode = "update"
        else:
            mode = "add"
        # Combine array of index tensors into one index tensor with shape (N,_)
        index_tensor = _op.stack(indices, axis=0)
        return _op.transform.scatter_nd(in_tensor, index_tensor, values, mode)

    def scalar_tensor(self, inputs, input_types):
        data = inputs[0]
        cast_map = {
            6: "float32",
            7: "float64",
            3: "int32",
            4: "int64",
        }
        type_key = inputs[1]
        if isinstance(data, _expr.Constant):
            data = data.data.numpy().tolist()
        return _expr.const(data, cast_map[type_key])

    def interpolate(self, inputs, input_types):
        if isinstance(inputs[1], _expr.Expr):
            out_size = inputs[1]
        elif isinstance(inputs[1], list):
            out_size = []
            for i in [0, 1]:
                size, _ = try_infer_value(
                    inputs[1][i],
                    lambda ret: ret.astype(np.int),
                    lambda: _op.expand_dims(inputs[1][i], axis=0),
                )
                out_size.append(size)
            out_size = _op.concatenate(out_size, axis=0)

        data = inputs[0]
        align_corners = inputs[4]
        method = inputs[3]
        if method.startswith("nearest"):
            method = "nearest_neighbor"

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return _op.image.resize(data, out_size, "NCHW", method, coord_trans)

    def numel(self, inputs, input_types):
        return _op.ndarray_size(inputs[0])

    def empty(self, inputs, input_types):
        shape = inputs[0]
        return _op.zeros(shape, _convert_dtype_value(inputs[1]))

    def bincount(self, inputs, input_types):
        data = inputs[0]
        weights = inputs[1]
        input_type = self.infer_type(data).dtype
        if input_type == "int64":
            logging.warning(
                "Casting an int64 input to int32, since we do not have int64 atomic add"
                "needed for bincount yet."
            )
            data = _op.cast(data, "int32")
        maximum = _op.max(data)
        dim = maximum + _expr.const(1, dtype="int32")
        if weights:
            weight_type = self.infer_type(weights)
            out_dtype = weight_type.dtype
            updates = weights
        else:
            out_dtype = "int32"
            updates = _op.ones_like(data)

        counts = _op.zeros(_op.reshape(dim, [1]), out_dtype)
        out = _op.scatter_add(counts, data, updates, axis=0)
        if input_type == "int32":
            # Torch always outputs int64 results for bincount
            return _op.cast(out, "int64")
        return out

    def scatter_add(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        index = inputs[2]
        src = inputs[3]
        return _op.scatter_add(data, index, src, axis=axis)

    def cumsum(self, inputs, input_types):
        data = inputs[0]
        dim = inputs[1]
        dtype = inputs[2]

        if inputs[2] is not None:
            dtype = _convert_dtype_value(inputs[2])

        return _op.cumsum(data, axis=dim, dtype=dtype)

    def masked_fill(self, inputs, input_types):
        mask = inputs[1]
        value = _op.cast(_wrap_const(inputs[2]), input_types[0])
        return _op.where(mask, value, inputs[0])

    def masked_select(self, inputs, input_types):
        mask = inputs[1]
        indices = self.nonzero([mask], input_types, is_numpy_style=True)
        return _op.adv_index([inputs[0]] + [indices[i] for i in range(indices.size)])

    def sort(self, inputs, input_types):
        data = inputs[0]
        dim = inputs[1]
        is_descending = inputs[2]
        # pytorch sort returns both sorted indices and values
        indices = _op.argsort(data, dim, not is_descending)
        return _op.gather(data, dim, indices), indices

    def argsort(self, inputs, input_types):
        data = inputs[0]
        dim = inputs[1]
        is_descending = inputs[2]
        return _op.argsort(data, dim, not is_descending)

    def is_floating_point(self, inputs, input_types):
        assert len(inputs) == 1

        if isinstance(inputs[0], _expr.Expr):
            input_type = self.infer_type(inputs[0]).dtype
        else:
            input_type = input_types[0]

        is_float = input_type in ["float32", "float64", "float16", "bfloat16"]
        return _expr.const(is_float)

    def unique(self, inputs, input_types):
        assert len(inputs) == 4
        [data, is_sorted, return_inverse, return_counts] = inputs
        if not is_sorted:
            logging.warning("TVM always assumes sorted=True for torch.unique")
            is_sorted = True
        if return_counts:
            [unique, indices, inverse_indices, num_uniq, counts] = _op.unique(
                data, is_sorted=is_sorted, return_counts=True
            )
            unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
            counts_sliced = _op.strided_slice(counts, begin=[0], end=num_uniq, slice_mode="size")
            return (unique_sliced, inverse_indices, counts_sliced)
        else:
            [unique, indices, inverse_indices, num_uniq] = _op.unique(
                data, is_sorted=is_sorted, return_counts=False
            )
            unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
            return (unique_sliced, inverse_indices)

    def flip(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        return _op.transform.reverse(data, axis=axis[0])

    ### [QTI Change]
    def fake_quantize(self, inputs, input_types):
        quantize = qnn.op.quantize(
            data=inputs[0],
            output_scale=_expr.const(inputs[1], 'float32'),
            output_zero_point=_expr.const(inputs[2], 'int32')
        )
        dequantize =qnn.op.dequantize(
            quantize,
            input_scale=_expr.const(inputs[1], 'float32'),
            input_zero_point=_expr.const(inputs[2], 'int32')
        )
        return dequantize
    ### [End QTI Change]

    # Operator mappings
    def create_convert_map(self):
        self.convert_map = {
            "aten::is_floating_point": self.is_floating_point,
            "aten::pixel_shuffle": self.pixel_shuffle,
            "aten::device": self.none,
            "prim::device": self.none,
            "aten::sub": self.make_elemwise("subtract"),
            "aten::sub_": self.make_elemwise("subtract"),
            "aten::max": self.max,
            "aten::min": self.min,
            "aten::mul": self.make_elemwise("multiply"),
            "aten::mul_": self.make_elemwise("multiply"),
            "aten::pow": self.make_elemwise("power"),
            "aten::arange": self.arange,
            "aten::meshgrid": self.meshgrid,
            "aten::div": self.make_elemwise("divide"),
            "aten::div_": self.make_elemwise("divide"),
            "aten::floor_divide": self.make_elemwise("floor_divide"),
            "aten::floor_divide_": self.make_elemwise("floor_divide"),
            "aten::true_divide": self.make_elemwise("divide"),
            "aten::addcdiv": self.addcdiv,
            "aten::addcmul": self.addcmul,
            "aten::ones": self.ones,
            "aten::ones_like": self.ones_like,
            "aten::zeros": self.zeros,
            "aten::zeros_like": self.zeros_like,
            "aten::full": self.full,
            "aten::full_like": self.full_like,
            "aten::linspace": self.linspace,
            "aten::reciprocal": self.reciprocal,
            "aten::repeat": self.repeat,
            "aten::repeat_interleave": self.repeat_interleave,
            "aten::to": self.to,
            "aten::squeeze": self.squeeze,
            "aten::unsqueeze": self.unsqueeze,
            "aten::unsqueeze_": self.unsqueeze,
            "aten::cat": self.concatenate,
            "aten::slice": self.slice,
            "aten::narrow": self.narrow,
            "aten::split": self.split,
            "aten::tensor_split": self.tensor_split,
            "aten::split_with_sizes": self.split_with_sizes,
            "aten::select": self.select,
            "aten::take": self.take,
            "aten::where": self.where,
            "aten::topk": self.topk,
            "aten::relu": self.relu,
            "aten::relu_": self.relu,
            "aten::prelu": self.prelu,
            "aten::leaky_relu": self.leaky_relu,
            "aten::leaky_relu_": self.leaky_relu,
            "aten::elu": self.elu,
            "aten::elu_": self.elu,
            "aten::celu": self.celu,
            "aten::gelu": self.gelu,
            "aten::selu": self.selu,
            "aten::silu": self.silu,
            "aten::glu": self.glu,
            "aten::log_sigmoid": self.log_sigmoid,
            "aten::adaptive_avg_pool1d": functools.partial(
                self.adaptive_avg_pool, _op.nn.adaptive_avg_pool1d
            ),
            "aten::adaptive_avg_pool2d": functools.partial(
                self.adaptive_avg_pool, _op.nn.adaptive_avg_pool2d
            ),
            "aten::adaptive_avg_pool3d": functools.partial(
                self.adaptive_avg_pool, _op.nn.adaptive_avg_pool3d
            ),
            "aten::adaptive_max_pool1d": functools.partial(
                self.adaptive_max_pool, _op.nn.adaptive_max_pool1d
            ),
            "aten::adaptive_max_pool2d": functools.partial(
                self.adaptive_max_pool, _op.nn.adaptive_max_pool2d
            ),
            "aten::adaptive_max_pool3d": functools.partial(
                self.adaptive_max_pool, _op.nn.adaptive_max_pool3d
            ),
            "aten::max_pool2d": self.maxpool_2d,
            "aten::max_pool2d_with_indices": self.maxpool_2d_with_indices,
            "aten::max_pool1d": self.maxpool_1d,
            "aten::max_pool3d": self.maxpool_3d,
            "aten::hardtanh": self.hardtanh,
            "aten::hardtanh_": self.hardtanh,
            "aten::_convolution": self.convolution,
            "aten::softmax": self.softmax,
            "aten::threshold": self.threshold,
            "aten::threshold_": self.threshold,
            "aten::contiguous": self.contiguous,
            "aten::batch_norm": self.batch_norm,
            "aten::instance_norm": self.instance_norm,
            "aten::layer_norm": self.layer_norm,
            "aten::group_norm": self.group_norm,
            "aten::transpose": self.transpose,
            "aten::transpose_": self.transpose,
            "aten::t": self.transpose,
            "aten::flatten": self.flatten,
            "aten::addmm": self.addmm,
            "aten::size": self.size,
            "aten::view": self.view,
            "aten::reshape": self.reshape,
            "aten::clone": self.clone,
            "aten::log_softmax": self.log_softmax,
            "aten::sigmoid": self.sigmoid,
            "aten::softplus": self.softplus,
            "aten::avg_pool1d": self.make_avg_pool(1),
            "aten::avg_pool2d": self.make_avg_pool(2),
            "aten::avg_pool3d": self.make_avg_pool(3),
            "aten::linear": self.linear,
            "aten::dropout": self.dropout,
            "aten::dropout_": self.dropout,
            "aten::feature_dropout": self.dropout,
            "aten::alpha_dropout": self.dropout,
            "aten::mean": self.mean,
            "aten::chunk": self.chunk,
            "aten::matmul": self.matmul,
            "aten::bmm": self.matmul,
            "aten::expand": self.expand,
            "aten::Int": self.int,
            "prim::NumToTensor": self.numtotensor,
            "prim::ImplicitTensorToNum": self.tensortonum,
            "aten::ScalarImplicit": self.tensortonum,
            "aten::constant_pad_nd": self.make_pad("constant"),
            "aten::reflection_pad1d": self.make_pad("reflect"),
            "aten::reflection_pad2d": self.make_pad("reflect"),
            "aten::replication_pad1d": self.make_pad("edge"),
            "aten::replication_pad2d": self.make_pad("edge"),
            "aten::replication_pad3d": self.make_pad("edge"),
            "aten::permute": self.transpose,
            "aten::sum": self.make_reduce("sum"),
            "aten::prod": self.make_reduce("prod"),
            "aten::argmin": self.make_reduce("argmin"),
            "aten::argmax": self.make_reduce("argmax"),
            "aten::norm": self.norm,
            "aten::frobenius_norm": self.frobenius_norm,
            "aten::std": self.std,
            "aten::var": self.variance,
            "aten::abs": self.make_unary("abs"),
            "aten::neg": self.make_unary("negative"),
            "aten::cos": self.make_unary("cos"),
            "aten::cosh": self.make_unary("cosh"),
            "aten::sin": self.make_unary("sin"),
            "aten::sinh": self.make_unary("sinh"),
            "aten::tan": self.make_unary("tan"),
            "aten::tanh": self.make_unary("tanh"),
            "aten::acos": self.make_unary("acos"),
            "aten::asin": self.make_unary("asin"),
            "aten::atan": self.make_unary("atan"),
            "aten::log": self.make_unary("log"),
            "aten::log2": self.make_unary("log2"),
            "aten::log10": self.make_unary("log10"),
            "aten::log1p": self.log1p,
            "aten::exp": self.make_unary("exp"),
            "aten::erf": self.make_unary("erf"),
            "aten::trunc": self.make_unary("trunc"),
            "aten::sign": self.make_unary("sign"),
            "aten::sqrt": self.make_unary("sqrt"),
            "aten::rsqrt": self.make_unary("rsqrt"),
            "aten::square": self.square,
            "aten::ceil": self.make_unary("ceil"),
            "aten::floor": self.make_unary("floor"),
            "aten::round": self.make_unary("round"),
            "aten::isfinite": self.make_unary("isfinite"),
            "aten::isinf": self.make_unary("isinf"),
            "aten::isnan": self.make_unary("isnan"),
            "aten::clamp": self.clamp,
            "aten::clamp_": self.clamp,
            "aten::detach": self.identity,
            "aten::upsample_bilinear2d": self.make_upsample("bilinear"),
            "aten::upsample_nearest2d": self.make_upsample("nearest_neighbor"),
            "aten::upsample_trilinear3d": self.make_upsample3d("trilinear"),
            "aten::upsample_nearest3d": self.make_upsample3d("nearest_neighbor"),
            "aten::expand_as": self.expand_as,
            "aten::broadcast_tensors": self.broadcast_tensors,
            "aten::lt": self.make_elemwise("less"),
            "aten::gt": self.make_elemwise("greater"),
            "aten::le": self.make_elemwise("less_equal"),
            "aten::ge": self.make_elemwise("greater_equal"),
            "aten::ne": self.make_elemwise("not_equal"),
            "aten::eq": self.make_elemwise("equal"),
            "aten::logical_not": self.logical_not,
            "aten::logical_xor": self.logical_xor,
            "aten::bitwise_not": self.bitwise_not,
            "aten::bitwise_xor": self.bitwise_xor,
            "aten::Bool": self.Bool,
            "aten::Float": self.Float,
            "aten::rsub": self.rsub,
            "aten::embedding": self.embedding,
            "aten::one_hot": self.one_hot,
            "aten::mm": self.matmul,
            "aten::add": self.add,
            "aten::add_": self.add,
            "aten::stack": self.stack,
            "aten::__getitem__": self.list_getitem,
            "aten::len": self.list_len,
            "aten::type_as": self.type_as,
            "aten::gather": self.gather,
            "aten::index_select": self.select,
            "aten::index": self.index,
            "torchvision::nms": self.nms,
            "aten::logsumexp": self.logsumexp,
            "torchvision::roi_align": self.roi_align,
            "torchvision::deform_conv2d": self.deform_conv2d,
            "aten::unbind": self.unbind,
            "aten::__and__": self.logical_and,
            "aten::logical_and": self.logical_and,
            "aten::_shape_as_tensor": self.shape_as_tensor,
            "aten::nonzero": self.nonzero,
            "aten::nonzero_numpy": self.nonzero_numpy,
            "aten::scatter": self.scatter,
            "aten::index_put": self.index_put,
            "aten::index_put_": self.index_put,
            "aten::scalar_tensor": self.scalar_tensor,
            "aten::__interpolate": self.interpolate,
            "aten::IntImplicit": self.identity,
            "aten::tensor": self.identity,  # used for example in tensor(1.0)
            "aten::numel": self.numel,
            "aten::empty": self.empty,
            "aten::bincount": self.bincount,
            "aten::scatter_add": self.scatter_add,
            "aten::__not__": self.logical_not,
            "aten::hardswish_": self.hard_swish,
            "aten::hardswish": self.hard_swish,
            "aten::hardsigmoid_": self.hard_sigmoid,
            "aten::hardsigmoid": self.hard_sigmoid,
            "aten::cumsum": self.cumsum,
            "aten::masked_fill": self.masked_fill,
            "aten::masked_select": self.masked_select,
            "aten::argsort": self.argsort,
            "aten::sort": self.sort,
            "aten::_unique2": self.unique,
            "aten::flip": self.flip,
            ### [QTI Change]
            "aten::fake_quantize_per_tensor_affine": self.fake_quantize,
            ### [End QTI Change]
        }

    def update_convert_map(self, custom_map):
        self.convert_map.update(custom_map)

    def report_missing_conversion(self, op_names):
        """Check if all ops in an input graph are supported by TVM"""
        known_ops = [
            "prim::Constant",
            "prim::GetAttr",
            "prim::ListConstruct",
            "prim::ListUnpack",
            "prim::TupleConstruct",
            "prim::TupleUnpack",
            "prim::RaiseException",
            "prim::If",
            "prim::Loop",
        ]
        known_ops += list(self.convert_map.keys())
        known_ops += list(qnn_torch.convert_map.keys())

        missing = [op_name for op_name in op_names if op_name not in known_ops]

        if missing:
            msg = "The following operators are not implemented: {}".format(missing)
            raise NotImplementedError(msg)

    def convert_block(self, block, outputs):
        """Translate Torch "Block", used for prim::If and prim::Loop"""
        ### [QTI Change]
        ops = _get_operator_nodes(
            block.nodes(),
            self.source_map,
            self.op_type_dict,
            self.use_parser_friendly_name,
        )
        ### [End QTI Change]
        ret_names = _get_input_names(block.returnNode())
        return self.convert_operators(ops, outputs, ret_names)

    def convert_if(self, if_node, outputs):
        """Translate Torch prim::If to Relay If"""
        cond = outputs[if_node.inputsAt(0).debugName()]
        blocks = list(if_node.blocks())
        true_branch = self.convert_block(blocks[0], outputs)
        false_branch = self.convert_block(blocks[1], outputs)
        assert len(true_branch) == 1 and len(false_branch) == 1
        return _expr.If(cond, true_branch[0], false_branch[0])

    def convert_loop(self, loop_node, outputs):
        """Translate Torch prim::Loop to Relay while_loop"""

        def get_input(index):
            ivalue = loop_node.inputsAt(index)
            inode = ivalue.node()
            if inode.kind() == "prim::Constant":
                return _expr.const(_get_constant(inode))
            var_name = ivalue.debugName()
            assert var_name in outputs
            return _wrap_const(outputs[var_name])

        # Refer to the spec for prim::Loop below
        # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
        # The first input: %max_trip_count
        # The second input: %initial_condition
        # The rest of input: loop variables
        max_loop_count = get_input(0)
        init_cond = get_input(1)
        num_loop_var = len(list(loop_node.inputs())) - 2
        init_vals = [get_input(i + 2) for i in range(num_loop_var)]

        # while loop has always max_loop_count being int64 max
        # max_loop_count.data (tvm.runtime.NDArray) is -1, so _get_constant again
        is_while_loop = (
            isinstance(max_loop_count, _expr.Constant)
            and _get_constant(loop_node.inputsAt(0).node()) == sys.maxsize
        )

        if is_while_loop:
            loop_iter_dtype = "bool"
            # while loop with non input dependent condition such as while i < 10:
            # init_cond is int, need to cast to bool to type check
            if isinstance(init_cond, _expr.Constant):
                init_cond = _op.cast(init_cond, "bool")
            init_loop_iter_val = init_cond
        else:
            loop_iter_dtype = "int32"
            # always count from 0
            init_loop_iter_val = _expr.const(0, dtype="int32")

        body_block = list(loop_node.blocks())[0]
        block_input_names = _get_input_names(body_block)
        num_block_inputs = len(block_input_names)
        name_val_pairs = list(zip(block_input_names, [init_loop_iter_val] + init_vals))
        outputs.update(name_val_pairs)

        def get_var(name, val):
            if val:
                checked_type = self.infer_type_with_prelude(val)
                if hasattr(checked_type, "shape"):
                    shape = get_const_tuple(checked_type.shape)
                    actual_shape = []
                    for dim in shape:
                        if isinstance(dim, int) and dim == 0:
                            actual_shape.append(Any())
                        else:
                            actual_shape.append(dim)
                ### [QTI Change]
                    expr = _expr.var(name, shape=actual_shape, dtype=checked_type.dtype)
                else:
                    expr = _expr.var(name, type_annotation=checked_type)
                return set_span(expr, val.span) if val.span else expr
                ### [End QTI Change]
            return _expr.var(name)

        ### [QTI Change]
        torch_node_name = self._get_node_name(loop_node)
        source_name = torch_node_name if torch_node_name else self.source_map[loop_node]
        loop_iter_var = set_span(
            _expr.var(block_input_names[0], shape=(), dtype=loop_iter_dtype), span=source_name, op_type=loop_node.kind()
        )
        loop_vars = set_span(
            [get_var(name, val) for name, val in name_val_pairs[1:]], span=source_name, op_type=loop_node.kind()
        )
        ### [End QTI Change]

        # Add non constant free variables to loop variables to prevent code blow up
        # Without this, if there are two for loops in a row, which often happens
        # if the outer loop is unrolled, the computation corresponding to the first for loop
        # is inlined inside loop body, turning O(N) + O(N) computation into O(N^2).
        # This issue was found when converting from Stacked LSTM test. Torch does not add the
        # outputof the eariler loop into loop variables of the next loop.
        # So the variable corresponding to the first loop output appears free in the second
        # loop body.
        free_vars = [
            var
            for var in _get_free_vars_from_block(body_block)
            if var in outputs
            and not isinstance(outputs[var], (_expr.Constant, int, float, str))
            and outputs[var]
        ]

        prev_outputs = {}
        for name in free_vars:
            prev_output = outputs[name]
            new_loop_var = get_var(name, prev_output)
            prev_outputs[name] = prev_output
            outputs[name] = set_span(new_loop_var, source_name)
            loop_vars.append(new_loop_var)
            init_vals.append(prev_output)

        def cond(*current_vals):
            i = current_vals[0]

            if is_while_loop:
                return _op.equal(i, _expr.const(True, "bool"))

            return _op.less(i, max_loop_count)

        def body(*current_vals):
            # Update loop variables using the prev iteration outputs
            assert len(current_vals) == num_block_inputs + len(free_vars)

            for (i, val) in enumerate(current_vals):
                if i < num_block_inputs:
                    outputs[block_input_names[i]] = val
                else:
                    outputs[free_vars[i - num_block_inputs]] = val

            block_outputs = self.convert_block(body_block, outputs)
            block_outputs += [outputs[name] for name in free_vars]

            if not is_while_loop:
                # iter var increment implicit in torch, so do it manually
                # for while loop, block_outputs[0] is already a boolean,
                # the result of termination check
                incr = _expr.const(1, dtype="int32")
                block_outputs[0] = current_vals[0] + incr

            return block_outputs

        loop = while_loop(cond, [loop_iter_var] + loop_vars, body)
        loop_val = loop(init_loop_iter_val, *init_vals)

        # restore original output values for free vars
        outputs.update(prev_outputs)

        # The first element is a loop counter or boolean condition, ignore it
        return [_expr.TupleGetItem(loop_val, i + 1) for i in range(num_loop_var)]

    def convert_operators(self, operators, outputs, ret_names):
        """Convert each Torch IR operators to Relay equivalent"""
        # an op node might not belong to any of scope in trace info natively
        # use a cunter to prevent from messing up its scope in span
        empty_counter = 0
        for node_name, op_node in operators:
            operator = op_node.kind()
            inputs = _get_op_inputs(op_node, outputs)
            ### [QTI Change]
            # we need to record what current operator is to provide correct source name
            # for operators needed to be taken care with (e.g. nms / arange ...)
            self.current_op.append(op_node)
            ### [End QTI Change]

            if operator == "prim::Constant":
                outputs[node_name] = _get_constant(op_node)
            elif operator == "prim::ListConstruct" and _should_construct_dynamic_list(op_node):
                ### [QTI Change]
                outputs[node_name] = set_span(
                    self.convert_to_list_adt(inputs),
                    self.source_map[op_node],
                    op_node.kind(),
                    self._get_node_name(op_node)
                )
                ### [End QTI Change]
            elif operator == "prim::ListConstruct":
                # This assumes that no more elements will be appended to this list
                # In this case, we keep the Python list
                outputs[node_name] = inputs
            elif operator == "prim::TupleConstruct":

                def _handel_nested_input(inputs):
                    inputs_list = []
                    for i, _ in enumerate(inputs):
                        if isinstance(inputs[i], list):
                            inputs_list.append(_handel_nested_input(inputs[i]))
                        else:
                            assert isinstance(inputs[i], _expr.Expr)
                            inputs_list.append(inputs[i])
                    return _expr.Tuple(inputs_list)

                ### [QTI Change]

                outputs[node_name] = set_span(
                    _handel_nested_input(inputs),
                    self.source_map[op_node],
                    op_node.kind(),
                    self._get_node_name(op_node)
                )
                ### [End QTI Change]
            elif operator in ["prim::ListUnpack", "prim::TupleUnpack"]:
                assert len(inputs) == 1
                if isinstance(inputs[0], (list, _expr.TupleWrapper)):
                    unpacked = inputs[0]
                else:
                    unpacked = _unpack_tuple(inputs[0])
                ### [QTI Change]
                torch_node_name = self._get_node_name(op_node)
                source_name = torch_node_name if torch_node_name else self.source_map[op_node]
                outputs.update(
                    zip(
                        _get_output_names(op_node),
                        set_span(unpacked, source_name, op_type=op_node.kind())
                    )
                )
                ### [End QTI Change]
            elif operator == "prim::prim::RaiseException":
                logging.warning("raising exceptions is ignored")
                outputs[node_name] = None
            elif operator == "prim::If":
                if_out = self.convert_if(op_node, outputs)
                ### [QTI Change]
                torch_node_name = self._get_node_name(op_node)
                source_name = torch_node_name if torch_node_name else self.source_map[op_node]
                outputs[node_name] = set_span(
                    if_out,
                    source_name,
                    op_type=op_node.kind(),
                )
                ### [End QTI Change]
            elif operator == "prim::Loop":
                loop_out = self.convert_loop(op_node, outputs)
                unpacked_names = _get_output_names(op_node)
                assert len(loop_out) == len(unpacked_names)
                ### [QTI Change]
                torch_node_name = self._get_node_name(op_node)
                source_name = torch_node_name if torch_node_name else self.source_map[op_node]
                outputs.update(
                    zip(
                        unpacked_names,
                        set_span(loop_out, source_name, op_type=op_node.kind())
                    )
                )
                ### [End QTI Change]
            else:
                relay_op = self.convert_map[operator]

                ### [QTI Change]
                self._set_parameter_source_name(op_node, outputs)
                relay_out = relay_op(
                    # since the elements in "outputs" may change due to span-filling process
                    # we have to call "_get_op_inputs" again rather than use "inputs" directly
                    _get_op_inputs(op_node, outputs),
                    _get_input_types(op_node, outputs, default_dtype=self.default_dtype),
                )
                torch_node_name = self._get_node_name(op_node)
                source_name = torch_node_name if torch_node_name else self.source_map[op_node]
                relay_out = set_span(relay_out, source_name, op_type=op_node.kind())
                ### [End QTI Change]
                self.record_output_type(relay_out)

                if isinstance(relay_out, tuple):
                    # This is for torch operators that return multiple outputs
                    # See _adaptive_max_2d above for example
                    out_names = _get_output_names(op_node)
                    outputs.update(zip(out_names, relay_out))
                else:
                    assert op_node.outputsSize() == 1
                    outputs[node_name] = relay_out

            self.current_op.pop()

        return [_wrap_const(outputs[ret_name]) for ret_name in ret_names]

    # ### [QTI Change]
    def _get_node_name(self, op_node):
        node_names = [self.output_name_to_node_name[output.debugName()] for output in op_node.outputs() if output.debugName() in self.output_name_to_node_name]
        if len(node_names) == len(list(op_node.outputs())) and len(set(node_names)) == 1:
            node_name = node_names[0]
        else:
            node_name = ''
        return node_name
    ### [End QTI Change]

    ### [QTI Change]
    def _set_parameter_source_name(self, op_node, outputs):
        """A helper function to rewrite source_name of parameter."""
        for name in _get_input_names(op_node):
            expr = outputs[name]
            if isinstance(expr, (_expr.Var, _expr.Constant)):
                name_sep = "_" if self.use_parser_friendly_name else "."
                source_name = [self.source_map[op_node]]
                if isinstance(expr, _expr.Var):
                    # variable name should have contained node source name
                    # for op with attributes in convert_params stage
                    # e.g. "aten::batch_norm_5.running_mean"
                    if expr.name_hint.startswith(source_name[0]):
                        source_name[0] = expr.name_hint.split(name_sep)[-1]
                    else:
                        source_name.append(expr.name_hint.split(name_sep)[-1])

                if isinstance(expr, _expr.Var):
                    new_expr = set_span(expr, name_sep.join(source_name), op_type=op_node.kind(), output_names=[expr.name_hint])
                else:
                    new_expr = set_span(expr, name_sep.join(source_name), op_type=op_node.kind())
                outputs[name] = new_expr
    ### [End QTI Change]


def _pytorch_result_type(dtypes, non_tensor_inputs):
    """This promotes TVM dtypes like PyTorch would"""
    import torch

    dtype_map = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if len(dtypes) > 0:
        result_type = dtypes[0]
        for dt in dtypes[1:]:
            if dt != result_type:  # we don't want to work with same types as we
                # don't do quantized here (which cannot be promoted?)
                result_type = _convert_data_type(
                    str(
                        torch.result_type(
                            torch.zeros((), dtype=dtype_map[result_type]),
                            torch.zeros((), dtype=dtype_map[dt]),
                        )
                    )
                )
    else:
        result_type = "bool"  # this is the smallest type...
    for inp in non_tensor_inputs:
        result_type = _convert_data_type(
            str(torch.result_type(torch.zeros((), dtype=dtype_map[result_type]), inp))
        )
    return result_type


# Helper functions for operator implementation
def _convert_dtype_value(val):
    """converts a PyTorch the PyTorch numeric type id to a torch scalar type."""
    convert_torch_dtype_map = {
        11: "torch.bool",
        7: "torch.float64",
        6: "torch.float32",
        5: "torch.float16",
        4: "torch.int64",
        3: "torch.int32",
        2: "torch.int16",
        1: "torch.int8",
        0: "torch.uint8",
        None: "torch.int64",
    }  # Default is torch.int64
    if val in convert_torch_dtype_map:
        return _convert_data_type(convert_torch_dtype_map[val])
    else:
        msg = "Torch data type value %d is not handled yet." % (val)
        raise NotImplementedError(msg)


def _convert_data_type(input_type, default_dtype=None):
    """converts the PyTorch scalar type input_type to a TVM dtype.
    optionally, default_dtype can be a TVM dtype that is used
    if input_type is None (but not when it is unknown)"""
    if input_type is None and default_dtype is not None:
        return default_dtype

    input_type = input_type.lower()
    if input_type in ["double", "float64", "torch.float64"]:
        return "float64"
    elif input_type in ["float", "float32", "torch.float32"]:
        return "float32"
    elif input_type in ["half", "float16", "torch.float16"]:
        return "float16"
    elif input_type in ["long", "int64", "torch.int64"]:
        return "int64"
    elif input_type in ["int", "int32", "torch.int32"]:
        return "int32"
    elif input_type in ["short", "int16", "torch.int16"]:
        return "int16"
    elif input_type in ["char", "int8", "torch.int8"]:
        return "int8"
    elif input_type in ["byte", "uint8", "torch.uint8"]:
        return "uint8"
    elif input_type in ["quint8", "torch.quint8"]:
        return "quint8"
    elif input_type in ["qint8", "torch.qint8"]:
        return "qint8"
    elif input_type in ["qint32", "torch.qint32"]:
        return "qint32"
    elif input_type in ["bool", "torch.bool"]:
        return "bool"
    elif input_type in ["str"]:
        return "str"
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(input_type))
    return "float32"  # Never reached


def _create_typed_const(data, dtype):
    """create a (scalar) constant of given value and dtype.
    dtype should be a TVM dtype"""

    if dtype == "float64":
        typed_data = _expr.const(np.float64(data), dtype=dtype)
    elif dtype == "float32":
        typed_data = _expr.const(np.float32(data), dtype=dtype)
    elif dtype == "float16":
        typed_data = _expr.const(np.float16(data), dtype=dtype)
    elif dtype == "int64":
        typed_data = _expr.const(np.int64(data), dtype=dtype)
    elif dtype == "int32":
        typed_data = _expr.const(np.int32(data), dtype=dtype)
    elif dtype == "int16":
        typed_data = _expr.const(np.int16(data), dtype=dtype)
    elif dtype == "int8":
        typed_data = _expr.const(np.int8(data), dtype=dtype)
    elif dtype == "uint8":
        typed_data = _expr.const(np.uint8(data), dtype=dtype)
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(dtype))
    return typed_data


def _wrap_const(c):
    if not isinstance(c, (_expr.Expr, list, tvm.tir.expr.Any)):
        return _expr.const(c)
    return c


def _run_jit_passes(graph, enable_lower_all_tuples=True):
    """The inline pass is necessary to unwrap prim::CallMethod"""
    # pylint: disable=c-extension-no-member
    import torch

    if is_version_greater_than("1.5.1"):
        # This is required for torchvision detection models from 1.6 above
        # It is the same as _jit_pass_inline, except that it has some special
        # case behaviors for some ops such as aten::__interpolate()
        torch._C._jit_pass_onnx_function_substitution(graph)
    else:
        torch._C._jit_pass_inline(graph)

    if enable_lower_all_tuples:
        torch._C._jit_pass_lower_all_tuples(graph)



def _get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.var(name, shape=tensor.shape, dtype=tensor.dtype)
    return tensor, var


def _get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def _get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def _get_input_names(node_or_graph):
    return [inp.debugName() for inp in node_or_graph.inputs()]


def _get_op_inputs(op_node, outputs):
    return [outputs[name] for name in _get_input_names(op_node)]


def _get_node_type(node):
    assert node.outputsSize() == 1
    return node.output().type().kind()


def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses


def _get_users(node):
    return [use.user for use in _get_uses(node)]


def _getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert len(attribute_names) == 1
    attr_name = node.s(attribute_names[0])
    return attr_name


def _getattr_full_name(getattrs):
    return ".".join([_getattr_attr_name(node) for node in getattrs])


def _get_pytorch_value_type(typ, default_dtype="float32"):
    kind = typ.kind()
    if kind == "TensorType":
        if typ.scalarType() is None:
            # Tensor's type can be unknown if we use torch.jit.script(...)
            # Defaults can be passed in, if not it is float32
            logging.warning("Untyped Tensor found, assume it is %s", default_dtype)
            return default_dtype
        else:
            return _convert_data_type(typ.scalarType())

    elif kind == "ListType":
        return "ListType"
    elif kind in ["IntType", "FloatType", "BoolType", "StringType", "OptionalType"]:
        pt_dtype = str(typ).lower()
        dtype = pt_dtype if pt_dtype == "OptionalType" else _convert_data_type(pt_dtype)
        return dtype
    else:
        return "UnsupportedType"


def _get_input_types(op_node, outputs, default_dtype="float32"):
    """Returns a TVM dtype for each input nodes derived from the torch type"""
    in_types = []
    for inp in op_node.inputs():
        if inp.node().kind() == "prim::GetAttr":
            # GetAttr nodes always return None when we call scalarType() on it
            name = inp.debugName()
            assert name in outputs
            if isinstance(outputs[name], _expr.Var):
                in_types.append(outputs[name].type_annotation.dtype)
            else:
                # For quantized modules with parameters, here we would get
                # "prim::GetAttr[name="_packed_params"]". Since the dtype corresponding to
                # _packed_params is not needed by quantized ops, we return an arbitrary type.
                in_types.append(default_dtype)
        else:
            in_types.append(_get_pytorch_value_type(inp.type(), default_dtype=default_dtype))

    return in_types


def _get_constant(node):
    """Retrieve a constant associated with this prim::Constant node"""
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty == "IntType":
            return node.i(attr_name)
        elif ty == "BoolType":
            return bool(node.i(attr_name))
        elif ty in ["FloatType", "LongType"]:
            return node.f(attr_name)
        elif ty in ["TensorType", "CompleteTensorType"]:
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if len(tensor.shape) == 0:  # tensor(0.1)
                # TODO(t-vi): When is this needed?
                return tensor.item()
            return _wrap_const(tensor.numpy())
        elif ty in ["DeviceObjType", "StringType"]:
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        else:
            raise NotImplementedError("Unsupported type: %s" % ty)
    else:
        assert num_attributes == 0
        return None


### [QTI Change]
def _rename_outputs(node, source_map, op_type_dict, use_parser_friendly_name):
    """Rewrite debug name of node outputs with its operator type"""

    def _get_source_name(op_type):
        op_idx = 0
        if op_type in op_type_dict:
            op_idx = op_type_dict[op_type] + 1
        op_type_dict[op_type] = op_idx
        return "_".join([op_type, str(op_idx)])

    # get source name of operator and rename all of its outputs
    # e.g. node.kind(): aten::adaptive_max_pool2d
    # node_src_name -> aten::adaptive_max_pool2d_x
    # output_1 -> aten::adaptive_max_pool2d_x_0
    # output_2 -> aten::adaptive_max_pool2d_x_1
    if node.kind() != "prim::GetAttr":
        node_src_name = _get_source_name(node.kind())
        for index, output in enumerate(node.outputs()):
            output.setDebugName("_".join([node_src_name, str(index)]))
        # update source map
        # if use_parser_friendly_name is True: e.g. prim::Constant_0 -> prim__Constant_0
        if use_parser_friendly_name:
            node_src_name = re.sub(r":|\.", "_", node_src_name)
        source_map[node] = node_src_name
### [End QTI Change]


### [QTI Change]
def _debug_rename(graph, use_parser_friendly_name):
    """Returns map between node and source name"""
    source_map, op_type_dict = {}, {}
    prim_with_blocks = ["prim::If", "prim::Loop"]

    def _traverse_graph(nodes):
        for node in nodes:
            if node.outputsSize() == 0:
                continue
            if node.kind() in prim_with_blocks:
                for block in node.blocks():
                    _traverse_graph(block.nodes())
            _rename_outputs(node, source_map, op_type_dict, use_parser_friendly_name)

    _traverse_graph(graph.nodes())
    return source_map
### [End QTI Change]


### [QTI Change]
def _get_operator_nodes(
    nodes,
    source_map=None,
    op_type_dict=None,
    use_parser_friendly_name=False,
):
### [End QTI Change]
    """Returns torch IR nodes that need conversion to Relay"""
    ### [QTI Change]
    ops, should_rename_graph = [], all([source_map, op_type_dict]) is not None
    ### [End QTI Change]

    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() == 0:
            continue

        ### [QTI Change]
        if should_rename_graph:
            _rename_outputs(node, source_map, op_type_dict, use_parser_friendly_name)
        ### [End QTI Change]

        if node.outputsSize() > 1:
            node_name = "_".join(_get_output_names(node))
        else:
            node_name = _get_output_name(node)

        if node.kind() != "prim::GetAttr":
            ops.append((node_name, node))

    return ops


def _get_relay_input_vars(graph, input_infos, prelude, is_module=True, default_dtype="float32"):
    """
    Return Relay vars from input shapes and create entries based on
    expected graph inputs - to allow translation
    """

    graph_inputs = list(graph.inputs())
    if is_module:
        # a module has "self" as first input, which we do not need/want
        graph_inputs = graph_inputs[1:]

    if not isinstance(input_infos, list):
        msg = "Graph inputs input_infos should be a list"
        raise RuntimeError(msg)

    if len(graph_inputs) != len(input_infos):
        msg = "PyTorch has {} inputs and input_infos lists {}.".format(
            len(graph_inputs), len(input_infos)
        )
        raise RuntimeError(msg)

    def get_relay_ty(ishape, itype, pt_type):
        if pt_type.kind() == "TensorType":
            if not (_is_int_seq(ishape) or len(ishape) == 0):
                msg = "Shape for Tensors must be lists of ints"
                raise RuntimeError(msg)
            if (pt_type.dim() is not None and pt_type.dim() != len(ishape)) or (
                pt_type.sizes() is not None
                and any([s1 != s2 for s1, s2 in zip(pt_type.sizes(), ishape)])
            ):
                msg = "Shapes of input list and information in the graph do not match"
                raise RuntimeError(msg)
            pt_dtype = pt_type.scalarType()
            if not pt_dtype and itype:
                pt_dtype = itype
            dtype = _convert_data_type(pt_dtype, default_dtype=default_dtype)
            return TensorType(ishape, dtype)
        elif pt_type.kind() == "TupleType":
            if not isinstance(ishape, tuple):
                msg = "Shapes for tuples must be tuples"
                raise RuntimeError(msg)
            return TupleType(
                [get_relay_ty(elem, itype, pt_t) for elem, pt_t in zip(ishape, pt_type.elements())]
            )
        elif pt_type.kind() == "ListType":
            if not isinstance(ishape, list):
                msg = "Shapes for lists must be lists"
                raise RuntimeError(msg)
            pt_elemtype = pt_type.getElementType()
            elem_tys = [get_relay_ty(s, itype, pt_elemtype) for s in ishape]
            if len(elem_tys) > 0 and not all(map(lambda ty: ty == elem_tys[0], elem_tys)):
                msg = "List elements need have identical types"
                raise RuntimeError(msg)
            rlist, _, _ = prelude.mod.get_type("List")
            return rlist(elem_tys[0])
        elif pt_type.kind() == "OptionalType":
            # we do not support None yet, so we fill in the type
            return get_relay_ty(ishape, itype, pt_type.getElementType())
        # TODO: scalar inputs
        raise NotImplementedError("unsupported input type")

    input_vars = {}

    new_input_infos = []
    for num, inp in enumerate(input_infos):
        if not isinstance(inp, tuple):
            msg = "Graph input {} is not a tuple".format(num)
            raise RuntimeError(msg)
        if len(inp) != 2 or not isinstance(inp[0], str):
            msg = (
                "Graph input {} is not valid,"
                " expected ('name', shape) or ('name', (shape, dtype))".format(inp)
            )
            raise RuntimeError(msg)
        if not isinstance(inp[1], tuple) or len(inp[1]) == 0 or not isinstance(inp[1][-1], str):
            new_input_infos.append((inp[0], (inp[1], default_dtype)))
        else:
            new_input_infos.append(inp)

    input_types = [
        (name, get_relay_ty(info[0], info[1], gi.type()))
        for (name, info), gi in zip(new_input_infos, graph_inputs)
    ]

    ir_inputs = [i.debugName() for i in graph_inputs]
    for ir_input, (name, itype) in zip(ir_inputs, input_types):
        inp = _expr.var(name, type_annotation=itype)
        # Translate from graph input to user input name
        input_vars[ir_input] = inp

    return input_vars


def _unpack_tuple(tup):
    def unpack(tup, num_fields):
        return [_expr.TupleGetItem(tup, i) for i in range(num_fields)]

    if isinstance(tup, _expr.Tuple):
        return unpack(tup, len(tup.fields))
    elif isinstance(tup.type_annotation, TupleType):
        return unpack(tup, len(tup.type_annotation.fields))
    # shouldn't happen
    assert False


def _get_free_vars_from_block(block):
    block_inp_names = _get_input_names(block)
    bound_names = block_inp_names
    free_vars = set()

    for node in block.nodes():
        inp_names = _get_input_names(node)
        list_diff = [name for name in inp_names if name not in bound_names]
        free_vars.update(list_diff)
        bound_names += _get_output_names(node)

    return free_vars


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """

    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """

    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


### [QTI Change]
def convert_params(graph, state_dict, source_map, torch_source_map, use_parser_friendly_name=False):
### [End QTI Change]
    """
    Return Relay vars and TVM NDArrays for input parameters
    A chain of prim::GetAttr nodes is processed one at a time
    """
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    params = {}
    param_tensors = {}
    packed_param_map = {}
    ### [QTI Change]
    param_debug_name_map = {}
    ### [End QTI Change]
    vars_by_name = {}
    seen = set()
    ### [QTI Change]
    attr_name_sep = "."
    ### [End QTI Change]

    for node in getattr_nodes:
        if _get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(_get_output_name, getattrs))

            full_attr = _getattr_full_name(getattrs)
            full_attr_node_name = _get_output_name(getattrs[-1])
            ### [QTI Change]
            # set variable name by concatenating first consumer's name with full attribute
            # e.g. "model.bn1.running_mean"
            users = _get_users(getattrs[-1])
            if len(users):
                var_name = attr_name_sep.join(
                    [
                        torch_source_map[users[0]],
                        full_attr.split(attr_name_sep)[-1],
                    ]
                )
            else:
                var_name = full_attr
            ### [End QTI Change]

            if full_attr.endswith("_packed_params"):  # for quantized models
                packed_param_map[full_attr_node_name] = full_attr
            elif full_attr in state_dict:
                ### [QTI Change]
                if var_name in vars_by_name:
                    var = vars_by_name[var_name]
                else:
                    torch_tensor = state_dict[full_attr]
                    tensor, var = _get_tensor_and_var(torch_tensor, var_name)
                    param_tensors[var_name] = tensor
                    # for quantized parameters to be correctly located
                    param_debug_name_map[full_attr_node_name] = var_name
                    vars_by_name[var_name] = var
                ### [End QTI Change]
                params[full_attr_node_name] = var

    ### [QTI Change]
    return params, param_tensors, packed_param_map, param_debug_name_map
    ### [End QTI Change]


def get_all_op_names(graph):
    """Return all operator names in the input graph"""
    nodes = list(graph.nodes())
    prim_with_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_with_blocks:
        prim_nodes = graph.findAllNodes(prim, recurse=True)
        for prim_node in prim_nodes:
            for block in prim_node.blocks():
                nodes += block.nodes()
    return set(node.kind() for node in nodes)


### [QTI Change]
def export_c_graph(location, graph):
    """Convert the graph to an onnx model and export it to the location."""
    import datetime
    import os

    if not os.path.exists(location):
        os.makedirs(location)
    time_stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    fname = os.path.join(location, "tvm_exported_c_graph_{}.txt".format(time_stamp))
    with open(f"{fname}", "w") as f:
        f.write(str(graph))
### [End QTI Change]


### [QTI Change]
def get_mangled_name_from_zipinfo(node, archive, zipinfos):
    # node.sourceRange() contain line number which indicating where the node come from,
    # we first extract line number from sourceRange, and then get its class name(mangled_name)
    # an example of node.sourceRange is showed below,
    # Serialized   File "code/__torch__/models.py", line 52
    #     _22 = (_18).forward(_20, )
    #     _23 = (_16).forward((_17).forward(_22, ), )
    #     input0 = torch.add(_23, _22, alpha=1)
    #                                     ~ <--- HERE
    #     input1 = torch.cat([(_15).forward(input0, ), _21], 1)
    #     return (_14).forward(input1, )

    # parse file path
    match = re.search('File "code/(\S+).py"', node.sourceRange())
    if not match:
        return None

    assert len(match.groups()) == 1
    code_filepath = match.group()[6:-1]
    # read file
    with archive.open(zipinfos[code_filepath].filename) as f:

        # parse line number. e.g., line 52
        line_number = re.findall('line (\d+)', node.sourceRange())[-1]
        if not line_number:
            return None

        if not line_number.isdigit():
            return None

        mangled_name = None
        line_number = int(line_number)-1
        codes = f.readlines()
        # reversely check line in file until there is a "class" string in line
        for i in range(line_number, -1, -1):
            if b'def forward(self: ' in codes[i]:
                code = codes[i].decode("utf-8")
                mangled_name_match = re.search('def forward\(self: (\S+),', code)
                if mangled_name_match:
                    mangled_name = mangled_name_match.group()[18:-1]
                break
        return mangled_name
### [End QTI Change]


### [QTI Change]
def _parse_graph(graph, model, output_name_to_node_name, input_model_path, torch_source_map):
    """
    given pytorch IR, populate output_name_to_node_name mapping

    e.g., given following IR
    graph(%model : __torch__.torch.fx.graph_module.___torch_mangle_0.GraphModule,
          %input : Tensor):
        %1 : __torch__.torch.nn.modules.linear.___torch_mangle_1.Linear = prim::GetAttr[name="fc"](%model)
        %2 : __torch__.torch.nn.modules.activation.___torch_mangle_2.ReLU = prim::GetAttr[name="relu1"](%model)
        %3 : Tensor = prim::GetAttr[name="weight"](%1)
        %4 : Tensor = prim::GetAttr[name="bias"](%1)
        %5 : Tensor = aten::linear(%input, %3, %4)
        %6 : Tensor = aten::relu(%5) # sourceRange contain __torch__.torch.nn.modules.activation.___torch_mangle_2.ReLU
        return (%6)
    populate mapping {'5': 'fc', '6': 'relu1'} to output_name_to_node_name

    :param graph: torch._C.Graph
    :param model: torch.nn.Module
    :param output_name_to_node_name: a dict map output_name to node_name, e.g., output_name_to_node_name = {'5': 'fc', '6': 'relu1'}
    """

    import zipfile
    archive = zipfile.ZipFile(input_model_path, 'r')
    zipinfos = {}
    for name, zipinfo in archive.NameToInfo.items():
        # quant_sim.torchscript/code/__torch__.py => code/__torch__.py
        name = name.split('/', 1)[-1]
        zipinfos[name] = zipinfo

    # get mangled_name dict
    # e.g., mangled_name_to_node_name = {'__torch__.torch.nn.modules.activation.___torch_mangle_2': 'relu1'}
    mangled_name_to_node_name = {}
    for node_name, module in model.named_modules():
        # qualified_name=module hierarchy + classname, where module hierarchy is mangled name
        # check https://github.com/pytorch/pytorch/blob/5bbec68/torch/_jit_internal.py#L1124
        # and https://github.com/pytorch/pytorch/blob/5bbec68/torch/_jit_internal.py#L1177
        mangled_name = module._c.qualified_name
        mangled_name_to_node_name[mangled_name] = node_name

    # node in graph are deserialized python source code
    # check https://github.com/pytorch/pytorch/blob/5bbec68/torch/csrc/jit/docs/serialization.md#code-object-naming
    # we use mangled name to map to node name
    for node in graph.nodes():
        outputs = [output for output in node.outputs()]
        for output in outputs:
            output_name = output.debugName()
            mangled_name = get_mangled_name_from_zipinfo(node, archive, zipinfos)
            if mangled_name and mangled_name in mangled_name_to_node_name:
                output_name_to_node_name[output_name] = mangled_name_to_node_name[mangled_name]

    # get torch source map
    prim_with_blocks = ["prim::If", "prim::Loop"]

    def parse_torch_node_name(nodes):
        for node in nodes:
            if node.outputsSize() == 0:
                continue
            if node.kind() in prim_with_blocks:
                for block in node.blocks():
                    parse_torch_node_name(block.nodes())
            # check if each debug name for output in a node mapped to same node name
            node_names = []
            for index, output in enumerate(node.outputs()):
                if output.debugName() in output_name_to_node_name:
                    node_names.append(output_name_to_node_name[output.debugName()])
            # if all output come from same node and their name are the same, use that name as node name for
            if len(node_names) == len(list(node.outputs())) and len(set(node_names)) == 1:
                # populate node name from AIMET
                torch_source_map[node] = output_name_to_node_name[list(node.outputs())[0].debugName()]

    parse_torch_node_name(graph.nodes())
### [End QTI Change]


### [QTI Change]
def from_pytorch(
    script_module,
    input_model_path,
    input_infos,
    custom_convert_map=None,
    default_dtype="float32",
    use_parser_friendly_name=False,
    export_renamed_c_graph_path=None,
):
### [End QTI Change]
    """Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    ### [QTI Change]
    input_model_path : str
        path to torchscript
    ### [End QTI Change]

    input_infos : List of tuples
        Can be (input name, input shape) or (input name, (input shape, input types))
        Graph level input shape and type list
        The same input names need to be used for deployment, so choose easy to
        remember names (such as: input0, input1)
        e.g.
        [('input0', (1, 2)), ('input1', (3, 4))]
        or
        [('input0', ((1, 2), 'int')), ('input1', ((3, 4), 'float'))]

    custom_convert_map : Dictionary of str to Relay op
        A custom op conversion map in the same format as _convert_map above

    ### [QTI Change]
    use_parser_friendly_name : bool
        When True, replace '.' with `_' in a original parameter name.
        The Relay text parser treats a variable name followed by a period as a tuple element access,
        so a variable name like "dense.weight" cannot be parsed correctly.
        Use this option when you want to run the AnnotateSpans pass on the imported module.

    export_renamed_c_graph_path : str, optional
        Export the renamed torch._C.Graph to the path.
        During the conversion, variable names in torch._C.Graph will be assigned based on their op
        types. The exported text file can be the reference to spans.
    ### [End QTI Change]

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    import torch

    mod = tvm.IRModule()
    prelude = Prelude(mod)
    ### [QTI Change]
    enable_lower_all_tuples = True

    converter = PyTorchOpConverter(prelude, default_dtype, use_parser_friendly_name)
    ### [End QTI Change]

    graph = script_module.graph.copy()

    ### [QTI Change]
    # Check if lower_all_tuples pass can be enabled
    graph_inputs = list(graph.inputs())
    for inp in graph_inputs:
        if inp.type().kind() == "TupleType" or inp.type().kind() == "ListType":
            enable_lower_all_tuples = False
            break
    _run_jit_passes(graph, enable_lower_all_tuples)
    ### [End QTI Change]


    if custom_convert_map:
        converter.update_convert_map(custom_convert_map)

    op_names = get_all_op_names(graph)
    converter.report_missing_conversion(op_names)

    is_module = isinstance(script_module, torch.jit.ScriptModule)
    params = script_module.state_dict() if is_module else {}
    outputs = _get_relay_input_vars(
        graph, input_infos, prelude, default_dtype=default_dtype, is_module=is_module
    )

    ### [QTI Change]
    # rename _C.Graph here for constructing meaningful source name of graph nodes
    # by doing so, we could Use source_map as the reference to rename model parameters
    converter.source_map = _debug_rename(graph, use_parser_friendly_name)
    # get mapping form debugName to layer_name
    _parse_graph(graph, script_module, converter.output_name_to_node_name, input_model_path, converter.torch_source_map)
    param_vars, tensors, packed_param_map, param_debug_name_map = convert_params(
        graph, params, converter.source_map, converter.torch_source_map, use_parser_friendly_name
    )
    ### [End QTI Change]

    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    outputs.update(param_vars)

    # For quantized models
    quantized_ops = set(["aten::quantize_per_tensor", "quantized::linear_dynamic"])
    if len(quantized_ops.intersection(set(op_names))) > 0:
        weight_quant_params = qnn_torch.get_weight_quant_params(script_module)
        qnn_torch.add_input_quant_params_to_op_inputs(graph)
        qnn_torch.add_quant_params_to_outputs(outputs, packed_param_map, weight_quant_params)
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        converter.update_convert_map(qnn_torch.convert_map)

    ### [QTI Change]
    operator_nodes = _get_operator_nodes(
        graph.nodes(),
        converter.source_map,
        converter.op_type_dict,
        use_parser_friendly_name,
    )
    ret_name = _get_input_names(graph.return_node())
    outputs = converter.convert_operators(operator_nodes, outputs, ret_name)

    # ListConstruct kept original python list. Convert to tuple.
    outputs = [_expr.Tuple(output) if isinstance(output, list) else output for output in outputs]

    if len(outputs) > 1:
        ret = _expr.Tuple(outputs)
    else:
        ret = outputs[0]
    ### [End QTI Change]

    # Separate data inputs and parameters to make sure data inputs are always in the beginning.
    func_args = []
    data_inputs = []
    for arg in _analysis.free_vars(ret):
        if arg.name_hint not in tvm_params.keys():
            data_inputs.append(arg)
        else:
            func_args.append(arg)
    func_args = data_inputs + func_args

    mod["main"] = tvm.relay.Function(func_args, ret)

    ### [QTI Change]
    if export_renamed_c_graph_path:
        export_c_graph(export_renamed_c_graph_path, graph)
    ### [End QTI Change]

    return transform.RemoveUnusedFunctions()(mod), tvm_params
