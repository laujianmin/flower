# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower type definitions."""


from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]
NDArrays = list[NDArray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str,dict]  # 增加一个dict用来diy 主要是每层重要参数位置
Value = Union[
    bool,
    bytes,
    float,
    int,
    str,
    list[bool],
    list[bytes],
    list[float],
    list[int],
    list[str],
]

# Value types for common.MetricsRecord
MetricsScalar = Union[int, float]
MetricsScalarList = Union[list[int], list[float]]
MetricsRecordValues = Union[MetricsScalar, MetricsScalarList]
# Value types for common.ConfigsRecord
ConfigsScalar = Union[MetricsScalar, str, bytes, bool, dict]
ConfigsScalarList = Union[MetricsScalarList, list[str], list[bytes], list[bool],list[dict]] # 增加一个dict用来diy 
ConfigsRecordValues = Union[ConfigsScalar, ConfigsScalarList]

Metrics = dict[str, Scalar]
MetricsAggregationFn = Callable[[list[tuple[int, Metrics]]], Metrics]

Config = dict[str, Scalar]
Properties = dict[str, Scalar]

# Value type for user configs
UserConfigValue = Union[bool, float, int, str]
UserConfig = dict[str, UserConfigValue]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


class ClientAppOutputCode(Enum):
    """ClientAppIO status codes."""

    SUCCESS = 0
    DEADLINE_EXCEEDED = 1
    UNKNOWN_ERROR = 2


@dataclass
class ClientAppOutputStatus:
    """ClientAppIO status."""

    code: ClientAppOutputCode
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: list[bytes]
    tensor_type: str


@dataclass
class GetParametersIns:
    """Parameters request for a client."""

    config: Config


@dataclass
class GetParametersRes:
    """Response when asked to return parameters."""

    status: Status
    parameters: Parameters



# Todo: 主要修改交互内容的格式
@dataclass
class FitIns:
    """Fit instructions for a client."""
    # Todo: 在客户端上传和下载时，注意核对数据格式。
    parameters: Parameters
    ckks_blocks : Union[list[bytes] ,None] # 密文块列表（fl），空（neo阶段）
    config: dict[str, Scalar]
    enc_lines: Union[list,None]  # 加密的列位置。 list-每一层   list[0]=[1,2,3,4]加密1234行 


@dataclass
class FitRes:
    """Fit response from a client."""
    # 客户端上传的数据，B正常，A加密一部分。
    status: Status
    parameters: Union[Parameters,None] # 
    ckks_blocks : Union[list[bytes] ,None] # 密文块列表（fl），空（neo阶段）
    num_examples: int
    metrics: dict[str, Scalar]

@dataclass
class FitResNeo:
    """协商阶段，客户端上传的数据格式"""
    he_budget: int
    Sens_layer: dict[str,list[dict]]
    # str: layer_name  dict[int,float]是当前层的重要列，与对应的重要性得分。
    # he_budget的值，就是len(dict[int,float]) 每层的加密预算相同。
# @dataclass
# class FitIns:
#     """Fit instructions for a client."""

#     parameters: Parameters
#     config: dict[str, Scalar]


# @dataclass
# class FitRes:
#     """Fit response from a client."""

#     status: Status
#     parameters: Parameters
#     num_examples: int
#     metrics: dict[str, Scalar]


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    status: Status
    loss: float
    num_examples: int
    metrics: dict[str, Scalar]


@dataclass
class GetPropertiesIns:
    """Properties request for a client."""

    config: Config


@dataclass
class GetPropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties


@dataclass
class ReconnectIns:
    """ReconnectIns message from server to client."""

    seconds: Optional[int]


@dataclass
class DisconnectRes:
    """DisconnectRes message from client to server."""

    reason: str


@dataclass
class ServerMessage:
    """ServerMessage is a container used to hold one instruction message."""

    get_properties_ins: Optional[GetPropertiesIns] = None
    get_parameters_ins: Optional[GetParametersIns] = None
    fit_ins: Optional[FitIns] = None
    evaluate_ins: Optional[EvaluateIns] = None


@dataclass
class ClientMessage:
    """ClientMessage is a container used to hold one result message."""

    get_properties_res: Optional[GetPropertiesRes] = None
    get_parameters_res: Optional[GetParametersRes] = None
    fit_res: Optional[FitRes] = None
    evaluate_res: Optional[EvaluateRes] = None


@dataclass
class RunStatus:
    """Run status information."""

    status: str
    sub_status: str
    details: str


@dataclass
class Run:  # pylint: disable=too-many-instance-attributes
    """Run details."""

    run_id: int
    fab_id: str
    fab_version: str
    fab_hash: str
    override_config: UserConfig
    pending_at: str
    starting_at: str
    running_at: str
    finished_at: str
    status: RunStatus

    @classmethod
    def create_empty(cls, run_id: int) -> "Run":
        """Return an empty Run instance."""
        return cls(
            run_id=run_id,
            fab_id="",
            fab_version="",
            fab_hash="",
            override_config={},
            pending_at="",
            starting_at="",
            running_at="",
            finished_at="",
            status=RunStatus(status="", sub_status="", details=""),
        )


@dataclass
class Fab:
    """Fab file representation."""

    hash_str: str
    content: bytes


class RunNotRunningException(BaseException):
    """Raised when a run is not running."""


class InvalidRunStatusException(BaseException):
    """Raised when an RPC is invalidated by the RunStatus."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
