import grpc
import numpy as np
from typing import Optional, Tuple, List, Union

# These would be generated from the .proto file
from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle
from exo.inference.shard import Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities
import json
import mlx.core as mx

class GRPCPeerHandle(PeerHandle):
  def __init__(self, _id: str, address: str, device_capabilities: DeviceCapabilities):
    self._id = _id
    self.address = address
    self._device_capabilities = device_capabilities
    self.channel = None
    self.stub = None

  def id(self) -> str:
    return self._id

  def device_capabilities(self) -> DeviceCapabilities:
    return self._device_capabilities

  async def connect(self):
    self.channel = grpc.aio.insecure_channel(self.address, options=[("grpc.max_metadata_size", 32*1024*1024)])
    self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)

  async def is_connected(self) -> bool:
    return self.channel is not None and self.channel.get_state() == grpc.ChannelConnectivity.READY

  async def disconnect(self):
    if self.channel:
      await self.channel.close()
    self.channel = None
    self.stub = None

  async def send_prompt(self, shard: Shard, prompt: str, image_str: Optional[str] = None, request_id: Optional[str] = None, inference_state: Optional[str] = None) -> Optional[np.array]:
    if inference_state is not None:
      inference_state = json.dumps(inference_state)
    request = node_service_pb2.PromptRequest(
      prompt=prompt,
      image_str=image_str,
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      request_id=request_id,
      inference_state=inference_state,
    )
    response = await self.stub.SendPrompt(request)

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def send_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None, inference_state: Optional[dict] = None) -> Optional[np.array]:
    request = node_service_pb2.TensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=tensor.shape, dtype=str(tensor.dtype)),
      request_id=request_id,
      inference_state=self.serialize_inference_state(inference_state),
    )
    response = await self.stub.SendTensor(request)

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    request = node_service_pb2.GetInferenceResultRequest(request_id=request_id)
    response = await self.stub.GetInferenceResult(request)
    if response.tensor is None:
      return None, response.is_finished
    return (
      np.frombuffer(response.tensor.tensor_data, dtype=np.dtype(response.tensor.dtype)).reshape(response.tensor.shape),
      response.is_finished,
    )

  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    request = node_service_pb2.CollectTopologyRequest(visited=visited, max_depth=max_depth)
    response = await self.stub.CollectTopology(request)
    topology = Topology()
    for node_id, capabilities in response.nodes.items():
      device_capabilities = DeviceCapabilities(model=capabilities.model, chip=capabilities.chip, memory=capabilities.memory, flops=capabilities.flops)
      topology.update_node(node_id, device_capabilities)
    for node_id, peers in response.peer_graph.items():
      for peer_id in peers.peer_ids:
        topology.add_edge(node_id, peer_id)
    return topology

  async def send_result(self, request_id: str, result: Union[List[int], np.ndarray], is_finished: bool) -> None:
    tensor = None
    if isinstance(result, np.ndarray):
        tensor = node_service_pb2.Tensor(tensor_data=result.tobytes(), shape=result.shape, dtype=str(result.dtype))
        result = []
    request = node_service_pb2.SendResultRequest(request_id=request_id, result=result, tensor=tensor, is_finished=is_finished)
    await self.stub.SendResult(request)

  async def send_opaque_status(self, request_id: str, status: str) -> None:
    request = node_service_pb2.SendOpaqueStatusRequest(request_id=request_id, status=status)
    await self.stub.SendOpaqueStatus(request)

  def serialize_inference_state(self, inference_state: dict) -> node_service_pb2.InferenceState:
      proto_inference_state = node_service_pb2.InferenceState()
      other_data = {}
      for k, v in inference_state.items():
          if isinstance(v, mx.array):
              np_array = np.array(v)
              tensor_data = node_service_pb2.Tensor(
                  tensor_data=np_array.tobytes(),
                  shape=list(np_array.shape),
                  dtype=str(np_array.dtype)
              )
              proto_inference_state.tensor_data[k].CopyFrom(tensor_data)
          elif isinstance(v, list) and all(isinstance(item, mx.array) for item in v):
              tensor_list = node_service_pb2.TensorList()
              for tensor in v:
                  np_array = np.array(tensor)
                  tensor_data = node_service_pb2.Tensor(
                      tensor_data=np_array.tobytes(),
                      shape=list(np_array.shape),
                      dtype=str(np_array.dtype)
                  )
                  tensor_list.tensors.append(tensor_data)
              proto_inference_state.tensor_list_data[k].CopyFrom(tensor_list)
          else:
              # For non-tensor data, we'll still use JSON
              other_data[k] = v
      if other_data:
        proto_inference_state.other_data_json = json.dumps(other_data)
      return proto_inference_state
