# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: node_service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12node_service.proto\x12\x0cnode_service\"?\n\rPromptRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x13\n\x06target\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\t\n\x07_target\"b\n\rTensorRequest\x12\x13\n\x0btensor_data\x18\x01 \x01(\x0c\x12\r\n\x05shape\x18\x02 \x03(\x05\x12\r\n\x05\x64type\x18\x03 \x01(\t\x12\x13\n\x06target\x18\x04 \x01(\tH\x00\x88\x01\x01\x42\t\n\x07_target\"%\n\x11ResetShardRequest\x12\x10\n\x08shard_id\x18\x01 \x01(\t\"\x07\n\x05\x45mpty2\xd7\x01\n\x0bNodeService\x12@\n\nSendPrompt\x12\x1b.node_service.PromptRequest\x1a\x13.node_service.Empty\"\x00\x12@\n\nSendTensor\x12\x1b.node_service.TensorRequest\x1a\x13.node_service.Empty\"\x00\x12\x44\n\nResetShard\x12\x1f.node_service.ResetShardRequest\x1a\x13.node_service.Empty\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'node_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PROMPTREQUEST']._serialized_start=36
  _globals['_PROMPTREQUEST']._serialized_end=99
  _globals['_TENSORREQUEST']._serialized_start=101
  _globals['_TENSORREQUEST']._serialized_end=199
  _globals['_RESETSHARDREQUEST']._serialized_start=201
  _globals['_RESETSHARDREQUEST']._serialized_end=238
  _globals['_EMPTY']._serialized_start=240
  _globals['_EMPTY']._serialized_end=247
  _globals['_NODESERVICE']._serialized_start=250
  _globals['_NODESERVICE']._serialized_end=465
# @@protoc_insertion_point(module_scope)
