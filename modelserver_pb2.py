# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modelserver.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11modelserver.proto\" \n\x0fSetCoefsRequest\x12\r\n\x05\x63oefs\x18\x01 \x03(\x02\"!\n\x10SetCoefsResponse\x12\r\n\x05\x65rror\x18\x01 \x01(\t\"\x1b\n\x0ePredictRequest\x12\t\n\x01X\x18\x01 \x03(\x02\"8\n\x0fPredictResponse\x12\t\n\x01y\x18\x01 \x01(\x02\x12\x0b\n\x03hit\x18\x02 \x01(\x08\x12\r\n\x05\x65rror\x18\x03 \x01(\t2p\n\x0bModelServer\x12\x31\n\x08SetCoefs\x12\x10.SetCoefsRequest\x1a\x11.SetCoefsResponse\"\x00\x12.\n\x07Predict\x12\x0f.PredictRequest\x1a\x10.PredictResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'modelserver_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_SETCOEFSREQUEST']._serialized_start=21
  _globals['_SETCOEFSREQUEST']._serialized_end=53
  _globals['_SETCOEFSRESPONSE']._serialized_start=55
  _globals['_SETCOEFSRESPONSE']._serialized_end=88
  _globals['_PREDICTREQUEST']._serialized_start=90
  _globals['_PREDICTREQUEST']._serialized_end=117
  _globals['_PREDICTRESPONSE']._serialized_start=119
  _globals['_PREDICTRESPONSE']._serialized_end=175
  _globals['_MODELSERVER']._serialized_start=177
  _globals['_MODELSERVER']._serialized_end=289
# @@protoc_insertion_point(module_scope)
