# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: !!! Switch to using Kaolin version

import collections
import json
import logging
import numpy as np
import torch
from enum import IntEnum


logger = logging.getLogger(__name__)

class BinaryIoDataType(IntEnum):
    INT8 = 0
    UINT8 = 1
    INT16 = 2
    INT32 = 3
    UINT32 = 4
    INT64 = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    STRING = 9
    DICT = 10
    LIST = 11
    UNSUPPORTED = 100


__np_type_mappings =  [(BinaryIoDataType.INT8, np.dtype(np.int8)),
                       (BinaryIoDataType.UINT8, np.dtype(np.uint8)),
                       (BinaryIoDataType.INT16, np.dtype(np.int16)),
                       (BinaryIoDataType.INT32, np.dtype(np.int32)),
                       (BinaryIoDataType.UINT32, np.dtype(np.uint32)),
                       (BinaryIoDataType.INT64, np.dtype(np.int64)),
                       (BinaryIoDataType.FLOAT16, np.dtype(np.float16)),
                       (BinaryIoDataType.FLOAT32, np.dtype(np.float32)),
                       (BinaryIoDataType.FLOAT64, np.dtype(np.float64))]
__np_type_to_bytes = {np.dtype(np.int8): 1,
                      np.dtype(np.uint8): 1,
                      np.dtype(np.int16): 2,
                      np.dtype(np.int32): 4,
                      np.dtype(np.uint32): 4,
                      np.dtype(np.int64): 8,
                      np.dtype(np.float16): 2,
                      np.dtype(np.float32): 4,
                      np.dtype(np.float64): 8}
__io_data_type_to_np = dict(__np_type_mappings)
__np_to_io_data_type = dict([(x[1], x[0]) for x in __np_type_mappings])

MESSAGE_TAG_KEY = 'tag'
MESSAGE_CONTENT_KEY = 'msg'


def encode_message(tag, content, binary=True):
    msg = {MESSAGE_TAG_KEY: tag, MESSAGE_CONTENT_KEY: content}
    if binary:
        return to_binary(msg)
    else:
        return json.dumps(msg)


def to_binary(value):
    return value_to_binary(value, 0)


def from_binary(bytes_msg: bytes):
    res, read_bytes = value_from_binary(bytes_msg, 0)
    if read_bytes != len(bytes_msg):
        logger.warning(f'Read {read_bytes}, not full message length {len(bytes_msg)}')
    return res


def np_type_from_type_id(type_id):
    """
    Returns NumPy dtype and bytes per element for given type ID.

    Args:
        type_id: Binary I/O data type ID

    Returns:
        tuple: (numpy_dtype, bytes_per_element) or (None, 0) if unknown
    """
    np_type = __io_data_type_to_np.get(type_id, None)
    if np_type is not None:
        num_bytes = __np_type_to_bytes.get(np_type, 0)
    else:
        num_bytes = 0

    return np_type, num_bytes


def value_to_type(converted_value):
    if isinstance(converted_value, str):
        return BinaryIoDataType.STRING
    elif isinstance(converted_value, collections.abc.Mapping):
        return BinaryIoDataType.DICT
    elif isinstance(converted_value, list):
        return BinaryIoDataType.LIST
    elif isinstance(converted_value, np.ndarray):
        return __np_to_io_data_type.get(converted_value.dtype, BinaryIoDataType.UNSUPPORTED)
    else:
        return BinaryIoDataType.UNSUPPORTED


def convert_value_to_supported_format(value):
    if isinstance(value, str):
        return value
    elif torch.is_tensor(value):
        return value.detach().cpu().numpy()
    elif isinstance(value, collections.abc.Mapping):
        return value
    elif isinstance(value, np.ndarray):
        # TODO: possibly convert type
        return value
    elif isinstance(value, list):
        if len(value) == 0:
            return value  # Keep empty lists as lists
        
        # Check if all elements are numbers (int or float)
        all_numbers = all(isinstance(item, (int, float, bool)) for item in value)
        
        if all_numbers:
            # Check if all are integers
            all_integers = all(isinstance(item, int) or isinstance(item, bool) for item in value)
            if all_integers:
                return np.array(value, dtype=np.int32)
            else:
                return np.array(value, dtype=np.float32)
        else:
            # Mixed types - keep as list for LIST support
            return value
    elif isinstance(value, int):
        return np.array([value], dtype=np.int32)
    elif isinstance(value, float):
        return np.array([value], dtype=np.float32)
    else:
        raise ValueError(f'Cannot encode value of type {type(value)} to binary')


def gap_until_offset_n(current_offset: int, n: int) -> int:
    """
    Calculate the gap needed to align current_offset to a multiple of n.
    Arrays such as Int32Array cannot start at offsets that are not
    a multiple of 4. This helps us find the right offset.

    Args:
        current_offset: Current byte offset
        n: Alignment requirement (e.g., 4 for 4-byte alignment)

    Returns:
        Number of bytes to add to reach proper alignment
    """
    return (n - (current_offset % n)) % n


def gap_until_offset_4(current_offset: int) -> int:
    return gap_until_offset_n(current_offset, 4)


def string_from_binary(bytes_msg, offset, byte_length):
    """
    Decodes UTF-8 string of specified length from binary data.

    Args:
        bytes_msg: Input binary data (bytes)
        offset: Byte offset in data
        byte_length: String length in bytes

    Returns:
        Decoded string
    """
    if byte_length == 0:
        return ''
    string_bytes = bytes_msg[offset:offset + byte_length]
    return string_bytes.decode('utf-8')


def string_to_binary(string):
    """
    Encodes string to binary data using UTF-8 encoding.

    Args:
        string: String to encode

    Returns:
        bytes containing UTF-8 encoded bytes
    """
    return string.encode('utf-8')


def typed_value_from_binary(bytes_msg, offset, length, type_code):
    # Length is byte length for strings, but num elements for other types
    if type_code == BinaryIoDataType.STRING:
        return string_from_binary(bytes_msg, offset, length), length
    elif type_code == BinaryIoDataType.DICT:
        value, read_bytes = _dict_from_binary(bytes_msg, length=length, offset=offset)
        return value, read_bytes
    elif type_code == BinaryIoDataType.LIST:
        value, read_bytes = _list_from_binary(bytes_msg, length=length, offset=offset)
        return value, read_bytes
    else:
        np_type, bytes_per_element = np_type_from_type_id(type_code)
        if np_type is None:
            return None, 0
        read_bytes = gap_until_offset_n(offset, bytes_per_element)
        value = np.frombuffer(bytes_msg, dtype=np_type, count=length, offset=offset + read_bytes)
        read_bytes += length * bytes_per_element
        return value, read_bytes


def value_from_binary(bytes_msg, offset):
    read_bytes = gap_until_offset_4(offset)

    metadata_length = 2
    metadata = np.frombuffer(bytes_msg, dtype=np.int32, count=metadata_length, offset=offset + read_bytes)
    read_bytes += metadata_length * 4
    shape_length = metadata[0]
    type_code = metadata[1]

    is_primitive = False
    if shape_length > 0:
        shape = np.frombuffer(bytes_msg, dtype=np.int32, count=shape_length, offset=offset + read_bytes)
        read_bytes += shape_length * 4
        length = np.prod(shape)
    else:
        length = 1
        is_primitive = True
    value, value_read_bytes = typed_value_from_binary(bytes_msg, offset + read_bytes, length, type_code)
    read_bytes += value_read_bytes
    if isinstance(value, np.ndarray):
        if is_primitive:
            value = value[0].item()
        else:
            try:
                # TODO: this array is not writable; figure out what the behavior should be
                value = torch.from_numpy(value).reshape([x for x in shape])  # return in torch, as that is Kaolin i/o convention
            except ValueError as e:
                logger.error(f'Decoded shape does not match value size {e}')
    return value, read_bytes


def value_to_binary(in_value, initial_offset=0):
    is_primitive_number = isinstance(in_value, int) or isinstance(in_value, float)
    value = convert_value_to_supported_format(in_value)

    type_code = value_to_type(value)
    if type_code == BinaryIoDataType.UNSUPPORTED:
        raise ValueError(f'Cannot encode value of type {type(value)}')

    # Insert alignment
    result = bytes(gap_until_offset_4(initial_offset))

    if type_code == BinaryIoDataType.STRING:
        encoded_value = string_to_binary(value)
        shape = np.array([len(encoded_value)], dtype=np.int32)
        bytes_per_elem = 1
    elif type_code == BinaryIoDataType.DICT:
        shape = np.array([len(value)], dtype=np.int32)
        encoded_value = _dict_to_binary(value, initial_offset=initial_offset + len(result) + 3 * 4)
        bytes_per_elem = 1  # alignment is already accounted for
    elif type_code == BinaryIoDataType.LIST:
        shape = np.array([len(value)], dtype=np.int32)
        encoded_value = _list_to_binary(value, initial_offset=initial_offset + len(result) + 3 * 4)
        bytes_per_elem = 1  # alignment is already accounted for
    else:
        encoded_value = value.tobytes()
        # set shape len to 0 for primitives
        shape = np.array([] if is_primitive_number else value.shape, dtype=np.int32)
        bytes_per_elem = __np_type_to_bytes.get(value.dtype, 1)

    # Encode metadata and shape
    result += np.array([len(shape), type_code], dtype=np.int32).tobytes() + shape.tobytes()

    # Insert alignment
    extra_bytes = gap_until_offset_n(len(result) + initial_offset, bytes_per_elem)
    result += bytes(extra_bytes)
    result += encoded_value
    return result


def named_value_from_binary(bytes_msg, offset):
    read_bytes = gap_until_offset_4(offset)
    name_length = np.frombuffer(bytes_msg, dtype=np.int32, count=1, offset=offset + read_bytes)[0]  # in bytes
    read_bytes += 4
    name = string_from_binary(bytes_msg, offset + read_bytes, name_length)
    read_bytes += name_length
    value, value_read_bytes = value_from_binary(bytes_msg, offset + read_bytes)
    read_bytes += value_read_bytes
    return name, value, read_bytes


def named_value_to_binary(name, value, initial_offset=0):
    # We assume offset is appropriate for int32
    bin_str = string_to_binary(name)
    result = bytes(gap_until_offset_4(initial_offset))
    result += int32_to_binary(len(bin_str))
    result += bin_str
    result += value_to_binary(value, initial_offset=initial_offset + len(result))
    return result


def _dict_from_binary(bytes_msg, length, offset=0):
    """Converts bytes message to dictionary.
    Must be compatible with: nvidia.Controller.prototype.encodeDrawingRequest.

    Args:
        @param bytes_msg: raw bytes to decode
        @param offset: start read offset in bytes
        @param length: number of key-value pairs

    Return:
        metadata dict, total_read_bytes
    """
    total_read_bytes = 0
    res = {}
    for i in range(length):
        name, value, read_bytes = named_value_from_binary(bytes_msg, offset + total_read_bytes)
        res[name] = value
        total_read_bytes += read_bytes

    return res, total_read_bytes


def _list_from_binary(bytes_msg, length, offset=0):
    """Converts bytes message to list.

    Args:
        @param bytes_msg: raw bytes to decode
        @param offset: start read offset in bytes
        @param length: number of elements

    Return:
        list, total_read_bytes
    """
    total_read_bytes = 0
    res = []
    for i in range(length):
        value, read_bytes = value_from_binary(bytes_msg, offset + total_read_bytes)
        res.append(value)
        total_read_bytes += read_bytes

    return res, total_read_bytes



def int32_to_binary(single_int):
    return np.array([single_int], dtype=np.int32).tobytes()


def _dict_to_binary(in_dict, initial_offset=0):
    result = bytes()

    for name, value in in_dict.items():
        result += named_value_to_binary(name, value, initial_offset=initial_offset + len(result))
    return result


def _list_to_binary(in_list, initial_offset=0):
    result = bytes()

    for value in in_list:
        result += value_to_binary(value, initial_offset=initial_offset + len(result))
    return result

def split_tensor(tensor, max_chunk_bytes=1_000_000):
    """
    Returns a list of smaller tensors, each <= max_chunk_bytes.
    """
    bytes_per_elem = tensor.element_size()        # e.g., float32 = 4 bytes
    total_elems = tensor.numel()
    elems_per_chunk = max_chunk_bytes // bytes_per_elem
    elems_per_chunk = max(1, elems_per_chunk)

    flat = tensor.contiguous().view(-1)

    chunks = [
        flat[i:i+elems_per_chunk].clone()
        for i in range(0, total_elems, elems_per_chunk)
    ]

    return chunks


#: Default chunk budget in bytes for splitting large pixel payloads.
#: MUST match ``protocol.DEFAULT_CHUNK_BYTES`` and the add-on's copy of this
#: module. This was 10_000_00 (1 MB) here while the add-on used 10_000_000
#: (10 MB) -- a dropped zero that split every 4K texture into ~269 websocket
#: frames instead of 27.
DEFAULT_CHUNK_BYTES = 10_000_000

#: Wire dtype tags for pixel payloads.
DTYPE_UINT8 = "uint8"
DTYPE_FLOAT32 = "float32"


def encode_pixels(image, dtype=DTYPE_UINT8):
    """Quantize a ``[0, 1]`` pixel tensor for transport.

    Quantization happens on-device before the host copy, so a 4K RGBA texture
    crosses the PCIe bus as 64 MB rather than 256 MB, and again as 64 MB on the
    wire. uint8 is lossless for the 8-bit basecolor data both ends store.

    Args:
        image: Torch tensor or NumPy array of pixels, nominally in ``[0, 1]``.
        dtype: One of ``DTYPE_UINT8`` / ``DTYPE_FLOAT32``.

    Returns:
        tuple[np.ndarray, str]: The array to put on the wire and its dtype tag.
    """
    if isinstance(image, torch.Tensor):
        if dtype == DTYPE_UINT8:
            quantized = (image.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            return quantized.cpu().numpy(), DTYPE_UINT8
        return image.detach().to(torch.float32).cpu().numpy(), DTYPE_FLOAT32

    image = np.asarray(image)
    if dtype == DTYPE_UINT8:
        return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), DTYPE_UINT8
    return image.astype(np.float32), DTYPE_FLOAT32


def decode_pixels(image, dtype=DTYPE_FLOAT32):
    """Inverse of :func:`encode_pixels`; returns a float32 tensor in ``[0, 1]``."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if dtype == DTYPE_UINT8 or image.dtype == torch.uint8:
        return image.to(torch.float32) / 255.0
    return image.to(torch.float32)


def send_large_image(name: str, task_name: str, image, chunk_size: int = DEFAULT_CHUNK_BYTES,
                     msg_type: str = "image", dtype: str = DTYPE_UINT8, extra: dict = None):

    """
    Send a large image through WebSocket in multiple binary chunks.

    Parameters:
        name        : image name string
        task_name   : server task identifier stored in each message
        image       : pixel tensor/array in [0, 1]
        chunk_size  : size in bytes (default: 10MB)
        msg_type    : routing tag written into every chunk's ``type`` field
        dtype       : wire dtype; see :func:`encode_pixels`
        extra       : additional key/values copied into every chunk message
    """
    payload, dtype_tag = encode_pixels(image, dtype=dtype)
    chunks = split_tensor(torch.from_numpy(payload), max_chunk_bytes=chunk_size)
    total_chunks = len(chunks)
    msgs = []
    for idx in range(total_chunks):
        chunk = chunks[idx]
        msg = {
            "type": msg_type,
            "name": name,
            "task_name": task_name,
            "chunk_index": idx,
            "chunk_total": total_chunks,
            "dtype": dtype_tag,
            "image": chunk,
        }
        if extra:
            msg.update(extra)
        msgs.append(msg)
    return msgs
# inference_response = {'new_texture': my_tensor}
# bin_str = to_binary(inference_response)
# # In blender:
# res = from_binary(bin_message)
# res['new_texture']
