"""Super initial osf typings, too rough yet ..."""

from typing import Any, ClassVar, List

from typing import (overload, Iterator)
import numpy

from ouster.client import BufferT, LidarScan, SensorInfo


class ChunkRef:
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, arg0: int) -> MessageRef: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    @property
    def end_ts(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def start_ts(self) -> int: ...
    @property
    def valid(self) -> bool: ...


class ChunksLayout:
    __members__: ClassVar[dict] = ...  # read-only
    STANDARD: ClassVar[ChunksLayout] = ...
    STREAMING: ClassVar[ChunksLayout] = ...
    def __init__(self, arg0: int) -> None: ...
    @classmethod
    def from_string(cls, arg0: str) -> ChunksLayout: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: tuple) -> None: ...


class LidarScanStreamMeta:
    type_id: ClassVar[str] = ...  # read-only
    @property
    def sensor_meta_id(self) -> int: ...
    @property
    def field_types(self) -> Any: ...


class LidarScanStream:
    type_id: ClassVar[str] = ...  # read-only
    def __init__(self, writer, sensor_meta_id: int, field_types = ...) -> None: ...
    def save(self, ts: int, ls) -> None: ...


class LidarSensor(MetadataEntry):
    type_id: ClassVar[str] = ...  # read-only

    @overload
    def __init__(self, arg0: SensorInfo) -> None: ...
    @overload
    def __init__(self, metadata_json: str) -> None: ...
    @property
    def info(self) -> Any: ...
    @property
    def metadata(self) -> str: ...

class Extrinsics(MetadataEntry):
    type_id: ClassVar[str] = ...  # read-only
    def __init__(self, extrinsics: numpy.ndarray, ref_meta_id: int = ..., name: str = ...) -> None: ...
    @property
    def extrinsics(self) -> numpy.ndarray: ...
    @property
    def ref_meta_id(self) -> int: ...
    @property
    def name(self) -> str: ...


class MessageRef:
    def __init__(self, *args, **kwargs) -> None: ...
    def decode(self) -> object: ...
    def of(self, arg0: object) -> bool: ...
    @property
    def id(self) -> int: ...
    @property
    def ts(self) -> int: ...
    @property
    def buffer(self) -> BufferT: ...



class MetadataEntry:
    def __init__(self) -> None: ...
    @classmethod
    def from_buffer(cls, arg0: List[int], arg1: str) -> MetadataEntry: ...
    def of(self, arg0: object) -> bool: ...
    @property
    def buffer(self) -> List[int]: ...
    @property
    def id(self) -> int: ...
    @property
    def static_type(self) -> str: ...
    @property
    def type(self) -> str: ...


class MetadataStore:
    def __init__(self) -> None: ...
    def find(self, *args, **kwargs) -> Any: ...
    def get(self, *args, **kwargs) -> Any: ...
    def items(self) -> Iterator: ...
    def __getitem__(self, index) -> Any: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...


class Reader:
    def __init__(self, arg0: str) -> None: ...
    def chunks(self) -> Iterator: ...
    def messages_standard(self) -> Iterator: ...
    @overload
    def messages(self) -> Iterator: ...
    @overload
    def messages(self, start_ts: int, end_ts: int) -> Iterator: ...
    @overload
    def messages(self, stream_ids: List[int]) -> Iterator: ...
    @overload
    def messages(self, stream_ids: List[int], start_ts: int, end_ts: int) -> Iterator: ...
    @property
    def end_ts(self) -> int: ...
    @property
    def id(self) -> str: ...
    @property
    def meta_store(self) -> Any: ...
    @property
    def start_ts(self) -> int: ...
    @property
    def has_stream_info(self) -> bool: ...
    @property
    def has_message_idx(self) -> bool: ...
    def ts_by_message_idx(self, stream_id: int, msg_idx: int) -> int: ...


class StreamStats:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def end_ts(self) -> int: ...
    @property
    def message_avg_size(self) -> int: ...
    @property
    def message_count(self) -> int: ...
    @property
    def start_ts(self) -> int: ...
    @property
    def stream_id(self) -> int: ...


class StreamingInfo(MetadataEntry):
    type_id: ClassVar[str] = ...  # read-only
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def chunks_info(self) -> Iterator: ...
    @property
    def stream_stats(self) -> Iterator: ...


class Writer:
    @overload
    def __init__(self, file_name: str) -> None: ...

    @overload
    def __init__(self, file_name: str, metadata_id: str,
                 chunk_size: int = ...) -> None: ...

    def addMetadata(self, arg0: object) -> int: ...
    def saveMessage(self, stream_id: int, ts: int, buffer: BufferT) -> int: ...
    def close(self) -> None: ...
    @property
    def filename(self) -> str: ...
    @property
    def meta_store(self) -> MetadataStore: ...

class WriterV2:
    @overload
    def __init__(self, filename: str, info: SensorInfo, chunk_size: int) -> None: ...

    @overload
    def __init__(self, filename: str, info: List[SensorInfo], chunk_size: int) -> None: ...

    @overload
    def save(self, stream_id: int, scan: LidarScan) -> None: ...

    @overload
    def save(self, stream_id: int, scan: List[LidarScan]) -> None: ...

    @overload
    def get_sensor_info(self) -> List[SensorInfo]: ...

    @overload
    def get_sensor_info(self, stream_id: int) -> SensorInfo: ...

    def sensor_info_count(self) -> int: ...

    def get_filename(self) -> str: ...

    def get_chunk_size(self) -> int: ...

    def close(self) -> None: ...

    def is_closed(self) -> bool: ...

    def __enter__(self) -> WriterV2: ...

    def __exit__(*args) -> None: ...

    
    
def slice_and_cast(lidar_scan: LidarScan, field_types = ...) -> LidarScan: ...

def init_logger(log_level: str,
                log_file_path: str = ...,
                rotating: bool = ...,
                max_size_in_bytes: int = ...,
                max_files: int = ...) -> bool:
    ...

def dump_metadata(file: str, full: bool = ...) -> str: ...
def parse_and_print(file: str, with_decoding: bool = ...) -> None: ...

def backup_osf_file_metablob(file: str, backup_file_name: str) -> None: ...
def restore_osf_file_metablob(file: str, backup_file_name: str) -> None: ...
def osf_file_modify_metadata(file: str, new_metadata: List[SensorInfo]) -> int: ...

def pcap_to_osf(file: str, meta: str, lidar_port: int, osf_filename: str,
                chunks_layout: str = ..., chunk_size: int = ...) -> bool: ...
