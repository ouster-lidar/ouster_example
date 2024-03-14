from typing import cast, Iterator, Dict, Optional, List, Tuple, Union

from more_itertools import ilen
from ouster import client
from ouster.client import LidarScan, SensorInfo, first_valid_packet_ts
from ouster.client import ScanSource, MultiScanSource
import ouster.osf as osf

from ouster.sdkx.multi import collate_scans   # type: ignore
from ouster.sdkx.forward_slicer import ForwardSlicer
from ouster.osf._osf import MessageRef

from ouster.sdkx.util import progressbar    # type: ignore


class OsfScanSource(MultiScanSource):
    """Implements MultiScanSource protocol using OSF Reader with multiple sensors."""

    def __init__(
        self,
        file_path: str,
        *,
        dt: int = 10**8,
        complete: bool = False,
        index: bool = False,
        cycle: bool = False,
        **_
    ) -> None:
        """
        Args:
            file_path: OSF file path to open as a scan source
            dt: max time difference between scans in the collated scan (i.e.
                time period at which every new collated scan is released/cut),
                default is 0.1s
            complete: set to True to only release complete scans (not implemnted)
            index: if this flag is set to true and the underlying osf file was
                not indexed, in the case the file will be indexed inplace, otherwise
                the file will be left intact. If the file already had index builtin
                then this flag does nothing (default is False)
            cycle: repeat infinitely after iteration is finished (default is False)
        """

        # TODO: implement 'complete' flag for OSF
        if complete:
            print("OSF 'complete' flag requested but isn't implemented for OSF")

        self._reader = osf.Reader(file_path)

        if not self._reader.has_message_idx:
            if index:
                print("OSF file not indexed! re-indexing file inplace...")
                try:
                    self._reindex_osf_inplace(self._reader, file_path)
                except RuntimeError as e:
                    print(f"Failed re-indexing OSF file!\n more details: {e}")
                # TODO: should we proceed if we failed to re-index for any given reason
                self._reader = osf.Reader(file_path)
            else:
                print("OSF file not indexed, indexing not requested!")

        self._cycle = cycle
        self._dt = dt

        self._sensors = [(sid, sm) for sid, sm in self._reader.meta_store.find(
            osf.LidarSensor).items()]

        # map stream_id to metadata entry
        self._sensor_idx: Dict[int, int]
        self._sensor_idx = {
            sid: sidx
            for sidx, (sid, _) in enumerate(self._sensors)
        }

        # load stored extrinsics (if any)
        extrinsics = self._reader.meta_store.find(osf.Extrinsics)
        for _, v in extrinsics.items():
            if v.ref_meta_id in self._sensor_idx:
                sidx = self._sensor_idx[v.ref_meta_id]
                print(f"Found extrinsics for sensor[{sidx}]:\n",
                      v.extrinsics)
                self._sensors[sidx][1].info.extrinsic = v.extrinsics

        self._metadatas = [sm.info for _, sm in self._sensors]

        # map stream_id to metadata entry
        self._stream_sensor_idx: Dict[int, int]
        self._stream_sensor_idx = {}
        for stream_type in [osf.LidarScanStream]:
            for stream_id, stream_meta in self._reader.meta_store.find(
                    stream_type).items():
                self._stream_sensor_idx[stream_id] = self._sensor_idx[
                    stream_meta.sensor_meta_id]

        scan_streams = self._reader.meta_store.find(osf.LidarScanStream)
        self._stream_ids = [mid for mid, _ in scan_streams.items()]
        # TODO: the following two properties (_scans_num, _len) are computed on
        # load but should rather be provided directly through OSF API. Obtain
        # these values directly from OSF API once implemented.

        start_ts = self._reader.start_ts
        end_ts = self._reader.end_ts
        self._scans_num = [ilen(self._msgs_iter_stream(
            mid, start_ts, end_ts)) for mid in self._stream_ids]
        self._len = ilen(collate_scans(self._msgs_iter(
            self._stream_ids, start_ts, end_ts, False),
            self.sensors_count, lambda msg: msg.ts, dt=self._dt))

    def _osf_convert(self, reader: osf.Reader, output: str) -> None:
        # TODO: figure out how to get the current chunk_size
        chunk_size = 0
        writer = osf.Writer(output, reader.id, chunk_size)
        for _, m in reader.meta_store.items():
            if m.of(osf.StreamingInfo):
                # StreamingInfo is always generated by Writer automatically in
                # default STREAMING chunks layout, so we don't copy the original
                continue
            writer.addMetadata(m)
        # convert should be able to read OSF in STANDARD layout too
        msgs = reader.messages() if reader.has_stream_info else reader.messages_standard()
        msgs_count = ilen(msgs)
        msgs = reader.messages() if reader.has_stream_info else reader.messages_standard()
        for idx, msg in enumerate(msgs):
            writer.saveMessage(msg.id, msg.ts, msg.buffer)
            progressbar(idx, msgs_count, "", "indexed")
        print("\nfinished building index")
        writer.close()

    def _reindex_osf_inplace(self, reader, osf_file):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True) as f:
            self._osf_convert(reader, f.name)
            try:
                import shutil
                shutil.copy2(f.name, osf_file)
            except OSError as e:
                raise RuntimeError(f"Error overwriteing osf file: {osf_file}"
                                   f"\nmore details: {e}")

    def _msgs_iter_stream(self, stream_id: int, start_ts: int, stop_ts: int
                          ) -> Iterator[MessageRef]:
        for _, msg in self._msgs_iter([stream_id], start_ts, stop_ts, False):
            yield msg

    def _msgs_iter(self, stream_ids: List[int], start_ts: int, stop_ts: int, cycle: bool
                   ) -> Iterator[Tuple[int, MessageRef]]:
        while True:
            for msg in self._reader.messages(stream_ids, start_ts, stop_ts):
                if msg.of(osf.LidarScanStream):
                    sidx = self._stream_sensor_idx[msg.id]
                    yield sidx, msg
            if not cycle:
                break

    def _scans_iter(self, start_ts: int, stop_ts: int, cycle: bool
                    ) -> Iterator[Tuple[int, LidarScan]]:
        for idx, msg in self._msgs_iter(self._stream_ids, start_ts, stop_ts, cycle):
            ls = msg.decode()
            if ls:
                yield idx, cast(LidarScan, ls)

    @property
    def sensors_count(self) -> int:
        return len(self._stream_ids)

    @property
    def metadata(self) -> List[SensorInfo]:
        return self._metadatas

    @property
    def is_live(self) -> bool:
        return False

    @property
    def is_seekable(self) -> bool:
        return True

    @property
    def is_indexed(self) -> bool:
        return self._reader.has_message_idx

    @property
    def fields(self) -> List[client.FieldTypes]:
        """Field types are present in the LidarScan objects on read from iterator"""
        return [client.get_field_types(m) for m in self.metadata]

    @property
    def scans_num(self) -> List[int]:
        return self._scans_num

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[List[Optional[LidarScan]]]:
        msgs_itr = self._scans_iter(
            self._reader.start_ts, self._reader.end_ts, self._cycle)
        return collate_scans(msgs_itr, self.sensors_count, first_valid_packet_ts, dt=self._dt)

    def _seek(self, key: int) -> None:
        """seek/jump to a specific item within the list of LidarScan objects that this particular scan
        source has access to"""
        ...

    def __getitem__(self, key: Union[int, slice]
                    ) -> Union[List[Optional[LidarScan]], List[List[Optional[LidarScan]]]]:

        if not self.is_indexed:
            raise RuntimeError(
                "can not invoke __getitem__ on non-indexed source")

        msgs_itr: Iterator[Tuple[int, LidarScan]]

        if isinstance(key, int):
            L = len(self)
            if key < 0:
                key += L
            if key < 0 or key >= L:
                raise IndexError("index is out of range")
            ts = [self._reader.ts_by_message_idx(
                mid, key) for mid in self._stream_ids]
            ts_start = min(ts)
            ts_stop = min(ts_start + self._dt, max(ts))
            msgs_itr = self._scans_iter(ts_start, ts_stop, False)
            return next(collate_scans(msgs_itr, self.sensors_count,
                                      first_valid_packet_ts, dt=self._dt))

        if isinstance(key, slice):
            L = len(self)
            k = ForwardSlicer.normalize(key, L)
            count = k.stop - k.start
            if count <= 0:
                return []
            ts_start = min([self._reader.ts_by_message_idx(mid, k.start)
                           for mid in self._stream_ids])
            ts_stop = max([self._reader.ts_by_message_idx(mid, k.stop - 1)
                          for mid in self._stream_ids])
            msgs_itr = self._scans_iter(ts_start, ts_stop, False)
            result = [msg for idx, msg in ForwardSlicer.slice(
                enumerate(collate_scans(msgs_itr, self.sensors_count,
                                        first_valid_packet_ts,
                                        dt=self._dt)), k) if idx < count]
            return result if k.step > 0 else list(reversed(result))

        raise TypeError(
            f"indices must be integer or slice, not {type(key).__name__}")

    # TODO: implement
    def set_playback_speed(self, int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Close osf file."""
        # TODO[pb]: Need to add Reader.close() method, because now it's
        # all happens in dtor, which is not very clear by lifecycle.
        if self._reader:
            del self._reader
            self._reader = None     # type: ignore

    def __del__(self) -> None:
        """Automatic release of any underlying resource."""
        # self.close() # TODO: currently this causes an exception, avoid
        pass

    def single_source(self, stream_idx: int) -> ScanSource:
        from ouster.client.scan_source_adapter import ScanSourceAdapter
        return ScanSourceAdapter(self, stream_idx)
