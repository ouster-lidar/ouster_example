from contextlib import closing
from os import path
import socket
import sys

import numpy as np
import pytest

from ouster import client

pytest.register_assert_rewrite('ouster.client._digest')
import ouster.client._digest as digest  # noqa

DATA_DIR = path.join(path.dirname(path.abspath(__file__)), "data")


@pytest.fixture(scope="module")
def stream_digest():
    digest_path = path.join(DATA_DIR, "os-992011000121_digest.json")
    with open(digest_path, 'r') as f:
        return digest.StreamDigest.from_json(f.read())


@pytest.fixture(scope="module")
def meta(stream_digest: digest.StreamDigest):
    return stream_digest.meta


def test_sensor_init(meta: client.SensorInfo) -> None:
    """Initializing a data stream with metadata makes no network calls."""
    with closing(client.Sensor("os.invalid", metadata=meta)):
        pass


def test_sensor_timeout(meta: client.SensorInfo) -> None:
    """Setting a zero timeout reliably raises an exception."""
    with closing(client.Sensor("os.invalid", metadata=meta,
                               timeout=0.0)) as source:
        with pytest.raises(client.ClientTimeout):
            next(iter(source))


@pytest.mark.skipif(sys.platform == "win32",
                    reason="winsock is OK with this; not sure why")
def test_sensor_port_in_use(meta: client.SensorInfo) -> None:
    """Using an unavailable port will throw."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('localhost', 0))
    _, port = sock.getsockname()
    with closing(sock):
        with pytest.raises(RuntimeError):
            with closing(client.Sensor("os.invalid", port, metadata=meta)):
                pass

        with pytest.raises(RuntimeError):
            with closing(client.Sensor("os.invalid", 0, port, metadata=meta)):
                pass


@pytest.fixture(scope="module")
def packet(stream_digest):
    bin_path = path.join(DATA_DIR, "os-992011000121_data.bin")
    with open(bin_path, 'rb') as b:
        return next(iter(digest.LidarBufStream(b, stream_digest.meta)))


@pytest.fixture
def packets(stream_digest: digest.StreamDigest):
    bin_path = path.join(DATA_DIR, "os-992011000121_data.bin")
    with open(bin_path, 'rb') as b:
        yield digest.LidarBufStream(b, stream_digest.meta)


def test_scans_simple(packets: client.PacketSource) -> None:
    """Test that the test data contains exactly one scan."""
    scans = iter(client.Scans(packets))
    assert next(scans) is not None

    with pytest.raises(StopIteration):
        next(scans)


def test_scans_meta(packets: client.PacketSource) -> None:
    """Sanity check metadata and column headers of a batched scan."""
    scans = iter(client.Scans(packets))
    scan = next(scans)

    assert scan.frame_id != -1
    assert scan.h == packets.metadata.format.pixels_per_column
    assert scan.w == packets.metadata.format.columns_per_frame
    assert len(scan.headers) == scan.w

    assert not scan.complete, "test data should have missing packet!"

    # check that the scan is missing exactly one packet's worth of columns
    valid_columns = [h.status for h in scan.headers].count(0xffffffff)
    assert valid_columns == (packets.metadata.format.columns_per_frame -
                             packets.metadata.format.columns_per_packet)

    missing_ts = [h.timestamp for h in scan.headers].count(0)
    assert missing_ts == packets.metadata.format.columns_per_packet

    # extra zero encoder value for first column
    zero_enc = [h.encoder for h in scan.headers].count(0)
    assert zero_enc == packets.metadata.format.columns_per_packet + 1


def test_scans_first_packet(packet: client.LidarPacket,
                            packets: client.PacketSource) -> None:
    """Check that data in the first packet survives batching to a scan."""
    scans = iter(client.Scans(packets))
    scan = next(scans)

    h = packet._pf.pixels_per_column
    w = packet._pf.columns_per_packet

    assert np.array_equal(packet.view(client.ChanField.RANGE),
                          scan.field(client.ChanField.RANGE)[:h, :w])

    assert np.array_equal(packet.view(client.ChanField.REFLECTIVITY),
                          scan.field(client.ChanField.REFLECTIVITY)[:h, :w])

    assert np.array_equal(packet.view(client.ChanField.INTENSITY),
                          scan.field(client.ChanField.INTENSITY)[:h, :w])

    assert np.array_equal(packet.view(client.ChanField.AMBIENT),
                          scan.field(client.ChanField.AMBIENT)[:h, :w])

    assert np.all(packet.view(client.ColHeader.FRAME_ID) == scan.frame_id)

    assert np.array_equal(packet.view(client.ColHeader.TIMESTAMP),
                          [h.timestamp for h in scan.headers][:w])

    assert np.array_equal(packet.view(client.ColHeader.ENCODER_COUNT),
                          [h.encoder for h in scan.headers][:w])

    assert np.array_equal(packet.view(client.ColHeader.VALID),
                          [h.status for h in scan.headers][:w])


def test_scans_complete(packets: client.PacketSource) -> None:
    """Test built-in filtering for complete scans.

    The test dataset only contains a single incomplete frame. Check that
    specifying ``complete=True`` discards it.
    """
    scans = iter(client.Scans(packets, complete=True))

    with pytest.raises(StopIteration):
        next(scans)


def test_scans_timeout(packets: client.PacketSource) -> None:
    """A zero timeout should deterministically throw."""
    scans = iter(client.Scans(packets, timeout=0.0))

    with pytest.raises(client.ClientTimeout):
        next(scans)


def test_parse_and_batch_packets(stream_digest,
                                 packets: client.PacketSource) -> None:
    """Test that parsing packets produces expected results.

    Checks hashes of all packet and scans fields and headers against
    known-good values.
    """
    other = digest.StreamDigest.from_packets(packets)
    stream_digest.check(other)
