#  type: ignore
from datetime import datetime
import itertools
import os
import logging
from typing import Optional, List, Tuple

import numpy as np

import click
from more_itertools import time_limited, side_effect
from prettytable import PrettyTable, PLAIN_COLUMNS  # type: ignore
from textwrap import indent

from ouster import client
from ouster.sdk.util import resolve_metadata
from ouster.sdk.viz import SimpleViz
from ouster.sdkx import packet_iter
from ouster.sdkx.parsing import default_scan_fields
from ouster.sdkx.util import resolve_extrinsics
from ouster.sdkx.viz import ExtendedScanViz
from .util import (click_ro_file, import_rosbag_modules)

HAS_MULTI = False
try:
    from ouster.sdk.util import resolve_metadata_multi
    from ouster.sdkx.multi import PcapMulti, ScansMulti, collate_scans
    from ouster.sdkx.multi_viz import MultiLidarScanViz
    HAS_MULTI = True
except ImportError as e:
    logging.debug(e)


@click.group(name="pcap")
def pcap_group() -> None:
    """Commands for working with pcap files."""
    pass


@pcap_group.command(name='info')
@click.argument('file', required=True, type=click_ro_file)
@click.option('-n', type=int, default=0, help="Read only INTEGER packets.")
def pcap_info(file: str, n: int) -> None:
    """Print information about a pcap file to stdout."""
    try:
        import ouster.pcap as pcap
    except ImportError:
        raise click.ClickException("Please verify that libpcap is installed")

    # helper to get value from single-element sets
    def singleton_or(s, default):
        return next(iter(s)) if len(s) == 1 else default

    pcap_size = os.path.getsize(file)
    # read full pcap with progress bar
    all_infos = None
    with click.progressbar(length=pcap_size, label="Reading pcap:") as bar:
        def progress_callback(current, diff, total):
            bar.update(diff)
        all_infos = pcap._packet_info_stream(file, n, progress_callback, 100)
        bar.update(pcap_size)

    # format output
    table = PrettyTable()
    table.field_names = [
        'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'AF', 'Frag', 'Size',
        'Count'
    ]

    for k, v in all_infos.udp_streams.items():
        af = singleton_or([k for k, v in v.ip_version_counts.items()], '<MULTIPLE>')
        size = singleton_or([k for k, v in v.payload_size_counts.items()], '<MULTIPLE>')
        frag = 'No' if [k for k, v in v.fragment_counts.items()] else 'Yes'

        table.add_row([
            k.src_ip, k.dst_ip, k.src_port, k.dst_port, af, frag, size, v.count
        ])

    encap = {
        0: '<MULTIPLE>',
        1: 'ETHERNET',
        42: 'SLL'
    }.get(all_infos.encapsulation_protocol, 'UNKNOWN')
    min_datetime = datetime.fromtimestamp(all_infos.timestamp_min)
    max_datetime = datetime.fromtimestamp(all_infos.timestamp_max)
    duration = max_datetime - min_datetime

    click.echo(f"File size:     {pcap_size/(2**20):.2f}M")
    click.echo(f"Packets read:  {all_infos.total_packets}")
    click.echo(f"Encapsulation: {encap}")
    click.echo(f"Capture start: {min_datetime}")
    click.echo(f"Capture end:   {max_datetime}")
    click.echo(f"Duration:      {duration}")
    click.echo("UDP Streams:")
    table.set_style(PLAIN_COLUMNS)
    table.align = 'r'
    table.align['Src IP'] = 'l'  # type: ignore
    click.echo(indent(str(table), '  '))


@pcap_group.command(name="record")
@click.argument('hostname', required=True, type=str)
@click.option('-d',
              'dest',
              required=False,
              default=".",
              type=click.Path(exists=True, file_okay=False, writable=True),
              help="Directory to output files. Defaults to current dir")
@click.option('-l', '--lidar-port', default=7502, help="default: 7502")
@click.option('-i', '--imu-port', default=7503, help="default: 7503")
@click.option('-n', '--n-frames', type=int, help="number of lidar frames")
@click.option('-s', '--n-seconds', default=0.0, help="max time to record")
@click.option('--chunk-size', default=0, help="split output by size (MB)")
@click.option('-b', '--buf-size', default=640, help="Max packets to buffer")
@click.option('-t', '--timeout', default=1.0, help="Seconds to wait for data")
@click.option('-p', '--prefix', default="", help="Recorded file name prefix")
@click.option('--legacy/--non-legacy',
              default=False,
              help="Use legacy metadata format or not")
def pcap_record(hostname: str, dest, lidar_port: int, imu_port: int,
                n_frames: Optional[int], n_seconds: float, chunk_size: int,
                buf_size: int, timeout: float, prefix: str,
                legacy: bool) -> None:
    """Record lidar and IMU packets from a sensor to a pcap file.

    Note: this will currently not configure the sensor or query the sensor for
    the port to listen on. You will need to set the sensor port and distination
    settings separately.
    """
    try:
        import ouster.pcap as pcap
    except ImportError:
        raise click.ClickException("Please verify that libpcap is installed")

    message = "Recording"
    if n_frames:
        message += f" {n_frames} lidar frames"
    if n_seconds:
        message += f" for up to {n_seconds} seconds"
    else:
        message += ", hit ctrl-c to exit"

    click.echo(f"Connecting to {hostname}")
    source = client.Sensor(hostname,
                           lidar_port,
                           imu_port,
                           buf_size=buf_size,
                           timeout=timeout if timeout > 0 else None,
                           _legacy_format=legacy)

    # fancy automatic file naming
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = source.metadata
    base_name = f"{prefix}{meta.prod_line}_{meta.fw_rev}_{meta.mode}_{time}"
    meta_path = os.path.join(dest, base_name) + ".json"

    try:
        click.echo(f"Writing metadata to {meta_path}")
        source.write_metadata(meta_path)

        # apply packet limit and timeout
        packets = iter(source)
        if n_frames:
            packets = packet_iter.n_frames(packets, n_frames)
        if n_seconds:
            packets = time_limited(n_seconds, packets)

        # ensure tmp file exists for size calculation
        tmp_path = os.path.join(dest, base_name) + ".tmp"
        with open(tmp_path, 'w'):
            pass

        # decide how to chunk output (only file size currently supported)
        pred = lambda _: False
        if chunk_size:
            pred = lambda _: os.path.getsize(tmp_path) > chunk_size * 2**20
        chunks = packet_iter.ichunked_framed(packets, pred)

        # write a pcap per chunk
        click.echo(f"{message}...")
        for i, chunk in enumerate(chunks):
            try:
                pcap.record(chunk,
                            tmp_path,
                            lidar_port=lidar_port,
                            imu_port=imu_port)
            finally:
                if os.path.exists(tmp_path):
                    pcap_path = os.path.join(dest, base_name) + f"-{i:03}.pcap"
                    os.rename(tmp_path, pcap_path)
                    sz = os.path.getsize(pcap_path)
                    click.echo(f"Wrote {sz/2**20:.1f} MiB to {pcap_path}")

    except KeyboardInterrupt:
        click.echo("\nInterrupted")
    finally:
        source.close()


@pcap_group.command(name="viz")
@click.argument('file', required=True, type=click.Path(exists=True))
@click.option('-f', '--meta', required=False, type=click_ro_file)
@click.option('-c', '--cycle', is_flag=True, help="Loop playback")
@click.option('-l', '--lidar-port', default=None, help="Dest. port of lidar data")
@click.option('-i', '--imu-port', default=None, help="Dest. port of imu data")
@click.option('-F', '--filter', is_flag=True, help="Drop scans missing data")
@click.option('-b', '--buf', default=50, help="Scans to buffer for stepping.")
@click.option('-r',
              '--rate',
              default=1.0,
              help="Playback rate. One of 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0")
@click.option('--extrinsics',
              type=float,
              required=False,
              nargs=16,
              help='Lidar sensor extrinsics to use in viz')
@click.option('--soft-id-check',
              is_flag=True,
              help="Continue parsing lidar packets even if init_id/sn doesn't "
              "match with metadata")
@click.option("-p", "--pause", is_flag=True, help="Pause after the first scan")
@click.option("--pause-at",
              default=-1,
              help="Lidar Scan number to pause")
@click.option('--multi',
              is_flag=True,
              hidden=not HAS_MULTI,
              help='Turn on multi sensor pcap handling and metadata resolutions')
def pcap_viz(file: str, meta: Optional[str], cycle: bool,
             lidar_port: Optional[int], imu_port: Optional[int], filter: bool,
             buf: int, rate: float, extrinsics: Optional[List[float]],
             soft_id_check: bool, pause: bool, pause_at: int,
             multi: bool) -> None:
    """Visualize data from a pcap file.

    To correctly visualize a pcap containing multiple UDP streams, you must
    specify a destination port. All packets recorded with a different
    destination port will be filtered out.
    """

    try:
        import ouster.pcap as pcap
    except ImportError as e:
        raise click.ClickException(
            "Please verify that libpcap is installed. Error: " + str(e))

    if not HAS_MULTI and multi:
        raise click.ClickException("--multi is not supported in this version.")

    if pause and pause_at == -1:
        pause_at = 0

    if rate not in SimpleViz._playback_rates:
        raise click.ClickException("Invalid rate specified")

    try:

        if not multi:
            # Single sensor pcap handling

            # the only reason why we can't always use PcapMulti is that we still
            # want to pass custom lidar_port and imu_port from command line (for
            # now at least)
            # TODO[pb]: Decide when we can remove the custom lidar_port/imu_port
            #           params from command line and switch everything to a
            #           single PcapMulti source (it will simplify branching in
            #           pcap_viz)

            meta = resolve_metadata(file, meta)
            if not meta:
                raise click.ClickException(
                    "File not found, please specify a metadata file with `-f`")
            with open(meta) as json:
                click.echo(f"Reading metadata from: {meta}")
                info = client.SensorInfo(json.read())

            source = pcap.Pcap(file,
                               info,
                               lidar_port=lidar_port,
                               imu_port=imu_port,
                               _soft_id_check=soft_id_check)

            # Handle extrinsics, for single sensor source
            ext_found = False
            if extrinsics:
                source.metadata.extrinsic = np.array(extrinsics).reshape((4, 4))
            else:
                # Lookup for known extrinsics
                ext_results = resolve_extrinsics(data_path=file,
                                                 infos=[source.metadata])
                if ext_results and ext_results[0]:
                    source.metadata.extrinsic = ext_results[0][0]
                    ext_found = True

            if extrinsics or ext_found:
                print(f"Using sensor extrinsics:\n{source.metadata.extrinsic}")

            # enable parsing flags field
            field_types = default_scan_fields(
                source.metadata.format.udp_profile_lidar, flags=True)

            scans_source = client.Scans(source,
                                        fields=field_types,
                                        complete=filter)

            ls_viz = ExtendedScanViz(scans_source.metadata)

        elif HAS_MULTI and multi:
            # Multi sensor pcap handling

            metadata_paths = resolve_metadata_multi(file)
            if not metadata_paths:
                raise click.ClickException(
                    "Metadata jsons not found. Make sure that metadata json files "
                    "have common prefix with a PCAP file")

            source = PcapMulti(file,
                               metadata_paths=metadata_paths,
                               _soft_id_check=soft_id_check,
                               _resolve_extrinsics=True)

            # print extrinsics if any were found
            for ext_source, m in zip(source.extrinsics_source,
                                     source._metadata):
                if ext_source:
                    print(f"Found extrinsics for {m.sn} "
                          f"(from {ext_source}):\n{m.extrinsic}")

            # enable parsing flags field
            field_types = [
                default_scan_fields(m.format.udp_profile_lidar, flags=True)
                for m in source.metadata
            ]

            # set sensor names as idx in the source
            for idx, m in enumerate(source.metadata):
                source.metadata[idx].hostname = f"sensoridx: {idx}"

            ls_viz = MultiLidarScanViz(source.metadata, source_name=file)

            lidar_source = ScansMulti(source,
                                      fields=field_types,
                                      complete=filter)

            scans_source = collate_scans(lidar_source, use_unsynced=True)

        if cycle:
            # NOTE: itertools.cycle() usage below consumes a lot of memory
            # since it keeps every object from iterator which is a whole file.
            # Need to be revisited if we want to play huge files on a limited
            # RAM machines.
            scans_source = itertools.cycle(scans_source)

        SimpleViz(ls_viz, rate=rate, pause_at=pause_at,
                  _buflen=buf).run(scans_source)

    finally:
        if source.id_error_count:
            print(f"WARNING: {source.id_error_count} lidar_packets with "
                  "mismatched init_id/sn were detected.")
            if not soft_id_check:
                print("NOTE: To disable strict init_id/sn checking use "
                      "--soft-id-check option (may lead to parsing "
                      "errors)")

    click.echo("Done")


@pcap_group.command(name="slice")
@click.argument('file', type=click_ro_file)
@click.option('-s', '--start-frame', default=0, help="Start frame index")
@click.option('-n', '--num-frames', default=10, help="Number of frames")
@click.option('-f', '--meta', required=False, type=click_ro_file)
@click.option('-l',
              '--lidar-port',
              default=None,
              help="Dest. port of lidar data")
@click.option('-i', '--imu-port', default=None, help="Dest. port of imu data")
@click.option('-o', '--output', default=None, type=click.Path(exists=False))
@click.option('--soft-id-check',
              is_flag=True,
              help="Continue parsing lidar packets even if init_id/sn doesn't "
              "match with metadata")
def pcap_slice(file: str, start_frame: int, num_frames: int,
               meta: Optional[str], lidar_port: Optional[int],
               imu_port: Optional[int], output: Optional[str],
               soft_id_check: bool) -> None:
    """Truncate a pcap file to the specified frames."""

    try:
        import ouster.pcap as pcap
    except ImportError:
        raise click.ClickException("Please verify that libpcap is installed")

    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-f`")
    with open(meta) as json:
        click.echo(f"Reading metadata from: {meta}")
        info = client.SensorInfo(json.read())

    source = pcap.Pcap(file,
                       info,
                       lidar_port=lidar_port,
                       imu_port=imu_port,
                       _soft_id_check=soft_id_check)

    frames = packet_iter.ichunked_framed(source)
    sel_frames = itertools.islice(frames, start_frame,
                                  start_frame + num_frames)

    default_output = f"./slice_{start_frame}-{num_frames}_{os.path.basename(file)}"
    new_pcap_path = output or default_output
    click.echo(f"Writing: {new_pcap_path}")

    try:
        with click.progressbar(sel_frames, length=num_frames,
                               label="Progress:") as prog_frames:
            # TODO[pb]: pass lidar_port and imu_port as they are read from Pcap
            # source?
            pcap.record(itertools.chain.from_iterable(prog_frames),
                        new_pcap_path)
    finally:
        if source.id_error_count:
            print(f"WARNING: {source.id_error_count} lidar_packets with "
                  "mismatched init_id/sn were detected.")
            if not soft_id_check:
                print("NOTE: To disable strict init_id/sn checking use "
                      "--soft-id-check option (may lead to parsing "
                      "errors)")


@pcap_group.command(name="to_bag")
@click.argument('file', required=True)
@click.option('-f', '--meta', required=False, type=click_ro_file)
@click.option('-l', '--lidar-port', default=None, help="Dest. port of lidar data")
@click.option('-i', '--imu-port', default=None, help="Dest. port of imu data")
@click.option('-o', '--output', required=False, help="BAG output filename")
@click.option('--soft-id-check',
              is_flag=True,
              help="Continue parsing lidar packets even if init_id/sn doesn't "
              "match with metadata")
def pcap_to_bag(file: str, meta: Optional[str], lidar_port: Optional[int],
                imu_port: Optional[int], output: Optional[str],
                soft_id_check: bool) -> None:
    return pcap_to_bag_impl(file, meta, lidar_port, imu_port, output, soft_id_check)


def pcap_to_bag_impl(file: str, meta: Optional[str], lidar_port: Optional[int],
                imu_port: Optional[int], output: Optional[str],
                soft_id_check: bool) -> None:
    """Convert pcap to bag.

    Requires the active ROS environment or ROS-less rospy/rosbag python
    modules installed. See error message for details.
    """

    try:
        import ouster.pcap as pcap
    except ImportError:
        raise click.ClickException("Please verify that libpcap is installed")

    # Checks only ros imports availability
    import_rosbag_modules(raise_on_fail=True)

    from ouster.sdkx.bag import PacketMsg

    import rosbag
    import rospy

    try:
        meta = resolve_metadata(file, meta)
        if not meta:
            raise click.ClickException(
                "File not found, please specify a metadata file with `-f`")
        with open(meta) as json:
            click.echo(f"Reading metadata from: {meta}")
            info = client.SensorInfo(json.read())
    except click.ClickException:
        # if meta can't be resolved we can work without one, so
        # creating a defaut info to enable pcap.Pcap() call below
        info = client.SensorInfo.from_default(client.LidarMode.MODE_1024x10)

    if not output:
        output = os.path.splitext(os.path.basename(file))[0] + '.bag'

    print("Converting: ")
    print(f"  PCAP file: {file}")
    print(f"  with json file: {meta}")
    print(f"  to BAG file: {output}")

    source: client.PacketSource
    source = pcap.Pcap(file,
                       info,
                       lidar_port=lidar_port,
                       imu_port=imu_port,
                       _soft_id_check=soft_id_check)

    # Get info from Pcap source of guessed port to show to the user.
    # yes, using some private members here, but it's useful to see what was guessed
    if source.ports[0]:
        lidar_port = source.ports[0]
    if source.ports[1]:
        imu_port = source.ports[1]

    print("\nUsing sensor data:")
    print(f"  lidar_port = {lidar_port}, imu_port = {imu_port}")

    lidar_topic = "/os_node/lidar_packets"
    imu_topic = "/os_node/imu_packets"

    ls_cnt = 0
    imu_cnt = 0

    print("\nConverting PCAP to BAG ... ")
    try:
        with rosbag.Bag(output, 'w') as outbag:
            for packet in source:
                ts = rospy.Time.from_sec(packet.capture_timestamp)
                msg = PacketMsg(buf=packet._data.tobytes())
                if isinstance(packet, client.LidarPacket):
                    outbag.write(lidar_topic, msg, ts)
                    ls_cnt += 1
                elif isinstance(packet, client.ImuPacket):
                    outbag.write(imu_topic, msg, ts)
                    imu_cnt += 1
    except KeyboardInterrupt:
        print("Interrupted! Finishing up ...")
    finally:
        print(f"\nSaved to BAG file: {output}")
        print(f"  LidarPackets : {ls_cnt}\t(topic: {lidar_topic})")
        print(f"  ImuPackets   : {imu_cnt}\t(topic: {imu_topic})")

        if source.id_error_count:
            print(f"WARNING: {source.id_error_count} lidar_packets with "
                  "mismatched init_id/sn were detected.")
            if not soft_id_check:
                print("NOTE: To disable strict init_id/sn checking use "
                      "--soft-id-check option (may lead to parsing "
                      "errors)")


@pcap_group.command(name="from_bag")
@click.argument('file', required=True, type=click.Path(exists=True))
@click.option('-f', '--meta', required=False, type=click_ro_file)
@click.option('-l', '--lidar-topic', default="", help="Topic with lidar data")
@click.option('-i', '--imu-topic', default="", help="Topic with imu data")
@click.option('-o', '--output', required=False, help="BAG output filename")
def bag_to_pcap(file: str, meta: Optional[str], lidar_topic: str,
                imu_topic: str, output: Optional[str]) -> None:
    """Convert bag to pcap.

    Requires the active ROS environment or ROS-less rospy/rosbag python
    modules installed. See error message for details.
    """

    try:
        import ouster.pcap as pcap
    except ImportError:
        raise click.ClickException("Please verify that libpcap is installed")

    # Checks only ros imports availability
    import_rosbag_modules(raise_on_fail=True)

    from ouster.sdkx.bag import BagSource

    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-f`")
    with open(meta) as json:
        click.echo(f"Reading metadata from: {meta}")
        info = client.SensorInfo(json.read())

    if not output:
        output = os.path.splitext(os.path.basename(file))[0] + '.pcap'

    print("Converting: ")
    print(f"  BAG file: {file}")
    print(f"  with json file: {meta}")
    print(f"  to PCAP file: {output}")

    source = BagSource(file,
                       info,
                       lidar_topic=lidar_topic,
                       imu_topic=imu_topic)

    print("\nUsing sensor data:")
    print(f"  topics = {source.topics}")

    lidar_port = info.udp_port_lidar if info.udp_port_lidar else 7502
    imu_port = info.udp_port_imu if info.udp_port_imu else 7503

    print("\nUsing output ports:")
    print(f"  lidar_port = {lidar_port}")
    print(f"  imu_port = {imu_port}")

    lp_cnt = 0
    imup_cnt = 0

    def count_packets(packet: client.Packet):
        nonlocal lp_cnt, imup_cnt
        if isinstance(packet, client.LidarPacket):
            lp_cnt += 1
        elif isinstance(packet, client.ImuPacket):
            imup_cnt += 1

    print("\nConverting BAG to PCAP ... ")
    packets_written = 0
    keyboard_int = False
    try:
        # TODO: For a better user experience we may want to add a progress bar
        # output for this and other converters commands.
        packets_written = pcap.record(side_effect(count_packets, source),
                                      output,
                                      lidar_port=lidar_port,
                                      imu_port=imu_port)

    except KeyboardInterrupt:
        print("Interrupted! Finishing up ...")
        keyboard_int = True
    finally:
        print(f"\nSaved to PCAP file: {output}")
        if not keyboard_int:
            print(f"  total packets written: {packets_written}")
        print(f"    LidarPackets:        {lp_cnt}")
        print(f"    ImuPackets:          {imup_cnt}")
        source.close()


@pcap_group.command(name="to_csv")
@click.argument('file', required=True, type=click.Path(exists=True))
@click.option('-f', '--meta', required=False, type=click_ro_file)
@click.option('--csv-dir', default=".", help="path to the directory to save csv files")
@click.option('--csv-base', default="csv_out", help="base filename string for pcap output")
@click.option('--csv-ext', default='csv', help="file extension to use, 'csv' by default.")
@click.option('--start-index', default=0, help="index of scan to start outputting")
@click.option('--num-scans', default=1, help="number of scans to save from pcap to csv files")
def pcap_to_csv(file: str,
                meta: Optional[str],
                num_scans: Optional[int],
                start_index: Optional[int],
                csv_dir: Optional[str],
                csv_base: Optional[str],
                csv_ext: Optional[str]) -> None:
    """Write scans from a pcap to csv files (one per lidar scan).

    The number of saved lines per csv file is always H x W, which corresponds to
    a full 2D image representation of a lidar scan.

    Each line in a csv file is (for DUAL profile):

        TIMESTAMP (ns), RANGE (mm), RANGE2 (mm), SIGNAL (photons),
            SIGNAL2 (photons), REFLECTIVITY (%), REFLECTIVITY2 (%),
            NEAR_IR (photons), X (m), Y (m), Z (m), X2 (m), Y2 (m), Z2(m),
            MEASUREMENT_ID, ROW, COLUMN

    If ``csv_ext`` ends in ``.gz``, the file is automatically saved in
    compressed gzip format. :func:`.numpy.loadtxt` can be used to read gzipped
    files transparently back to :class:`.numpy.ndarray`.
    """

    # ensure that base csv_dir exists
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    output_paths = [os.path.join(csv_dir, f'{csv_base}_{idx:06d}.{csv_ext}') for
            idx in range(start_index, start_index + num_scans)]

    pcap_to_csv_impl(file, meta, start_index, num_scans, output_paths)


def pcap_to_csv_impl(file: str,
                meta: Optional[str],
                start_index: Optional[int],
                num_scans: Optional[int],
                output_names: Optional[List[str]]) -> None:

    try:
        import ouster.pcap as pcap
    except ImportError:
        raise click.ClickException("Please verify that libpcap is installed")

    metadata_path = resolve_metadata(file, meta)
    if not metadata_path:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-f`")
    with open(metadata_path) as json:
        click.echo(f"Reading metadata from: {metadata_path}")
        metadata = client.SensorInfo(json.read())

    source = pcap.Pcap(file, metadata)

    dual = False
    if metadata.format.udp_profile_lidar == client.UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16_DUAL:
        dual = True
        print("Note: You've selected to convert a dual returns pcap to CSV. Each row "
              "will represent a single pixel, so that both returns for that pixel will "
              "be on a single row. As this is an example we provide for getting "
              "started, we realize that you may have conversion needs which are not met "
              "by this function. You can find the source code on the Python SDK "
              "documentation website to modify it for your own needs.")

    # construct csv header and data format
    def get_fields_info(scan: client.LidarScan) -> Tuple[str, List[str]]:
        field_names = 'TIMESTAMP (ns), ROW, DESTAGGERED IMAGE COLUMN, MEASUREMENT_ID'
        field_fmts = ['%d'] * 4
        for chan_field in scan.fields:
            field_names += f', {chan_field}'
            if chan_field in [client.ChanField.RANGE, client.ChanField.RANGE2]:
                field_names += ' (mm)'
            if chan_field in [client.ChanField.REFLECTIVITY, client.ChanField.REFLECTIVITY2]:
                field_names += ' (%)'
            if chan_field in [client.ChanField.SIGNAL, client.ChanField.SIGNAL2,
                    client.ChanField.NEAR_IR]:
                field_names += ' (photons)'
            field_fmts.append('%d')
        field_names += ', X1 (m), Y1 (m), Z1 (m)'
        field_fmts.extend(3 * ['%.4f'])
        if dual:
            field_names += ', X2 (m), Y2 (m), Z2 (m)'
            field_fmts.extend(3 * ['%.4f'])
        return field_names, field_fmts

    field_names: str = ''
    field_fmts: List[str] = []

    # [doc-stag-pcap-to-csv]
    from itertools import islice
    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    # if num_scans is None
    scans = islice(scans, start_index, start_index + num_scans)

    row_layer = np.fromfunction(lambda i, j: i,
            (metadata.format.pixels_per_column,
                metadata.format.columns_per_frame), dtype=int)
    column_layer = np.fromfunction(lambda i, j: j,
            (metadata.format.pixels_per_column,
                metadata.format.columns_per_frame), dtype=int)
    column_layer_staggered = client.destagger(metadata, column_layer,
            inverse=True)

    idx = None
    for idx, scan in enumerate(scans):

        # initialize the field names for csv header
        if not field_names or not field_fmts:
            field_names, field_fmts = get_fields_info(scan)

        # copy per-column timestamps and measurement_ids for each beam
        timestamps = np.tile(scan.timestamp, (scan.h, 1))
        measurement_ids = np.tile(scan.measurement_id, (scan.h, 1))

        # grab channel data
        fields_values = [scan.field(ch) for ch in scan.fields]

        frame = np.dstack((timestamps, row_layer, column_layer_staggered,
            measurement_ids, *fields_values))

        # output points in "image" vs. staggered order
        frame = client.destagger(metadata, frame)

        # destagger XYZ separately since it has a different type
        xyz = xyzlut(scan.field(client.ChanField.RANGE))
        xyz_destaggered = client.destagger(metadata, xyz)

        if dual:
            xyz2 = xyzlut(scan.field(client.ChanField.RANGE2))
            xyz2_destaggered = client.destagger(metadata, xyz2)

            # get all data as one H x W x num fields int64 array for savetxt()
            frame = np.dstack(tuple(map(lambda x: x.astype(object),
                (frame, xyz_destaggered, xyz2_destaggered))))

        else:
            # get all data as one H x W x num fields int64 array for savetxt()
            frame = np.dstack(tuple(map(lambda x: x.astype(object),
                (frame, xyz_destaggered))))

        # write csv out to file
        csv_path = output_names[idx]
        print(f'write frame index #{idx + start_index}, to file: {csv_path}')

        header = '\n'.join([f'frame num: {idx}', field_names])

        np.savetxt(csv_path,
                   frame.reshape(-1, frame.shape[2]),
                   fmt=field_fmts,
                   delimiter=',',
                   header=header)
    # [doc-etag-pcap-to-csv]
    if idx is None:
        print("No CSVs outputted. Check your start index to ensure that it "
              "doesn't start past the total number of frames in your PCAP")
