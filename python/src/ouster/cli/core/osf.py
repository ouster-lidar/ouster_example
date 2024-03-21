import click
import os
from pprint import pprint

from typing import Iterator, Dict, cast, Optional, List, Tuple, Union
import numpy as np

from ouster.sdk.pcap import PcapScanSource  # type: ignore
from ouster.sdk.osf import OsfScanSource  # type: ignore


def osf_from_pcap_impl(file: str, meta: Optional[str], output: Optional[str],
                       chunk_size: int, flags: bool, raw_headers: bool,
                       raw_fields: bool, extrinsics: Optional[List[float]],
                       multi: bool, soft_id_check: bool) -> None:
    # TODO: deprecate the multi parameter
    """
    Convert PCAP file to OSF.
    """
    try:
        from ouster.sdk import client
        import ouster.sdk.osf as osf
    except ImportError as e:
        raise click.ClickException("Error: " + str(e))

    if not output:
        output = os.path.splitext(os.path.basename(file))[0] + '.osf'

    scan_source = PcapScanSource(file,
                                 flags=flags,
                                 raw_headers=raw_headers,
                                 raw_fields=raw_fields,
                                 _soft_id_check=soft_id_check)

    click.echo(f"Converting: \n"
               f"  PCAP file: {file}\n"
               f"  with json files: {scan_source._metadata_paths}\n"
               f"  to OSF file: {output}\n"
               f"  chunks_layout: STREAMING, chunks_size: "
               f"{chunk_size if chunk_size else 'DEFAULT'},"
               f" FLAGS: {flags}, RAW fields: {raw_fields}\n")

    if extrinsics and scan_source.sensors_count == 1:
        scan_source._source.metadata[0].extrinsic = np.array(extrinsics).reshape((4, 4))    # type: ignore
        scan_source._source.extrinsics_source[0] = "osf_from_pcap_param"                    # type: ignore

    # Using osf.Writer to save pcap to OSF
    writer = osf.Writer(output, "from_pcap pythonic", chunk_size)

    # sensor idx in multi source mapped to meta ids in OSF
    sensor_id = dict()

    click.echo("\nUsing sensors data:")
    for idx, sinfo in enumerate(scan_source.metadata):
        lidar_sensor_meta = osf.LidarSensor(sinfo.original_string())
        sensor_id[idx] = writer.addMetadata(lidar_sensor_meta)
        click.echo(
            f"  [{idx};{sinfo.sn}->{sensor_id[idx]}]: "
            f"lidar_port = {sinfo.udp_port_lidar}")

    click.echo()

    for idx, ext_source in enumerate(scan_source._source.extrinsics_source):    # type: ignore

        if ext_source is None:
            # skip extrinsics if they weren't explicitly set during previous
            # steps
            continue

        # add Extrinsics to OSF for a corresponding sensor
        sinfo = scan_source.metadata[idx]
        extrinsics_meta = osf.Extrinsics(sinfo.extrinsic, sensor_id[idx],
                                         ext_source)
        writer.addMetadata(extrinsics_meta)
        click.echo(
            f"Adding extrinsics for [{idx};{sinfo.sn}->{sensor_id[idx]}]: "
            f"from: {ext_source}:\n{sinfo.extrinsic}")

    click.echo()

    for idx, sinfo in enumerate(scan_source.metadata):
        click.echo(f"LidarScan resolved field_types for [{idx};{sinfo.sn}]:")
        pprint(list(scan_source.fields)[idx], indent=4)

    # OSF stream writers by type and sensor idx in multi source
    lidar_stream = {}
    for idx in range(scan_source.sensors_count):
        lidar_stream[idx] = osf.LidarScanStream(writer, sensor_id[idx],
                                                list(scan_source.fields)[idx])

    # running counts of saved messages to OSF
    ls_cnt: Dict[int, int] = {}
    for idx, _ in enumerate(list(scan_source.fields)):
        ls_cnt[idx] = 0
    click.echo("\nConverting PCAP to OSF ... ")
    try:
        # ask datasource to use custom fields for LidarScans thus
        # enabling FLAGS fields if they were requested via `--flags` param
        for scans in scan_source:
            for idx, scan in enumerate(scans):
                if scan:
                    ts = client.first_valid_packet_ts(scan)
                    if ts:
                        lidar_stream[idx].save(ts, scan)
                        ls_cnt[idx] += 1
                    else:
                        click.echo(F"warning: stream[{idx}]/LidarScan[{ls_cnt[idx]}]"
                                   " has no valid packet timestamp skipping..")
    except KeyboardInterrupt:
        click.echo("Interrupted! Finishing up ...")
    finally:
        click.echo(f"\nSaved to OSF file: {output}\n"
                   "  Lidar Scan messages:")
        pprint(ls_cnt)
        writer.close()

        # checking for bad init_ids
        if scan_source._source.id_error_count:  # type: ignore
            click.echo(f"WARNING: {scan_source._id_error_count} lidar_packets with "
                       "mismatched init_id/sn were detected.")
            if not soft_id_check:
                click.echo("NOTE: To disable strict init_id/sn checking use "
                           "--soft-id-check option (may lead to parsing "
                           "errors)")


@click.group(name="osf", hidden=True)
@click.pass_context
def osf_group(ctx) -> None:
    """Commands for working with OSF files and converting data to OSF."""
    try:
        from ouster.sdk.osf import _osf
    except ImportError as e:
        raise click.ClickException("Error: " + str(e))
    ctx.ensure_object(dict)
    sdk_log_level = ctx.obj.get('SDK_LOG_LEVEL', None)
    if sdk_log_level:
        _osf.init_logger(sdk_log_level)


@osf_group.command(name='info')  # type: ignore
@click.argument('file', required=True, type=click.Path(exists=True))
@click.option('-s', '--short', is_flag=True, help='Print less metadata info')
@click.pass_context
def osf_info(ctx, file: str, short: bool) -> None:
    """Print information about an OSF file to stdout.

    Parses all metadata entries, output is in JSON format.
    """
    try:
        from ouster.sdk.osf import _osf
    except ImportError as e:
        raise click.ClickException("Error: " + str(e))

    if not ctx.obj.get('SDK_LOG_LEVEL', None):
        # If not SDK_LOG_LEVEL passed we set to "error" logging so to ensure
        # that json output is not interferred with other SDK logging messages
        # and thus ruining valid json structure
        _osf.init_logger("error")

    print(_osf.dump_metadata(file, not short))


@osf_group.command(name='parse')  # type: ignore
@click.argument('file',
                required=True,
                type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--decode', is_flag=True, help="Decode messages")
@click.option('-v',
              '--verbose',
              is_flag=True,
              help="Verbose LidarScan outputs (only when used with -d option)")
@click.option('-r',
              '--check-raw-headers',
              is_flag=True,
              help="Check RAW_HEADERS fields by reconstructing lidar_packets"
              " and batching LidarScan back (without fields data) and compare."
              "(applies only when used with -d option)")
@click.option('-s',
              '--standard',
              is_flag=True,
              help="Show standard layout with chunks")
def osf_parse(file: str, decode: bool, verbose: bool, check_raw_headers: bool,
              standard: bool) -> None:
    """
    Read an OSF file and print messages type, timestamp and counts to stdout.
    Useful to check chunks layout and decoding of all known messages (-d option).
    """
    try:
        from ouster.sdk import client
        import ouster.sdk.osf as osf
    except ImportError as e:
        raise click.ClickException("Error: " + str(e))

    # NOTE[pb]: Mypy quirks or some of our Python packages structure quirks, idk :(
    from ouster.sdk.client._client import get_field_types
    from ouster.sdk.util import scan_to_packets, packets_to_scan, cut_raw32_words  # type: ignore

    reader = osf.Reader(file)

    orig_layout = "STREAMING" if reader.has_stream_info else "STANDARD"

    print(f"filename: {file}, layout: {orig_layout}")

    # map stream_id to metadata entry
    scan_stream_sensor: Dict[int, osf.LidarSensor]
    scan_stream_sensor = {}
    for scan_stream_id, scan_stream_meta in reader.meta_store.find(
            osf.LidarScanStream).items():
        scan_stream_sensor[scan_stream_id] = reader.meta_store[
            scan_stream_meta.sensor_meta_id]

    ls_cnt = 0
    other_cnt = 0

    def proc_msgs(msgs: Iterator[osf.MessageRef]):
        nonlocal ls_cnt, other_cnt, decode
        for m in msgs:
            if m.of(osf.LidarScanStream):
                prefix = "Ls"
                ls_cnt += 1
            else:
                prefix = "UN"
                other_cnt += 1
            d = ""
            verbose_str = ""
            if decode:
                obj = m.decode()
                d = "[D]" if obj else ""
                if m.of(osf.LidarScanStream):
                    ls = cast(client.LidarScan, obj)

                    d = d + \
                        (" [poses: YES]" if client.poses_present(ls) else "")

                    if verbose:
                        verbose_str += f"{ls}"

                    if check_raw_headers:
                        d = d + " " if d else ""
                        if client.ChanField.RAW_HEADERS in ls.fields:
                            sinfo = scan_stream_sensor[m.id].info

                            # roundtrip: LidarScan -> packets -> LidarScan
                            packets = scan_to_packets(ls, sinfo)

                            # recovered lidar scan
                            field_types = get_field_types(ls)
                            ls_rec = packets_to_scan(
                                packets, sinfo, fields=field_types)

                            ls_no_raw32 = cut_raw32_words(ls)
                            ls_rec_no_raw32 = cut_raw32_words(ls_rec)

                            assert ls_rec_no_raw32 == ls_no_raw32, "LidarScan should be" \
                                " equal when recontructed from RAW_HEADERS fields" \
                                " packets back"

                            d += "[RAW_HEADERS: OK]"
                        else:
                            d += "[RAW_HEADERS: NONE]"

            print(f"  {prefix}\tts: {m.ts}\t\tstream_id: {m.id}\t{d}")
            if verbose_str:
                print(60 * '-')
                print(f"{verbose_str}")
                print(60 * '-')

    if not standard and reader.has_stream_info:
        proc_layout = "STREAMING"
        proc_msgs(reader.messages())
    else:
        proc_layout = "STANDARD"
        for chunk in reader.chunks():
            print(f"Chunk [{chunk.offset}\t\t]: start_ts = {chunk.start_ts}, "
                  f"end_ts = {chunk.end_ts}")
            proc_msgs(iter(chunk))

    showed_as_str = ""
    if orig_layout != proc_layout:
        showed_as_str = f"showed as: {proc_layout}"

    print()
    print(f"SUMMARY: [layout: {orig_layout}] {showed_as_str}")
    print(f"  lidar_scan    (Ls)    count = {ls_cnt}")
    print(f"  other                 count = {other_cnt}")


@osf_group.command(name="viz")
@click.argument("file",
                required=True,
                type=click.Path(exists=True, dir_okay=False))
@click.option('-e',
              '--on-eof',
              default='loop',
              type=click.Choice(['loop', 'stop', 'exit']),
              help="Loop, stop, or exit after reaching end of file")
@click.option("-p", "--pause", is_flag=True, help="Pause at first lidar scan")
@click.option("--pause-at",
              default=-1,
              help="Lidar Scan number to pause")
@click.option("-r", "--rate", default=1.0, help="Playback rate")
@click.option("--extrinsics",
              type=float,
              required=False,
              nargs=16,
              help="Lidar sensor extrinsics to use in viz (instead possible "
                   " extrinsics stored in OSF)")
@click.option("--skip-extrinsics",
              is_flag=True,
              help="Don't use any extrinsics (leaves them at Identity)")
@click.option("--sensor-id",
              type=int,
              required=False,
              default=0,
              help="Viz only the single sensor by sensor_id")
@click.option("--multi",
              is_flag=True,
              help="Use multi sensor viz")
@click.option("--accum-num",
              default=0,
              help="Integer number of scans to accumulate")
@click.option("--accum-every",
              default=None,
              type=float,
              help="Accumulate every Nth scan")
@click.option("--accum-every-m",
              default=None,
              type=float,
              help="Accumulate scan every M meters traveled")
@click.option("--accum-map",
              is_flag=True,
              help="Enable the overall map accumulation mode")
@click.option("--accum-map-ratio",
              default=0.001,
              help="Ratio of random points of every scan to add to an overall map")
def osf_viz(file: str, on_eof: str, pause: bool, pause_at: int, rate: float,
            extrinsics: Optional[List[float]], skip_extrinsics: bool,
            sensor_id: int, multi: bool, accum_num: int,
            accum_every: Optional[int], accum_every_m: Optional[float],
            accum_map: bool, accum_map_ratio: float) -> None:
    """Visualize Lidar Scan Data from an OSF file.

    Only one LidarScan stream will be shown, unless ``--multi`` is set.
    """
    try:
        import ouster.sdk.osf as osf
        from ouster.sdk.viz import SimpleViz, LidarScanViz, scans_accum_for_cli
        from ouster.sdk.viz.multi_viz import MultiLidarScanViz  # type: ignore
    except ImportError as e:
        raise click.ClickException(str(e))

    if pause and pause_at == -1:
        pause_at = 0

    if rate not in SimpleViz._playback_rates:
        raise click.ClickException("Invalid rate specified")

    def single_viz(file: str, on_eof: str,
                   extrinsics: Optional[List[float]], skip_extrinsics: bool,
                   sensor_id: int) -> Tuple[osf.Scans, LidarScanViz]:
        scan_source: osf.Scans
        scan_source = osf.Scans(file,
                                cycle=(on_eof == 'loop'),
                                sensor_id=sensor_id)
        # overwrite extrinsics of a sensor stored in OSF if --extrinsics arg is
        # provided
        if extrinsics and not skip_extrinsics:
            scan_source.metadata.extrinsic = np.array(
                extrinsics).reshape((4, 4))
            click.echo(
                f"Overwriting sensor extrinsics to:\n"
                f"{scan_source.metadata.extrinsic}")
        if skip_extrinsics:
            scan_source.metadata.extrinsic = np.eye(4)
            click.echo(
                f"Setting all sensor extrinsics to Identity:\n"
                f"{scan_source.metadata.extrinsic}")

        ls_viz: LidarScanViz
        ls_viz = LidarScanViz(scan_source.metadata)
        return scan_source, ls_viz

    def multi_viz(file: str, on_eof: str) -> Tuple[OsfScanSource, MultiLidarScanViz]:
        # Multi sensor viz
        scan_source: OsfScanSource
        scan_source = OsfScanSource(file,
                                    cycle=(on_eof == 'loop'))

        # TODO: reconsider this?
        for idx, (sid, _) in enumerate(scan_source._sensors):
            scan_source.metadata[idx].hostname = f"sensorid: {sid}"

        ls_viz: MultiLidarScanViz
        ls_viz = MultiLidarScanViz(scan_source.metadata, source_name=file)
        return scan_source, ls_viz

    # TODO[pb]: Switch to aligned Protocol/Interfaces that we
    # should get after some refactoring/designing
    scan_source: Union[osf.Scans, OsfScanSource]
    ls_viz: Union[LidarScanViz, MultiLidarScanViz]

    if not multi:
        scan_source, ls_viz = single_viz(file, on_eof, extrinsics,
                                         skip_extrinsics, sensor_id)
        scans = scan_source
    else:
        scan_source, ls_viz = multi_viz(file, on_eof)
        scans = iter(scan_source)  # type: ignore

    scans_accum = scans_accum_for_cli(scan_source.metadata,
                                      accum_num=accum_num,
                                      accum_every=accum_every,
                                      accum_every_m=accum_every_m,
                                      accum_map=accum_map,
                                      accum_map_ratio=accum_map_ratio)

    SimpleViz(ls_viz,
              rate=rate,
              pause_at=pause_at,
              on_eof=on_eof,
              scans_accum=scans_accum).run(scans)

    click.echo("Done")
