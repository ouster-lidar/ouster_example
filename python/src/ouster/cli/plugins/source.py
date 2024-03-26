#  type: ignore
import click
import re
import threading
from itertools import islice
from ouster.cli.core import cli
from ouster.cli.core.cli_args import CliArgs
from ouster.cli.core.util import click_ro_file
from ouster.sdk import open_source
from ouster.sdk.client.core import ClientTimeout
import ouster.cli.core.pcap
import ouster.cli.core.sensor
import ouster.cli.core.osf as osf_cli
from typing import (List, Optional, Iterable, Tuple)
from .io_type import (extension_from_io_type, io_type, OusterIoType)
from .source_save import SourceSaveCommand
from .source_util import (CoupledTee,
                          SourceCommandContext,
                          SourceCommandCallback,
                          SourceCommandType,
                          source_multicommand,
                          _join_with_conjunction)

# TODO: generalize the current solution for any like-plugins that want
# to alter or extend the default command set
_has_mapping = False
try:
    from ouster.cli.plugins.cli_source_mapping import (SourceMappingSaveCommand,    # type: ignore # noqa: F401
                                                       source_slam)
    _has_mapping = True
except ImportError:
    pass


_source_arg_name: str = 'source'


@click.command()
@click.argument('keyval', metavar='[KEY VAL]...', type=str, nargs=-1)
@click.option('-d', 'dump', is_flag=True, help='Dump current configuration')
@click.option('-c', 'file', type=click.File(), help='Read config from file')
@click.option('-u', 'auto', is_flag=True, help='Set automatic udp dest')
@click.option('-p', 'persist', is_flag=True, help='Persist configuration')
@click.option('-s/-n', 'standby', default=None, help='Set STANDBY or NORMAL')
@click.pass_context
@source_multicommand(type=SourceCommandType.MULTICOMMAND_UNSUPPORTED,
                     retrieve_click_context=True)
def sensor_config(ctx: SourceCommandContext, click_ctx: click.core.Context,
                  *args, **kwargs) -> None:
    """
    Manipulate the sensor configuration.

      Update the sensor configuration or dump it to stdout. The first positional
      argument is the sensor hostname; remaining arguments are interpreted as
      config parameter key/value pairs, for example:

      \b
          $ ouster-cli sensor config os-99xxxxxxxxxx \\
          lidar_mode 2048x10 azimuth_window "[20000, 60000]"

      If no options or config param values are specified, use the default UDP
      ports, automatic UDP destination, full azimuth azimuth window, and set the
      operating mode to NORMAL.
    """
    # Implements ouster-cli source <hostname> config
    source = ctx.source_uri
    kwargs['hostname'] = source
    click_ctx.forward(ouster.cli.core.sensor.config, *args, **kwargs)


@click.command
@click.option('--legacy/--non-legacy',
              default=False,
              help="Use legacy metadata format or not")
@click.pass_context
@source_multicommand(type=SourceCommandType.MULTICOMMAND_UNSUPPORTED,
                     retrieve_click_context=True)
def sensor_info(ctx: SourceCommandContext, click_ctx: click.core.Context,
                *args, **kwargs) -> None:
    """Display info about the SOURCE."""  # Implements ouster-cli source <hostname> metadata
    source = ctx.source_uri
    kwargs['hostname'] = source
    click_ctx.forward(ouster.cli.core.sensor.metadata, *args, **kwargs)


@click.command
@click.option('-n', type=int, default=0, help="Read only INTEGER packets.")
@click.pass_context
@source_multicommand(type=SourceCommandType.PROCESSOR,
                     retrieve_click_context=True)
def pcap_info(ctx: SourceCommandContext, click_ctx: click.core.Context,
              *args, **kwargs) -> None:
    """Display info about the SOURCE."""
    source = ctx.source_uri
    kwargs['file'] = source
    click_ctx.forward(ouster.cli.core.pcap.pcap_info, *args, **kwargs)


@click.command
@click.option('-s', '--short', is_flag=True, help='Print less metadata info')
@click.pass_context
@source_multicommand(type=SourceCommandType.PROCESSOR,
                     retrieve_click_context=True)
def osf_info(ctx: SourceCommandContext, click_ctx: click.core.Context,
             *args, **kwargs) -> None:
    """Display info about the SOURCE."""  # Implements ouster-cli source <sourcefile>.osf info
    source = ctx.source_uri
    kwargs['file'] = source
    click_ctx.forward(osf_cli.osf_info, *args, **kwargs)


@click.command()
@click.option("-p", "--pause", is_flag=True, help="Pause at first lidar scan")
@click.option("-e", "--on-eof", default='loop', type=click.Choice(['loop', 'stop', 'exit']),
              help="Loop, stop or exit after reaching end of file")
@click.option("--pause-at",
              default=-1,
              help="Lidar Scan number to pause at")
@click.option("--accum-num",
              default=0,
              help="Integer number of scans to accumulate")
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
@click.pass_context
@source_multicommand(type=SourceCommandType.CONSUMER)
def source_viz(ctx: SourceCommandContext, pause: bool, on_eof: str, pause_at: int, accum_num: int,
               accum_every: Optional[int], accum_every_m: Optional[float],
               accum_map: bool, accum_map_ratio: float) -> SourceCommandCallback:
    """Visualize LidarScans in a 3D viewer."""
    try:
        from ouster.sdk.viz import SimpleViz, scans_accum_for_cli
    except ImportError as e:
        raise click.ClickException(str(e))

    # ugly workarounds ensue
    if on_eof == 'loop':
        source = ctx.scan_source
        from ouster.sdk.client import ScanSourceAdapter
        if isinstance(source, ScanSourceAdapter):
            source = source._scan_source
        # NOTE: setting it here instead of at open_source stage because we do not want to propagate
        #       the flag up to `source` command
        source._cycle = True

    if pause and pause_at == -1:
        pause_at = 0

    ctx.scan_iter, scans = CoupledTee.tee(ctx.scan_iter,
                            terminate=ctx.terminate_evt)
    metadata = ctx.scan_source.metadata
    scans_accum = scans_accum_for_cli(metadata,
                                accum_num=accum_num,
                                accum_every=accum_every,
                                accum_every_m=accum_every_m,
                                accum_map=accum_map,
                                accum_map_ratio=accum_map_ratio)

    def viz_thread_fn():
        sv = SimpleViz(metadata, scans_accum=scans_accum, rate=1.0, pause_at=pause_at, on_eof=on_eof)
        sv.run(scans)
        ctx.terminate_evt.set()

    if ctx.main_thread_fn is not None:
        raise RuntimeError("A main-thread required function has already been set.")
    ctx.main_thread_fn = viz_thread_fn


def extract_slice_indices(click_ctx: Optional[click.core.Context],
                          param: Optional[click.core.Argument], value: str):
    """Validate and extract slice indices of the form start:[stop][:step]."""
    index_matches = re.findall("^\[?(-?[0-9]+):(-?[0-9]*):?(-?[0-9]*)\]?$", value)  # noqa: W605

    # Check that indices can be parsed
    if not index_matches or len(index_matches[0]) != 3:
        raise click.exceptions.BadParameter("slice indices must be of the form start:[stop][:step]")
    parsed_indices = [int(i) if i != "" else None for i in index_matches[0]]
    start, stop, step = parsed_indices[0], parsed_indices[1], parsed_indices[2]
    # Check that indices are non-negative
    if any(i < 0 if i is not None else False for i in parsed_indices):
        raise click.exceptions.BadParameter("slice indices must be non-negative")
    # Check that stop > start
    if (stop is not None) and (not stop > start):
        raise click.exceptions.BadParameter("slice stop index must be greater than start")
    # Check that step > 1
    if (step is not None) and (not step > 0):
        raise click.exceptions.BadParameter("slice step index must be greater than 0")

    return start, stop, step


@click.command()
@click.argument('indices', required=True, callback=extract_slice_indices)
@click.pass_context
@source_multicommand(type=SourceCommandType.PROCESSOR)
def source_slice(ctx: SourceCommandContext,
                 indices: Tuple[Optional[int]]) -> SourceCommandCallback:
    """Slice LidarScans streamed from SOURCE."""
    start, stop, step = indices
    ctx.scan_iter = islice(ctx.scan_iter, start, stop, step)


class SourceMultiCommand(click.MultiCommand):
    """This class implements the ouster-cli source command group.  It uses the
    `io_type` method to determine the source type and map it to the
    available sub commands for that type.

    The source is also added to the click context so that sub commands that use
    @click.pass_context have access to it."""

    def __init__(self, *args, **kwargs):
        kwargs['no_args_is_help'] = True

        super().__init__(*args, **kwargs)
        self.commands = {
            'ANY': {
                'viz': source_viz,
                'slice': source_slice,
            },
            OusterIoType.SENSOR: {
                'config': sensor_config,
                'metadata': sensor_info,
                'save': SourceSaveCommand('save', context_settings=dict(ignore_unknown_options=True,
                                                                        allow_extra_args=True)),

            },
            OusterIoType.PCAP: {
                'info': pcap_info,
                'save': SourceSaveCommand('save', context_settings=dict(ignore_unknown_options=True,
                                                                        allow_extra_args=True)),
            },
            OusterIoType.OSF: {
                'info': osf_info,
                'save': SourceSaveCommand('save', context_settings=dict(ignore_unknown_options=True,
                                                                        allow_extra_args=True)),
            }
        }

        if _has_mapping:
            # extend current set of verbs
            self.commands['ANY']['slam'] = source_slam
            self.commands[OusterIoType.OSF]['save'] = SourceMappingSaveCommand(
                'save', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))

    def get_supported_source_types(self):
        return [iotype for iotype in self.commands.keys() if isinstance(iotype, OusterIoType)]

    def get_source_file_extension_str(self):
        exts = sorted(
            [extension_from_io_type(src_type)
                for src_type in self.commands.keys() if extension_from_io_type(src_type)]
        )
        return _join_with_conjunction(exts)

    def list_commands(self, click_ctx: click.core.Context):
        """Get the source type from the click context
        and return the list of appropriate sub command names"""
        source = click_ctx.params.get(_source_arg_name)

        if not source and CliArgs().has_any_of(click_ctx.help_option_names):
            # Build a map from command name to command
            command_to_types = {}
            for src_type in self.commands.keys():
                for command_name in self.commands[src_type].keys():
                    if command_name not in command_to_types:
                        command_to_types[command_name] = {}
                    if src_type == "ANY":
                        for supported_src_type in self.get_supported_source_types():
                            command_to_types[command_name][supported_src_type] = self.commands[src_type][command_name]
                    else:
                        command_to_types[command_name][src_type] = self.commands[src_type][command_name]

            # Prefix command name with names of supported source types
            command_to_types_renamed = {}
            for key, value in command_to_types.items():
                prefix = _join_with_conjunction([t.name.upper() for t in value.keys()], separator="|", conjunction="")
                command_to_types_renamed[f"{prefix} {key}"] = value

            return command_to_types_renamed

        file_extensions_str = self.get_source_file_extension_str()
        if not source:
            param_decls = [_source_arg_name]
            param = click.core.Argument(param_decls=param_decls)
            raise click.exceptions.MissingParameter(None, click_ctx, param=param)
        try:
            return {**self.commands[io_type(source)], **self.commands["ANY"]}
        except ValueError as e:  # noqa: F841
            click.echo(click_ctx.get_usage())
            raise click.exceptions.UsageError("Source type expected to be a sensor hostname, "
                                              f"ip address, or a(n) {file_extensions_str} file. "
                                              "For a sensor source, please check that you can "
                                              "ping the sensor hostname/ip address. For a file "
                                              "source, please check that the file path you have "
                                              "provided exists.")
        except KeyError as e:  # noqa: F841
            click.echo(click_ctx.get_usage())
            raise click.exceptions.UsageError("Source type expected to be a sensor hostname, "
                                              f"ip address, or a(n) {file_extensions_str} file. "
                                              "For a sensor source, please check that you can "
                                              "ping the sensor hostname/ip address. For a file "
                                              "source, please check that the file path you have "
                                              "provided exists.")

    def get_command(self, click_ctx: click.core.Context, name: str):
        """Get the click.Command object for the given command name"""
        source = click_ctx.params.get(_source_arg_name)
        click_ctx.ensure_object(SourceCommandContext)
        ctx: SourceCommandContext = click_ctx.obj
        # add source to context so the command can access it
        ctx.source_uri = source

        command_list = self.list_commands(click_ctx)
        if name in command_list:
            if not source:
                # If called by --help (without source), return the first implementation of a command
                # NOTE: This results in help printing the docstring of only the first implementation
                return command_list[name][list(command_list[name].keys())[0]]
            else:
                ctx.invoked_command_names.append(name)
                return command_list[name]
        return None

    def invoke(self, click_ctx: click.core.Context):
        """Called when the source command is invoked.
        If called without any args, prints the help.
        Otherwise, the superclass method is called."""
        if not click_ctx.protected_args:
            print(self.get_help(click_ctx))
            return
        super().invoke(click_ctx)


@cli.group(cls=SourceMultiCommand, chain=True)
@click.argument(_source_arg_name, required=True)
@click.option('-m', '--meta', required=False, type=click_ro_file,
        help="Metadata for PCAP, helpful if automatic metadata resolution fails")
@click.option('-l', '--lidar-port', default=None, type=int, help="Dest. port of lidar data")
@click.option('-i', '--imu-port', default=None, help="Dest. port of imu data")
@click.option('-e', '--extrinsics', type=float, nargs=16, help='Lidar sensor extrinsics.')
@click.option('-x', '--do-not-reinitialize', is_flag=True, default=False,
              help="Do not reinitialize (by default it will reinitialize if needed)")
@click.option('-y', '--no-auto-udp-dest', is_flag=True, default=False,
              help="Do not automatically set udp_dest (by default it will auto set udp_dest")
@click.option('-s', '--soft-id-check', is_flag=True, hidden=True,
              help="Continue parsing lidar packets even if init_id/sn doesn't match with metadata")  # noqa
@click.option('-t', '--timeout', default=1.0, help="Seconds to wait for data")
@click.option('-F', '--filter', is_flag=True, help="Drop scans missing data")
@click.option('-b', '--buf-size', default=1280, help="Max packets to buffer")
@click.option('--legacy/--non-legacy',
              default=False,
              help="Use legacy metadata format or not")
def source(source, meta: str, lidar_port: int, imu_port: int, extrinsics: Optional[List[float]],
           do_not_reinitialize: bool, no_auto_udp_dest: bool, soft_id_check: bool, timeout: int,
           filter: bool, buf_size: int, legacy: bool):
    """Run a command with the specified source (SENSOR, PCAP, or OSF) as SOURCE.
    For example, a sensor source: ouster-cli source os1-992xxx.local viz
    """
    pass


@source.result_callback()
@click.pass_context
def process_commands(click_ctx: click.core.Context, callbacks: Iterable[SourceCommandCallback],
                     source: str, meta: str, lidar_port: int, imu_port: int,
                     extrinsics: Optional[List[float]], do_not_reinitialize: bool,
                     no_auto_udp_dest: bool, soft_id_check: bool,
                     timeout: int, filter: bool, buf_size: int, legacy: bool) -> None:
    """Process all commands in a SourceMultiCommand, using each command's callback"""

    callbacks = list(callbacks)
    ctx: SourceCommandContext = click_ctx.obj
    command_names = ctx.invoked_command_names

    # ---- Lint commands ----
    # Ensure that no commands are duplicated
    names_duplicate_check = set()
    for name in command_names:
        if name in names_duplicate_check:
            raise click.exceptions.UsageError(f"'{name}' is duplicated in the multi-command chain. "
                                              "Please invoke it only once. ")
        names_duplicate_check.add(name)

    # Ensure that no other commands are present, if a MULTICOMMAND_UNSUPPORTED
    # is present
    multicommand = True
    for idx, c in enumerate(callbacks):
        if c.type is SourceCommandType.MULTICOMMAND_UNSUPPORTED:
            multicommand = False
            if len(callbacks) != 1:
                raise click.exceptions.UsageError(f"'{command_names[idx]}' does not support multi-command chaining. "
                                                  "Please invoke it without other commands. ")

    # Ensure that the order of commands is PROCESSOR < CONSUMER
    first_consumer_name, first_consumer_idx = None, None
    for idx, c in enumerate(callbacks):
        if c.type is SourceCommandType.PROCESSOR:
            if (first_consumer_idx is not None) and (idx >= first_consumer_idx):
                raise click.exceptions.UsageError(f"'{command_names[idx]}' must be invoked before "
                                                  f"'{first_consumer_name}'. Please reorder the multi-command chain. ")
        elif (c.type is SourceCommandType.CONSUMER) and (first_consumer_idx is None):
            first_consumer_idx = idx
            first_consumer_name = command_names[idx]

    if not multicommand:
        # Execute single non-multicommand command
        c = callbacks[0]
        c.callback_fn(ctx)
    else:
        # Execute multicommands
        # Open source
        ctx.scan_source = open_source(source, sensor_idx=0, meta=meta,
                                      lidar_port=lidar_port, imu_port=imu_port,
                                      extrinsics=extrinsics,
                                      do_not_reinitialize=do_not_reinitialize,
                                      no_auto_udp_dest=no_auto_udp_dest,
                                      soft_id_check=soft_id_check,
                                      timeout=timeout, complete=filter,
                                      buf_size=buf_size, legacy_format=legacy)
        ctx.scan_iter = iter(ctx.scan_source)

        # print any timeout exceptions we get
        scans = ctx.scan_iter

        def save_iter():
            try:
                for scan in scans:
                    yield scan
            except ClientTimeout as ex:
                print(f"ERROR: {ex}")
            return
        ctx.scan_iter = save_iter()
        try:
            # Execute multicommand callbacks

            # Dataflow between callbacks occur entirely through the scan iterator. Callbacks may additionally
            # register functions in ctx.thread_fns to be run in individual threads
            # Callback invariants:
            #   1. Must leave ctx.scan_iter in a consumable state after invocation.
            #       If a callback consumes the original ctx.scan_iter, it must set ctx.scan_iter
            #       to an unconsumed iterator (via CoupledTee or a new iterator)
            #   2. May add a Callable[None -> None] to ctx.thread_fns. These will be executed in
            #       individual threads.
            #   3. A single callback may also register a function to be run from the main thread in
            #       ctx.main_thread_fn. This is required to support viz on macOS, where openGL
            #       applications must be run from the main thread
            #   4. Any registered thread_fn must support termination of execution via
            #       ctx.terminate_evt.is_set()
            #   5. Every registered thread_fn must request the ouster-cli process terminate by calling
            #       ctx.terminate_evt.set() before terminating

            # Most callbacks will do one of the following:
            #   1. execute immediately, and take no further action (ie, config)
            #   2. map a processing function onto scans, such that processing is implicitly called while iterating
            #   3. set ctx.scan_iter to a new iterator, and register a processing thread in ctx.thread_fns
            #   4. create a CoupledTee from ctx.scan_iter, and re-set ctx.scan_iter to one of the resultant tees

            ctx.thread_fns = []
            ctx.main_thread_fn = None
            ctx.terminate_evt = threading.Event()
            for c in callbacks:
                c.callback_fn(ctx)

            # Create threads from functions registered by the callbacks
            threads = []
            for thread_fn in ctx.thread_fns:
                threads.append(threading.Thread(target=thread_fn))

            # Define a function to consume ctx.scan_iter
            def pipeline_flush():
                try:
                    for _ in ctx.scan_iter:
                        pass
                except Exception as ex:
                    # Terminate everything if we get an unhandled exception
                    ctx.terminate_evt.set()
                    raise ex

            threads.append(threading.Thread(target=pipeline_flush))

            # Start all threads
            for thread in threads:
                thread.start()

            # Execute main thread fn, if set
            if ctx.main_thread_fn is not None:
                ctx.main_thread_fn()
                ctx.terminate_evt.set()

            # Wait for threads to terminate
            for thread in threads:
                thread.join()

            # Todo MDB this is broken, uncomment when id error handling is fixed
            # true_source = ctx.scan_source._scan_source
            # if true_source._source._id_error_count > 0:
            #    print(f"WARNING: {true_source._source._id_error_count} lidar_packets with "
            #          "mismatched init_id/sn were detected.")
            #    if not soft_id_check:
            #        print("NOTE: To disable strict init_id/sn checking use "
            #              "--soft-id-check option (may lead to parsing "
            #              "errors)")
        finally:
            # Attempt to close scansource
            try:
                ctx.scan_source.close()
            except:  # noqa: E722
                pass
