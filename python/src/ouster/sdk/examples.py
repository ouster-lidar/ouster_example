"""Example code for Ouster Python SDK.

All examples commented out from main. Feel free to uncomment to try.

Note: if you want to run matplotlib within docker you will need tkinter
"""

import argparse
from contextlib import closing

import numpy as np
from more_itertools import time_limited

from ouster import client


def configure_sensor_params(hostname: str) -> None:
    """Configure sensor params given hostname

    Args:
        hostname: hostname of the sensor
    """

    # create empty config
    config = client.SensorConfig()

    # set the values that you need: see sensor docs for param meanings
    config.operating_mode = client.OperatingMode.OPERATING_NORMAL
    config.ld_mode = client.LidarMode.MODE_1024x10
    config.udp_port_lidar = 7502
    config.udp_port_imu = 7503

    # set the config on sensor, using persist bool if desired
    client.set_config(hostname, config, persist=False)

    # if you like, you can view the entire set of parameters
    config = client.get_config(hostname)
    print(f"sensor config of {hostname}:\n{config}")


def get_metadata(hostname: str) -> None:
    """Print metadata given hostname

    Args:
        hostname: hostname of the sensor
    """
# [doc-stag-get-metadata]
    with closing(client.Sensor(hostname)) as source:
        print(source.metadata)
# [doc-etag-get-metadata]


def display_range_2d(hostname: str, lidar_port: int = 7502) -> None:
    """Display range data taken live from sensor as an image

    Args:
        hostname: hostname of the sensor
        lidar_port: UDP port to listen on for lidar data
    """
    import matplotlib.pyplot as plt  # type: ignore

    # get single scan [doc-stag-single-scan]
    metadata, sample = client.Scans.sample(hostname, 1, lidar_port)
    scan = next(sample)[0]
    # [doc-etag-single-scan]

    # initialize plot
    fig, ax = plt.subplots()
    fig.canvas.set_window_title("example: display_range_2d")

    # plot using imshow
    range = scan.field(client.ChanField.RANGE)
    plt.imshow(client.destagger(metadata, range), resample=False)

    # configure and show plot
    plt.title("Range Data from {}".format(hostname))
    plt.axis('off')
    plt.show()


def display_all_2d(hostname: str,
                   lidar_port: int = 7502,
                   n_scans: int = 5) -> None:
    """Display all channels of n consecutive lidar scans taken live from sensor

    Args:
        hostname: hostname of the sensor
        lidar_port: UDP port to listen on for lidar data
        n_scans: number of scans to show
    """
    import matplotlib.pyplot as plt  # type: ignore

    # [doc-stag-display-all-2d]
    # take sample of n scans from sensor
    metadata, sample = client.Scans.sample(hostname, n_scans, lidar_port)

    # initialize and configure subplots
    fig, axarr = plt.subplots(n_scans,
                              4,
                              sharex=True,
                              sharey=True,
                              figsize=(12.0, n_scans * .75),
                              tight_layout=True)
    fig.suptitle("{} consecutive scans from {}".format(n_scans, hostname))
    fig.canvas.set_window_title("example: display_all_2D")

    # set row and column titles of subplots
    column_titles = ["range", "reflectivity", "ambient", "intensity"]
    row_titles = ["Scan {}".format(i) for i in list(range(n_scans))]
    for ax, column_title in zip(axarr[0], column_titles):
        ax.set_title(column_title)
    for ax, row_title in zip(axarr[:, 0], row_titles):
        ax.set_ylabel(row_title)

    # plot 2D scans
    for count, scan in enumerate(next(sample)):
        axarr[count, 0].imshow(
            client.destagger(metadata, scan.field(client.ChanField.RANGE)))
        axarr[count, 1].imshow(
            client.destagger(metadata,
                             scan.field(client.ChanField.REFLECTIVITY)))
        axarr[count, 2].imshow(
            client.destagger(metadata, scan.field(client.ChanField.AMBIENT)))
        axarr[count, 3].imshow(
            client.destagger(metadata, scan.field(client.ChanField.INTENSITY)))
    # [doc-etag-display-all-2d]

    # configure and show plot
    [ax.get_xaxis().set_visible(False) for ax in axarr.ravel()]
    [ax.set_yticks([]) for ax in axarr.ravel()]
    [ax.set_yticklabels([]) for ax in axarr.ravel()]
    plt.show()


def display_intensity_live(hostname: str, lidar_port: int = 7502) -> None:
    """
    Display intensity from live sensor

    Args:
        hostname: hostname of the sensor
        lidar_port: UDP port to listen on for lidar data

    """
    import cv2  # type: ignore

    print("press ESC from visualization to exit")

    # [doc-stag-live-plot-intensity]
    # establish sensor connection
    with closing(client.Scans.stream(hostname, lidar_port,
                                     complete=False)) as stream:
        show = True
        while show:
            for scan in stream:
                # uncomment if you'd like to see frame id printed
                # print("frame id: {} ".format(scan.frame_id))
                signal = client.destagger(
                    stream.metadata, scan.field(client.ChanField.INTENSITY))
                signal = (signal / np.max(signal) * 255).astype(np.uint8)
                cv2.imshow("scaled intensity", signal)
                # [doc-etag-live-plot-intensity]
                key = cv2.waitKey(1) & 0xFF
                # 27 is esc
                if key == 27:
                    show = False
                    break
        cv2.destroyAllWindows()


def display_xyz_points(hostname: str, lidar_port: int = 7502) -> None:
    """Display range from a single scan as 3D points

    Args:
        hostname: hostname of the sensor
        lidar_port: UDP port to listen on for lidar data
    """
    import matplotlib.pyplot as plt  # type: ignore

    # get single scan
    metadata, sample = client.Scans.sample(hostname, 1, lidar_port)
    scan = next(sample)[0]

    # set up figure
    plt.figure()
    ax = plt.axes(projection='3d')
    r = 3
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r, r])

    plt.title("3D Points from {}".format(hostname))

    # [doc-stag-plot-xyz-points]
    # transform data to 3d points and graph
    xyzlut = client.XYZLut(metadata)
    xyz = xyzlut(scan)

    [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
    ax.scatter(x, y, z, c=z / max(z), s=0.2)
    # [doc-etag-plot-xyz-points]
    plt.show()


def write_xyz_to_csv(hostname: str,
                     lidar_port: int = 7502,
                     cloud_prefix: str = 'xyz',
                     n_scans: int = 5) -> None:
    """Write xyz sample from live sensor to csv

    Args:
        hostname: hostname of the sensor
        lidar_port: UDP port to listen on for lidar data
        cloud_prefix: filename prefix for written csvs
        n_scans: number of scans to write
    """
    metadata, sample = client.Scans.sample(hostname, n_scans, lidar_port)
    h = metadata.format.pixels_per_column
    w = metadata.format.columns_per_frame
    xyzlut = client.XYZLut(metadata)

    for count, scan in enumerate(next(sample)):
        out_name = "{}_{}.txt".format(cloud_prefix, count)
        print("writing {}..".format(out_name))
        np.savetxt(out_name, xyzlut(scan).reshape(h * w, 3), delimiter=" ")


def plot_imu_z_acc_over_time(hostname: str,
                             lidar_port: int = 7502,
                             imu_port: int = 7503,
                             n_seconds: int = 10) -> None:
    """Plot the z acceleration from the IMU over time

    Args:
        hostname: hostname of the sensor
        imu_port: UDP port to listen on for imu data
        n_seconds: seconds of time to take a sample over
    """
    import matplotlib.pyplot as plt  # type: ignore

    # connect to sensor and get imu packets within n_seconds
    source = client.Sensor(hostname, lidar_port, imu_port, buf_size=640)
    with closing(source):
        ts, z_accel = zip(*[(p.sys_ts, p.accel[2])
                            for p in time_limited(n_seconds, source)
                            if isinstance(p, client.ImuPacket)])

    # initialize plot
    fig, ax = plt.subplots(figsize=(12.0, 2))
    ax.plot(ts, z_accel)

    plt.title("Z Accel from IMU over {} Seconds".format(n_seconds))
    ax.set_xticks(np.arange(min(ts), max(ts), step=((max(ts) - min(ts)) / 5)))
    # add end ticker to x axis
    ax.set_xticks(list(ax.get_xticks()) + [max(ts)])

    ax.set_xlim([min(ts), max(ts)])
    ax.set_ylabel("z accel")
    ax.set_xlabel("timestamp (ns)")

    ax.ticklabel_format(useOffset=False, style="plain")
    plt.show()


def main() -> None:
    examples = {
        "configure-sensor": configure_sensor_params,
        "get-metadata": get_metadata,
        "plot-range-image": display_range_2d,
        "plot-all-channels": display_all_2d,
        "plot-xyz-points": display_xyz_points,
        "plot-imu-z-accel": plot_imu_z_acc_over_time,
        "live-plot-intensity": display_intensity_live,
        "write-xyz-to-csv": write_xyz_to_csv,
    }

    description = "Ouster Python SDK examples. The EXAMPLE must be one of:\n  " + str.join(
        '\n  ', examples.keys())

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('hostname',
                        metavar='HOSTNAME',
                        type=str,
                        help='Sensor hostname, e.g. "os-122033000087"')
    parser.add_argument('example',
                        metavar='EXAMPLE',
                        type=str,
                        help='Name of the example to run')

    args = parser.parse_args()

    try:
        example = examples[args.example]
    except KeyError:
        print(f"No such example: {args.example}")
        exit(1)

    print(f"example: {args.example}")
    example(args.hostname)  # type: ignore


if __name__ == "__main__":
    main()