/**
 * @file
 * @brief ouster_pyclient python module
 *
 * Note: the type annotations in `sensor.pyi` need to be updated whenever this
 * file changes. See the mypy documentation for details.
 */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "ouster/client.h"
#include "ouster/compat.h"
#include "ouster/packet.h"
#include "ouster/types.h"

namespace py = pybind11;

/*
 * Check that buffer is a 1-d byte array of size > bound and return an internal
 * pointer to the data for writing. Check is strictly greater to account for the
 * extra byte required to determine if a datagram is bigger than expected.
 */
inline uint8_t* getptr(size_t bound, py::buffer& buf) {
    auto info = buf.request();
    if (info.format != py::format_descriptor<uint8_t>::format() ||
        info.ndim != 1 || info.size <= bound) {
        throw std::invalid_argument(
            "Incompatible argument: expected a bytearray of size > " +
            std::to_string(bound));
    }
    return (uint8_t*)info.ptr;
}


/*
 * Pybind11 can't deal with opaque pointers directly, so we have to wrap it in a
 * another struct
 */
struct pyclient {
    std::shared_ptr<ouster::sensor::client> val;
};


PYBIND11_MODULE(_sensor, m) {
    using ouster::sensor::data_format;
    using ouster::sensor::packet_format;
    using ouster::sensor::sensor_info;

    using namespace ouster;

    m.doc() = "ouster.client._sensor";
    socket_init();

    // clang-format off

    // Client Handle
    py::class_<pyclient>(m, "Client");

    // Version Info
    py::class_<util::version>(m, "Version")
        .def(py::init<>())
        .def("__eq__", [](const util::version& u, const util::version& v) { return u == v; })
        .def("__lt__", [](const util::version& u, const util::version& v) { return u < v; })
        .def("__le__", [](const util::version& u, const util::version& v) { return u <= v; })
        .def("__str__", [](const util::version& u) { return to_string(u); })
        .def_readwrite("major", &util::version::major)
        .def_readwrite("minor", &util::version::minor)
        .def_readwrite("patch", &util::version::patch);

    m.def("version_of_string", &util::version_of_string);

    m.attr("invalid_version") = util::invalid_version;

    m.attr("min_version") = sensor::min_version;

    // Data Format
    py::class_<data_format>(m, "DataFormat")
        .def_readwrite("pixels_per_column", &data_format::pixels_per_column)
        .def_readwrite("columns_per_packet", &data_format::columns_per_packet)
        .def_readwrite("columns_per_frame", &data_format::columns_per_frame)
        .def_readwrite("pixel_shift_by_row", &data_format::pixel_shift_by_row);

    // Packet Format
    py::class_<packet_format>(m, "PacketFormat")
        .def_readonly("lidar_packet_size", &packet_format::lidar_packet_size)
        .def_readonly("imu_packet_size", &packet_format::imu_packet_size)
        .def_readonly("columns_per_packet", &packet_format::columns_per_packet)
        .def_readonly("pixels_per_column", &packet_format::pixels_per_column)
        .def_readonly("encoder_ticks_per_rev", &packet_format::encoder_ticks_per_rev)
        .def("imu_sys_ts", [](packet_format& pf, py::buffer buf) { return pf.imu_sys_ts(getptr(pf.imu_packet_size, buf)); })
        .def("imu_accel_ts", [](packet_format& pf, py::buffer buf) { return pf.imu_accel_ts(getptr(pf.imu_packet_size, buf)); })
        .def("imu_gyro_ts", [](packet_format& pf, py::buffer buf) { return pf.imu_gyro_ts(getptr(pf.imu_packet_size, buf)); })
        .def("imu_av_x", [](packet_format& pf, py::buffer buf) { return pf.imu_av_x(getptr(pf.imu_packet_size, buf)); })
        .def("imu_av_y", [](packet_format& pf, py::buffer buf) { return pf.imu_av_y(getptr(pf.imu_packet_size, buf)); })
        .def("imu_av_z", [](packet_format& pf, py::buffer buf) { return pf.imu_av_z(getptr(pf.imu_packet_size, buf)); })
        .def("imu_la_x", [](packet_format& pf, py::buffer buf) { return pf.imu_la_x(getptr(pf.imu_packet_size, buf)); })
        .def("imu_la_y", [](packet_format& pf, py::buffer buf) { return pf.imu_la_y(getptr(pf.imu_packet_size, buf)); })
        .def("imu_la_z", [](packet_format& pf, py::buffer buf) { return pf.imu_la_z(getptr(pf.imu_packet_size, buf)); });

    // Sensor Info
    py::class_<sensor_info>(m, "SensorInfo")
        .def(py::init<>())
        .def_readwrite("hostname", &sensor_info::name)
        .def_readwrite("sn", &sensor_info::sn)
        .def_readwrite("fw_rev", &sensor_info::fw_rev)
        .def_readwrite("mode", &sensor_info::mode)
        .def_readwrite("prod_line", &sensor_info::prod_line)
        .def_readwrite("format", &sensor_info::format)
        .def_readwrite("beam_azimuth_angles", &sensor_info::beam_azimuth_angles)
        .def_readwrite("beam_altitude_angles", &sensor_info::beam_altitude_angles)
        .def_readwrite("imu_to_sensor_transform", &sensor_info::imu_to_sensor_transform)
        .def_readwrite("lidar_to_sensor_transform", &sensor_info::lidar_to_sensor_transform)
        .def_readwrite("extrinsic", &sensor_info::extrinsic)
        .def("__str__", [](const sensor_info& i) { return to_string(i); });

    // Get metadata
    m.def("default_sensor_info", &sensor::default_sensor_info, py::arg("mode"));

    m.def("get_metadata",
          [](pyclient& cli, int timeout_sec) {
              return sensor::get_metadata(*cli.val, timeout_sec);
          },
          py::arg("cli"),
          py::arg("timeout_sec") = 30);

    m.def("parse_metadata", &sensor::parse_metadata);

    m.def("get_format", &sensor::get_format);

    // Lidar Mode
    py::enum_<sensor::lidar_mode>(m, "LidarMode")
        .value("MODE_512x10", sensor::lidar_mode::MODE_512x10)
        .value("MODE_512x20", sensor::lidar_mode::MODE_512x20)
        .value("MODE_1024x10", sensor::lidar_mode::MODE_1024x10)
        .value("MODE_1024x20", sensor::lidar_mode::MODE_1024x20)
        .value("MODE_2048x10", sensor::lidar_mode::MODE_2048x10)
        .export_values()
        .def("__str__", [](const sensor::lidar_mode& u) { return to_string(u); });

    m.def("lidar_mode_of_string", &sensor::lidar_mode_of_string);

    // Timestamp Mode
    py::enum_<sensor::timestamp_mode>(m, "TimestampMode")
        .value("TIME_FROM_INTERNAL_OSC", sensor::timestamp_mode::TIME_FROM_INTERNAL_OSC)
        .value("TIME_FROM_SYNC_PULSE_IN", sensor::timestamp_mode::TIME_FROM_SYNC_PULSE_IN)
        .value("TIME_FROM_PTP_1588", sensor::timestamp_mode::TIME_FROM_PTP_1588)
        .export_values()
        .def("__str__", [](const sensor::timestamp_mode& u) { return to_string(u); });

    m.def("timestamp_mode_of_string", &sensor::timestamp_mode_of_string);

    // Client State
    py::enum_<sensor::client_state>(m, "ClientState", py::arithmetic())
        .value("TIMEOUT", sensor::client_state::TIMEOUT)
        .value("ERROR", sensor::client_state::CLIENT_ERROR)
        .value("LIDAR_DATA", sensor::client_state::LIDAR_DATA)
        .value("IMU_DATA", sensor::client_state::IMU_DATA)
        .value("EXIT", sensor::client_state::EXIT)
        .export_values();

    // Sensor API
    m.def("n_cols_of_lidar_mode", &sensor::n_cols_of_lidar_mode);

    m.def("init_client",
          [](const std::string& hostname, int lidar_port, int imu_port) -> py::object {
              auto cli = sensor::init_client(hostname, lidar_port, imu_port);
              return cli ? py::cast(pyclient{cli}) : py::none{};
          },
          py::arg("hostname") = "",
          py::arg("lidar_port") = 7502,
          py::arg("imu_port") = 7503);

    m.def("init_client",
          [](const std::string& hostname, const std::string& udp_dest_host,
             sensor::lidar_mode mode, sensor::timestamp_mode ts_mode,
             int lidar_port, int imu_port, int timeout_sec) -> py::object {
              auto cli = sensor::init_client(hostname, udp_dest_host, mode, ts_mode,
                                             lidar_port, imu_port);
              return cli ? py::cast(pyclient{cli}) : py::none{};
          },
          py::arg("hostname"),
          py::arg("udp_dest_host"),
          py::arg("mode") = sensor::lidar_mode::MODE_1024x10,
          py::arg("ts_mode") = sensor::timestamp_mode::TIME_FROM_INTERNAL_OSC,
          py::arg("lidar_port") = 0, py::arg("imu_port") = 0,
          py::arg("timeout_sec") = 30);

    m.def("poll_client",
          [](const pyclient& cli, const int timeout_sec) -> sensor::client_state {
              py::gil_scoped_release release;
              return sensor::poll_client(*cli.val, timeout_sec);
          },
          py::arg("cli"),
          py::arg("timeout_sec") = 1);

    m.def("read_lidar_packet", [](const pyclient& cli, py::buffer buf,
                                  const packet_format& pf) {
        return sensor::read_lidar_packet(*cli.val, getptr(pf.lidar_packet_size, buf), pf);
    });

    m.def("read_imu_packet", [](const pyclient& cli, py::buffer buf,
                                const packet_format& pf) {
        return sensor::read_imu_packet(*cli.val, getptr(pf.imu_packet_size, buf), pf);
    });

    m.add_object("_cleanup", py::capsule([]() { socket_quit(); }));

    m.attr("__version__") = VERSION_INFO;

    // clang-format on
}