/**
 * Copyright (c) 2018, Ouster, Inc.
 * All rights reserved.
 */

#include "ouster/core/data_format.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ouster/core/defaults.h"
#include "ouster/core/impl/table.h"
#include "ouster/core/sensor_config.h"

using nonstd::optional;
using ouster::impl::lookup;
using ouster::impl::rlookup;
using ouster::impl::Table;

namespace ouster {
namespace sdk {
namespace core {

namespace impl {

OUSTER_API_VAR Table<UDPProfileLidar, const char*, MAX_NUM_PROFILES> udp_profile_lidar_strings{{
    {UDPProfileLidar::UNKNOWN, "UNKNOWN"},
    {UDPProfileLidar::LEGACY, "LEGACY"},
    {UDPProfileLidar::RNG19_RFL8_SIG16_NIR16_DUAL, "RNG19_RFL8_SIG16_NIR16_DUAL"},
    {UDPProfileLidar::RNG19_RFL8_SIG16_NIR16, "RNG19_RFL8_SIG16_NIR16"},
    {UDPProfileLidar::RNG15_RFL8_NIR8, "RNG15_RFL8_NIR8"},
    {UDPProfileLidar::FIVE_WORD_PIXEL, "FIVE_WORD_PIXEL"},
    {UDPProfileLidar::FUSA_RNG15_RFL8_NIR8_DUAL, "FUSA_RNG15_RFL8_NIR8_DUAL"},
    {UDPProfileLidar::RNG15_RFL8_NIR8_DUAL, "RNG15_RFL8_NIR8_DUAL"},
    {UDPProfileLidar::RNG15_RFL8_NIR8_ZONE16, "RNG15_RFL8_NIR8_ZONE16"},
    {UDPProfileLidar::RNG19_RFL8_SIG16_NIR16_ZONE16, "RNG19_RFL8_SIG16_NIR16_ZONE16"},
    {UDPProfileLidar::RNG15_RFL8_WIN8, "RNG15_RFL8_WIN8"},
    {UDPProfileLidar::RNG19_RFL8_SIG16_ZONE16_DUAL, "RNG19_RFL8_SIG16_ZONE16_DUAL"},
    {UDPProfileLidar::RNG19_RFL8_SIG16_NIR16_RGB16, "RNG19_RFL8_SIG16_NIR16_RGB16"},
    {UDPProfileLidar::RNG19_RFL8_SIG16_NIR16_RGB16_DUAL, "RNG19_RFL8_SIG16_NIR16_RGB16_DUAL"},
    {UDPProfileLidar::OFF, "OFF"},
}};

OUSTER_API_VAR Table<UDPProfileIMU, const char*, 3> udp_profile_imu_strings{{
    {UDPProfileIMU::LEGACY, "LEGACY"},
    {UDPProfileIMU::ACCEL32_GYRO32_NMEA, "ACCEL32_GYRO32_NMEA"},
    {UDPProfileIMU::OFF, "OFF"},
}};

OUSTER_API_VAR Table<HeaderType, const char*, 2> udp_profile_type_strings{{
    {HeaderType::STANDARD, "STANDARD"},
    {HeaderType::FUSA, "FUSA"},
}};

}  // namespace impl

bool operator==(const DataFormat& lhs, const DataFormat& rhs) {
    return (lhs.pixels_per_column == rhs.pixels_per_column &&
            lhs.columns_per_packet == rhs.columns_per_packet &&
            lhs.columns_per_frame == rhs.columns_per_frame &&
            lhs.imu_measurements_per_packet == rhs.imu_measurements_per_packet &&
            lhs.pixel_shift_by_row == rhs.pixel_shift_by_row &&
            lhs.column_window == rhs.column_window &&
            lhs.udp_profile_lidar == rhs.udp_profile_lidar && lhs.header_type == rhs.header_type &&
            lhs.udp_profile_imu == rhs.udp_profile_imu && lhs.fps == rhs.fps &&
            lhs.zone_monitoring_enabled == rhs.zone_monitoring_enabled);
}

bool operator!=(const DataFormat& lhs, const DataFormat& rhs) {
    return !(lhs == rhs);
}

static ColumnWindow default_column_window(uint32_t columns_per_frame) {
    return {0, columns_per_frame - 1};
}

DataFormat default_data_format(LidarMode mode) {
    auto repeat = [](int name, const std::vector<int>& value) {
        std::vector<int> res{};
        for (int i = 0; i < name; i++) {
            res.insert(res.end(), value.begin(), value.end());
        }
        return res;
    };

    uint32_t pixels_per_column = 64;
    uint32_t columns_per_packet = DEFAULT_COLUMNS_PER_PACKET;
    uint32_t columns_per_frame = mode.columns;
    ColumnWindow column_window = default_column_window(columns_per_frame);
    uint32_t imu_measurements_per_packet = 0;
    uint32_t imu_packets_per_frame = 0;

    std::vector<int> offset;
    switch (columns_per_frame) {
        case 512:
            offset = repeat(16, {9, 6, 3, 0});
            break;
        case 1024:
            offset = repeat(16, {18, 12, 6, 0});
            break;
        case 2048:
            offset = repeat(16, {36, 24, 12, 0});
            break;
        case 4096:
            offset = repeat(16, {72, 48, 24, 0});
            break;
        default:
            throw std::invalid_argument{"default_data_format"};
    }

    return {pixels_per_column,
            columns_per_packet,
            columns_per_frame,
            imu_measurements_per_packet,
            imu_packets_per_frame,
            offset,
            column_window,
            UDPProfileLidar::LEGACY,
            UDPProfileIMU::LEGACY,
            HeaderType::STANDARD,
            static_cast<uint16_t>(mode.fps),
            false};
}

int DataFormat::valid_columns_per_frame() const {
    auto start = column_window.first;
    auto end = column_window.second;

    if (start <= end) {
        return end - start + 1;
    } else {
        return static_cast<int>(end + (columns_per_frame - start) + 1);
    }
}

int DataFormat::lidar_packets_per_frame() const {
    if (udp_profile_lidar == UDPProfileLidar::OFF) {
        return 0;
    }

    int start_packet = static_cast<int>(column_window.first / columns_per_packet);
    int end_packet = static_cast<int>(column_window.second / columns_per_packet);
    if (column_window.second < column_window.first) {
        // the valid azimuth window wraps through 0
        // Determine the number of packets for a full frame with no window
        int max_packets = static_cast<int>(columns_per_frame / columns_per_packet) +
                          (((columns_per_frame % columns_per_packet) != 0u) ? 1 : 0);
        // We expect to get [start, max_packets] and [0, end] packet indexes
        int expected_packets = (max_packets - start_packet) + 1 + end_packet;
        // If start and end packets are the same, we have every packet
        if (start_packet == end_packet) {
            return max_packets;
        }
        return expected_packets;
    } else {
        // no wrapping of azimuth the window through 0
        return end_packet - start_packet + 1;
    }
}

uint32_t DataFormat::max_frame_id() const {
    if (header_type == HeaderType::FUSA && udp_profile_lidar != UDPProfileLidar::LEGACY) {
        return std::numeric_limits<uint32_t>::max();
    }
    return std::numeric_limits<uint16_t>::max();
}

std::string to_string(UDPProfileLidar profile) {
    auto res = lookup(impl::udp_profile_lidar_strings, profile);
    return res ? res.value() : "UNKNOWN";
}

optional<UDPProfileLidar> udp_profile_lidar_of_string(const std::string& str) {
    return rlookup(impl::udp_profile_lidar_strings, str.c_str());
}

std::string to_string(UDPProfileIMU profile) {
    auto res = lookup(impl::udp_profile_imu_strings, profile);
    return res ? res.value() : "UNKNOWN";
}

optional<UDPProfileIMU> udp_profile_imu_of_string(const std::string& str) {
    return rlookup(impl::udp_profile_imu_strings, str.c_str());
}

std::string to_string(HeaderType profile) {
    auto res = lookup(impl::udp_profile_type_strings, profile);
    return res ? res.value() : "UNKNOWN";
}

optional<HeaderType> udp_profile_type_of_string(const std::string& str) {
    return rlookup(impl::udp_profile_type_strings, str.c_str());
}

}  // namespace core
}  // namespace sdk
}  // namespace ouster
