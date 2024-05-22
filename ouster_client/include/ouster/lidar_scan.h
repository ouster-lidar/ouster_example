/**
 * Copyright (c) 2018, Ouster, Inc.  All rights reserved.  @file
 * @brief Holds lidar data by field in row-major order
 */

#pragma once

#include <Eigen/Core>
#include <chrono>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ouster/defaults.h"
#include "ouster/types.h"

namespace ouster {

// forward declarations
namespace impl {
struct FieldSlot;
}

/**
 * Alias for the lidar scan field types
 */
using LidarScanFieldTypes =
    std::vector<std::pair<sensor::ChanField, sensor::ChanFieldType>>;

/**
 * Data structure for efficient operations on aggregated lidar data.
 *
 * Stores each field (range, intensity, etc.) contiguously as a H x W block of
 * 4-byte unsigned integers, where H is the number of beams and W is the
 * horizontal resolution (e.g. 512, 1024, 2048).
 *
 * Note: this is the "staggered" representation where each column corresponds
 * to a single measurement in time. Use the destagger() function to create an
 * image where columns correspond to a single azimuth angle.
 */
class LidarScan {
   public:
    template <typename T>
    using Header = Eigen::Array<T, Eigen::Dynamic, 1>;  ///< Header typedef

    /** XYZ coordinates with dimensions arranged contiguously in columns. */
    using Points = Eigen::Array<double, Eigen::Dynamic, 3>;

   private:
    Header<uint64_t> timestamp_;
    Header<uint64_t> packet_timestamp_;
    Header<uint16_t> measurement_id_;
    Header<uint32_t> status_;
    std::vector<mat4d> pose_;
    std::map<sensor::ChanField, impl::FieldSlot> fields_;
    LidarScanFieldTypes field_types_;

    LidarScan(size_t w, size_t h, LidarScanFieldTypes field_types,
              size_t columns_per_packet);

   public:
    /**
     * Pointer offsets to deal with strides.
     *
     * @warning Members variables: use with caution, some of these will become
     * private.
     */
    std::ptrdiff_t w{0};

    /**
     * Pointer offsets to deal with strides.
     *
     * @warning Members variables: use with caution, some of these will become
     * private.
     */
    std::ptrdiff_t h{0};

    /**
     * Frame status - information from the packet header which corresponds to a
     * frame
     *
     * @warning Member variables: use with caution, some of these will become
     * private.
     */
    uint64_t frame_status{0};

    /**
     * The current frame ID.
     *
     * @warning Members variables: use with caution, some of these will become
     * private.
     */
    int64_t frame_id{-1};

    using FieldIter =
        decltype(field_types_)::const_iterator;  ///< An STL Iterator of the
                                                 ///< field types

    /** The default constructor creates an invalid 0 x 0 scan. */
    LidarScan();

    /**
     * Initialize a scan with fields configured for the LEGACY udp profile.
     *
     * @param[in] w horizontal resolution, i.e. the number of measurements per
     *              scan.
     * @param[in] h vertical resolution, i.e. the number of channels.
     *
     * Note, the number of columns per packet is set to the default
     * (DEFAULT_COLUMNS_PER_PACKET).
     */
    LidarScan(size_t w, size_t h);

    /**
     * Initialize a scan with the default fields for a particular udp profile.
     *
     * @param[in] w horizontal resolution, i.e. the number of measurements per
     *              scan.
     * @param[in] h vertical resolution, i.e. the number of channels.
     * @param[in] profile udp profile.
     * @param[in] columns_per_packet The number of columns per packet,
     *                               this argument is optional.
     */
    LidarScan(size_t w, size_t h, sensor::UDPProfileLidar profile,
              size_t columns_per_packet = DEFAULT_COLUMNS_PER_PACKET);

    /**
     * Initialize a scan with a custom set of fields.
     *
     * @tparam Iterator A standard template iterator for the custom fields.
     *
     * @param[in] w horizontal resoulution, i.e. the number of measurements per
     *              scan.
     * @param[in] h vertical resolution, i.e. the number of channels.
     * @param[in] begin begin iterator of pairs of channel fields and types.
     * @param[in] end end iterator of pairs of channel fields and types.
     * @param[in] columns_per_packet The number of columns per packet,
     *                               this argument is optional.
     */
    template <typename Iterator>
    LidarScan(size_t w, size_t h, Iterator begin, Iterator end,
              size_t columns_per_packet = DEFAULT_COLUMNS_PER_PACKET)
        : LidarScan(w, h, {begin, end}, columns_per_packet){}

    /**
     * Initialize a lidar scan from another lidar scan.
     *
     * @param[in] other The other lidar scan to initialize from.
     */
    LidarScan(const LidarScan& other);

    /**
     * Initialize a lidar scan from another with only the indicated fields.
     * Casts, zero pads or removes fields from the original scan if necessary.
     *
     * @param[in] other The other lidar scan to initialize from.
     * @param[in] fields Fields to have in new lidar scan.
     */
    LidarScan(const LidarScan& other, const LidarScanFieldTypes& fields);

    /** @copydoc LidarScan(const LidarScan& other) */
    LidarScan(LidarScan&& other);

    /**
     * Copy.
     *
     * @param[in] other The lidar scan to copy from.
     */
    LidarScan& operator=(const LidarScan& other);

    /**
     * Copy via Move semantic.
     *
     * @param[in] other The lidar scan to copy from.
     */
    LidarScan& operator=(LidarScan&& other);

    /**
     * Lidar scan destructor.
     */
    ~LidarScan();

    /**
     * Get frame shot limiting status
     */
    sensor::ShotLimitingStatus shot_limiting() const;

    /**
     * Get frame thermal shutdown status
     */
    sensor::ThermalShutdownStatus thermal_shutdown() const;

    /**
     * @defgroup ClientLidarScanField Access fields in a lidar scan
     * Access a lidar data field.
     *
     * @throw std::invalid_argument if T does not match the runtime field type.
     *
     * @tparam T The type parameter T must match the dynamic type of the field.
     * See the constructor documentation for expected field types or query
     * dynamically for generic operations.
     *
     * @param[in] f the field to view.
     *
     * @return a view of the field data.
     */

    /**
     * @copydoc ClientLidarScanField
     */
    template <typename T = uint32_t,
              typename std::enable_if<std::is_unsigned<T>::value, T>::type = 0>
    Eigen::Ref<img_t<T>> field(sensor::ChanField f);

    /**
     * @copydoc ClientLidarScanField
     */
    template <typename T = uint32_t,
              typename std::enable_if<std::is_unsigned<T>::value, T>::type = 0>
    Eigen::Ref<const img_t<T>> field(sensor::ChanField f) const;

    /**
     * Get the type of the specified field.
     *
     * @param[in] f the field to query.
     *
     * @return the type tag associated with the field.
     */
    sensor::ChanFieldType field_type(sensor::ChanField f) const;

    /** A const forward iterator over field / type pairs. */
    FieldIter begin() const;

    /** @copydoc begin() */
    FieldIter end() const;

    /**
     * Access the measurement timestamp headers.
     *
     * @return a view of timestamp as a w-element vector.
     */
    Eigen::Ref<Header<uint64_t>> timestamp();

    /**
     * @copydoc timestamp()
     */
    Eigen::Ref<const Header<uint64_t>> timestamp() const;

    /**
     * Access the packet timestamp headers (usually host time).
     *
     * @return a view of timestamp as a w-element vector.
     */
    Eigen::Ref<Header<uint64_t>> packet_timestamp();

    /**
     * Access the host timestamp headers (usually host time).
     *
     * @return a view of timestamp as a w-element vector.
     */
    Eigen::Ref<const Header<uint64_t>> packet_timestamp() const;

    /**
     * Return the first valid packet timestamp
     *
     * @return the first valid packet timestamp, 0 if none available
     */
    uint64_t get_first_valid_packet_timestamp() const;

    /**
     * Access the measurement id headers.
     *
     * @return a view of measurement ids as a w-element vector.
     */
    Eigen::Ref<Header<uint16_t>> measurement_id();

    /** @copydoc measurement_id() */
    Eigen::Ref<const Header<uint16_t>> measurement_id() const;

    /**
     * Access the measurement status headers.
     *
     * @return a view of measurement statuses as a w-element vector.
     */
    Eigen::Ref<Header<uint32_t>> status();

    /** @copydoc status() */
    Eigen::Ref<const Header<uint32_t>> status() const;

    /**
     * Access the vector of poses (per each timestamp).
     *
     * @return a reference to vector with poses (4x4) homogeneous.
     */
    std::vector<mat4d>& pose();
    /** @copydoc pose() */
    const std::vector<mat4d>& pose() const;

    /**
     * Assess completeness of scan.
     * @param[in] window The column window to use for validity assessment
     * @return whether all columns within given column window were valid
     */
    bool complete(sensor::ColumnWindow window) const;

    friend bool operator==(const LidarScan& a, const LidarScan& b);
};

/**
 * Get string representation of lidar scan field types.
 *
 * @param[in] field_types The field types to get the string representation of.
 *
 * @return string representation of the lidar scan field types.
 */
std::string to_string(const LidarScanFieldTypes& field_types);

/**
 * Get the lidar scan field types from a lidar scan
 *
 * @param[in] ls The lidar scan to get the lidar scan field types from.
 *
 * @return The lidar scan field types
 */
LidarScanFieldTypes get_field_types(const LidarScan& ls);

/**
 * Get the lidar scan field types from lidar profile
 *
 * @param[in] udp_profile_lidar lidar profile
 *
 * @return The lidar scan field types
 */
LidarScanFieldTypes get_field_types(sensor::UDPProfileLidar udp_profile_lidar);

/**
 * Get the lidar scan field types from sensor info
 *
 * @param[in] info The sensor info to get the lidar scan field types from.
 *
 * @return The lidar scan field types
 */
LidarScanFieldTypes get_field_types(const sensor::sensor_info& info);

/**
 * Get string representation of a lidar scan.
 *
 * @param[in] ls The lidar scan to get the string representation of.
 *
 * @return string representation of the lidar scan.
 */
std::string to_string(const LidarScan& ls);

/** \defgroup ouster_client_lidar_scan_operators Ouster Client lidar_scan.h
 * Operators
 * @{
 */

/**
 * Equality for scans.
 *
 * @param[in] a The first scan to compare.
 * @param[in] b The second scan to compare.
 *
 * @return if a == b.
 */
bool operator==(const LidarScan& a, const LidarScan& b);

/**
 * NOT Equality for scans.
 *
 * @param[in] a The first scan to compare.
 * @param[in] b The second scan to compare.
 *
 * @return if a != b.
 */
inline bool operator!=(const LidarScan& a, const LidarScan& b) {
    return !(a == b);
}
/** @}*/

/** Lookup table of beam directions and offsets. */
struct XYZLut {
    LidarScan::Points direction;  ///< Lookup table of beam directions
    LidarScan::Points offset;     ///< Lookup table of beam offsets
};

/**
 * Generate a set of lookup tables useful for computing Cartesian coordinates
 * from ranges.
 *
 * The lookup tables are:
 * - direction: a matrix of unit vectors pointing radially outwards.
 * - offset: a matrix of offsets dependent on beam origin distance from lidar
 *           origin.
 *
 * Each table is an n x 3 array of doubles stored in column-major order where
 * each row corresponds to the nth point in a lidar scan, with 0 <= n < h*w.
 *
 * Projections to XYZ made with this XYZLut will be in the coordinate frame
 * defined by transform*beam_to_lidar_transform.
 *
 * @param[in] w number of columns in the lidar scan. e.g. 512, 1024, or 2048.
 * @param[in] h number of rows in the lidar scan.
 * @param[in] range_unit the unit, in meters, of the range,  e.g.
 * sensor::range_unit.
 * @param[in] beam_to_lidar_transform transform between beams and
 * lidar origin. Translation portion is in millimeters.
 * @param[in] transform additional transformation to apply to resulting points.
 * @param[in] azimuth_angles_deg azimuth offsets in degrees for each of h beams.
 * @param[in] altitude_angles_deg altitude in degrees for each of h beams.
 *
 * @return xyz direction and offset vectors for each point in the lidar scan.
 */
XYZLut make_xyz_lut(size_t w, size_t h, double range_unit,
                    const mat4d& beam_to_lidar_transform,
                    const mat4d& transform,
                    const std::vector<double>& azimuth_angles_deg,
                    const std::vector<double>& altitude_angles_deg);

/**
 * Convenient overload that uses parameters from the supplied sensor_info.
 * Projections to XYZ made with this XYZLut will be in the sensor coordinate
 * frame defined in the sensor documentation.
 *
 * @param[in] sensor metadata returned from the client.
 *
 * @return xyz direction and offset vectors for each point in the lidar scan.
 */
inline XYZLut make_xyz_lut(const sensor::sensor_info& sensor) {
    return make_xyz_lut(
        sensor.format.columns_per_frame, sensor.format.pixels_per_column,
        sensor::range_unit, sensor.beam_to_lidar_transform,
        sensor.lidar_to_sensor_transform, sensor.beam_azimuth_angles,
        sensor.beam_altitude_angles);
}

/** \defgroup ouster_client_lidar_scan_cartesian Ouster Client lidar_scan.h
 * XYZLut related items.
 * @{
 */
/**
 * Convert LidarScan to Cartesian points.
 *
 * @param[in] scan a LidarScan.
 * @param[in] lut lookup tables generated by make_xyz_lut.
 *
 * @return Cartesian points where ith row is a 3D point which corresponds
 *         to ith pixel in LidarScan where i = row * w + col.
 */
LidarScan::Points cartesian(const LidarScan& scan, const XYZLut& lut);

/**
 * Convert a staggered range image to Cartesian points.
 *
 * @param[in] range a range image in the same format as the RANGE field of a
 * LidarScan.
 * @param[in] lut lookup tables generated by make_xyz_lut.
 *
 * @return Cartesian points where ith row is a 3D point which corresponds
 *         to ith pixel in LidarScan where i = row * w + col.
 */
LidarScan::Points cartesian(const Eigen::Ref<const img_t<uint32_t>>& range,
                            const XYZLut& lut);
/** @}*/

/** \defgroup ouster_client_destagger Ouster Client lidar_scan.h
 * @{
 */
/**
 * Generate a destaggered version of a channel field.
 *
 * In the default staggered representation, each column corresponds to a single
 * timestamp. In the destaggered representation, each column corresponds to a
 * single azimuth angle, compensating for the azimuth offset of each beam.
 *
 * Destaggering is used for visualizing lidar data as an image or for algorithms
 * that exploit the structure of the lidar data, such as beam_uniformity in
 * ouster_viz, or computer vision algorithms.
 *
 * @tparam T the datatype of the channel field.
 *
 * @param[in] img the channel field.
 * @param[in] pixel_shift_by_row offsets, usually queried from the sensor.
 * @param[in] inverse perform the inverse operation.
 *
 * @return destaggered version of the image.
 */
template <typename T>
inline img_t<T> destagger(const Eigen::Ref<const img_t<T>>& img,
                          const std::vector<int>& pixel_shift_by_row,
                          bool inverse = false);

/**
 * Generate a staggered version of a channel field.
 *
 * @tparam T the datatype of the channel field.
 *
 * @param[in] img the channel field.
 * @param[in] pixel_shift_by_row offsets, usually queried from the sensor.
 *
 * @return staggered version of the image.
 */
template <typename T>
inline img_t<T> stagger(const Eigen::Ref<const img_t<T>>& img,
                        const std::vector<int>& pixel_shift_by_row) {
    return destagger(img, pixel_shift_by_row, true);
}
/** @}*/
/**
 * Parse lidar packets into a LidarScan.
 *
 * Make a function that batches a single scan (revolution) of data to a
 * LidarScan.
 */
class ScanBatcher {
    std::ptrdiff_t w;
    std::ptrdiff_t h;
    uint16_t next_valid_m_id;
    uint16_t next_headers_m_id;
    uint16_t next_valid_packet_id;
    std::vector<uint8_t> cache;
    uint64_t cache_packet_ts;
    bool cached_packet = false;

    void _parse_by_col(const uint8_t* packet_buf, LidarScan& ls);
    void _parse_by_block(const uint8_t* packet_buf, LidarScan& ls);

   public:
    sensor::packet_format pf;  ///< The packet format object used for decoding

    /**
     * Create a batcher given information about the scan and packet format.
     *
     * @param[in] w number of columns in the lidar scan. One of 512, 1024, or
     * 2048.
     * @param[in] pf expected format of the incoming packets used for parsing.
     */
    ScanBatcher(size_t w, const sensor::packet_format& pf);

    /**
     * Create a batcher given information about the scan and packet format.
     *
     * @param[in] info sensor metadata returned from the client.
     */
    ScanBatcher(const sensor::sensor_info& info);

    /**
     * Add a packet to the scan.
     * @deprecated this method is deprecated in favor of one that accepts a
     * reference to a LidarPacket.
     *
     * @param[in] packet_buf a buffer containing raw bytes from a lidar packet.
     * @param[in] ls lidar scan to populate.
     *
     * @return true when the provided lidar scan is ready to use.
     */
    [[deprecated(
        "Use ScanBatcher::operator(LidarPacket&, LidarScan&) instead.")]] bool
    operator()(const uint8_t* packet_buf, LidarScan& ls);

    /**
     * Add a packet to the scan.
     *
     * @param[in] packet a LidarPacket.
     * @param[in] ls lidar scan to populate.
     *
     * @return true when the provided lidar scan is ready to use.
     */
    bool operator()(const ouster::sensor::LidarPacket& packet, LidarScan& ls);

    /**
     * Add a packet to the scan.
     *
     * @param[in] packet_buf a buffer containing raw bytes from a lidar packet.
     * @param[in] packet_ts timestamp of the packet (usually HOST time on
     * receive).
     * @param[in] ls lidar scan to populate.
     *
     * @return true when the provided lidar scan is ready to use.
     */
    bool operator()(const uint8_t* packet_buf, uint64_t packet_ts,
                    LidarScan& ls);
};

}  // namespace ouster

#include "ouster/impl/cartesian.h"
#include "ouster/impl/lidar_scan_impl.h"
