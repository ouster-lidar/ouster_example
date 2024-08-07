/**
 * Copyright (c) 2023, Ouster, Inc.
 * All rights reserved.
 */

#include "ouster/impl/profile_extension.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "ouster/lidar_scan.h"
#include "ouster/types.h"

using namespace ouster::sensor;
using impl::FieldInfo;

TEST(ProfileExtension, ProfileExtensionTest) {
    int profile_nr = 100;
    std::string name = "DUAL_RETURNS_COPYCAT";
    // clang-format off
    std::vector<std::pair<std::string, FieldInfo>> fields{
        {ChanField::RANGE, {UINT32, 0, 0x0007ffff, 0}},
        {ChanField::FLAGS, {UINT8, 2, 0b11111000, 3}},
        {ChanField::REFLECTIVITY, {UINT8, 3, 0, 0}},
        {ChanField::RANGE2, {UINT32, 4, 0x0007ffff, 0}},
        {ChanField::FLAGS2, {UINT8, 6, 0b11111000, 3}},
        {ChanField::REFLECTIVITY2, {UINT8, 7, 0, 0}},
        {ChanField::SIGNAL, {UINT16, 8, 0, 0}},
        {ChanField::SIGNAL2, {UINT16, 10, 0, 0}},
        {ChanField::NEAR_IR, {UINT16, 12, 0, 0}},
        {ChanField::RAW32_WORD1, {UINT32, 0, 0, 0}},
        {ChanField::RAW32_WORD2, {UINT32, 4, 0, 0}},
        {ChanField::RAW32_WORD3, {UINT32, 8, 0, 0}},
        {ChanField::RAW32_WORD4, {UINT32, 12, 0, 0}}};
    size_t chan_data_size = 16;

    EXPECT_NO_THROW(add_custom_profile(profile_nr, name, fields, chan_data_size));
    UDPProfileLidar prof = static_cast<UDPProfileLidar>(profile_nr);
    EXPECT_EQ(udp_profile_lidar_of_string(name).value(), prof);

    auto scan = ouster::LidarScan(40, 60, prof);

    for (const auto& field: fields)
    {
        auto res = scan.fields().find(field.first);
        EXPECT_TRUE(res != scan.fields().end());
    }
    EXPECT_EQ(scan.fields().size(), fields.size());

    // TODO: would be good to check parsing here too -- Tim T.

    // profile already exists
    EXPECT_THROW(
        add_custom_profile(profile_nr, name, fields, chan_data_size),
        std::invalid_argument);
    EXPECT_THROW(
        add_custom_profile(UDPProfileLidar::PROFILE_RNG19_RFL8_SIG16_NIR16_DUAL,
                           "NEW_NAME_DUALRETURNS", fields, chan_data_size),
        std::invalid_argument);
    EXPECT_THROW(
        add_custom_profile(110,
                           to_string(UDPProfileLidar::PROFILE_RNG19_RFL8_SIG16_NIR16_DUAL),
                           fields, chan_data_size),
        std::invalid_argument);
    // nr 0 is prohibited
    EXPECT_THROW(
        add_custom_profile(0, name, fields, chan_data_size),
        std::invalid_argument);
}
