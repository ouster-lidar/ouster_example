/**
 * Copyright (c) 2018, Ouster, Inc.
 * All rights reserved.
 *
 * @file
 * @brief Utilities for post-processing image data
 */

#pragma once

#include <Eigen/Core>

#include "ouster/types.h"
#include "ouster/visibility.h"

namespace ouster {
namespace sdk {
namespace core {
namespace image {

/** Adjusts brightness to between 0 and 1 using a simple auto-exposure method.
 *  Works with 1 or 3 channel images. For 3 channel images it applies
 *  autoexposure in luminance space to keep colors consistent.
 */
class OUSTER_API_CLASS AutoExposure {
    const double lo_percentile_,
        hi_percentile_;  // percentiles used for scaling
    const int ae_update_every_;
    const double damping_;

    double lo_state_ = -1.0;
    double hi_state_ = -1.0;
    double lo_ = -1.0;
    double hi_ = -1.0;

    bool initialized_ = false;
    int counter_ = 0;

    template <typename T>
    void apply(Eigen::Ref<img_t<T>> image, bool update_state);

    template <typename T>
    void apply(Eigen::TensorMap<rgb_img_t<T>> image, bool update_state);

    template <typename T>
    void apply(Eigen::Ref<img_t<T>> r, Eigen::Ref<img_t<T>> g,
               Eigen::Ref<img_t<T>> b, bool update_state);

   public:
    /** Default constructor using default percentile and update values. */
    OUSTER_API_FUNCTION
    AutoExposure();

    /**
     * Constructor specifying update modulo, and using default percentiles.
     *
     * @param[in] update_every update every this number of frames.
     */
    OUSTER_API_FUNCTION
    AutoExposure(int update_every);

    /**
     * Constructor specifying low and high percentiles, and update modulo.
     *
     * @param[in] lo_percentile low percentile to use for adjustment.
     * @param[in] hi_percentile high percentile to use for adjustment.
     * @param[in] update_every update every this number of frames.
     * @param[in] damping update ratio for AE setpoints, 0 = fast, 1 = slow
     */
    OUSTER_API_FUNCTION
    AutoExposure(double lo_percentile, double hi_percentile, int update_every,
                 double damping = 0.9);

    /**
     * Scales the image so that contrast is stretched between 0 and 1.
     *
     * The top percentile is 1 - hi_percentile and the bottom percentile is
     * lo_percentile. Similar to linear 'contrast-stretch', i.e. normalization.
     *
     * @param[in] image Reference to the image, modified in place.
     * @param[in] update_state Update lo/hi percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<float>> image, bool update_state = true);

    /**
     * Scales the image so that contrast is stretched between 0 and 1.
     *
     * The top percentile is 1 - hi_percentile and the bottom percentile is
     * lo_percentile. Similar to linear 'contrast-stretch', i.e. normalization.
     *
     * @param[in] image Reference to the image, modified in place.
     * @param[in] update_state Update lo/hi percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<double>> image, bool update_state = true);

    /**
     * Apply global RGB auto-exposure in-place stretching constrast between 0
     * and 1.
     *
     * @param[in] image RGB image tensor (H x W x 3), modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::TensorMap<rgb_img_t<float>> image, bool update_state = true);

    /**
     * Apply global RGB auto-exposure in-place stretching constrast between 0
     * and 1.
     *
     * @param[in] image RGB image tensor (H x W x 3), modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::TensorMap<rgb_img_t<double>> image, bool update_state = true);

    /**
     * Apply global RGB auto-exposure in-place stretching constrast between 0
     * and 1.
     *
     * @param[in] r Red channel image, modified in-place.
     * @param[in] g Green channel image, modified in-place.
     * @param[in] b Blue channel image, modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<float>> r, Eigen::Ref<img_t<float>> g,
                Eigen::Ref<img_t<float>> b, bool update_state = true);

    /**
     * Apply global RGB auto-exposure in-place stretching constrast between 0
     * and 1.
     *
     * @param[in] r Red channel image, modified in-place.
     * @param[in] g Green channel image, modified in-place.
     * @param[in] b Blue channel image, modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<double>> r, Eigen::Ref<img_t<double>> g,
                Eigen::Ref<img_t<double>> b, bool update_state = true);

    /**
     * Convert fp16-bit RGB to float and apply global RGB auto-exposure
     * stretching constrast between 0 and 1.
     *
     * @param[in] input input RGB image tensor (H x W x 3) as float16_t.
     * @param[out] out output RGB image tensor (H x W x 3) as float.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::TensorMap<const rgb_img_t<ouster::sdk::core::float16_t>> input,
                Eigen::TensorMap<rgb_img_t<float>> out, bool update_state = true);
};

/**
 * Corrects beam uniformity by minimizing median difference between rows,
 * thereby correcting subtle horizontal line artifacts in images, especially the
 * ambient image.
 */
class OUSTER_API_CLASS BeamUniformityCorrector {
   private:
    int counter_ = 0;
    Eigen::ArrayXd dark_count_;

    template <typename T>
    void apply(Eigen::Ref<img_t<T>> image, bool update_state);

   public:
    /**
     * Applies dark count correction to an image, modifying it in-place to have
     * reduced horizontal line artifacts.
     *
     * @param[in] image Reference to the image, modified in-place.
     * @param[in] update_state Update dark counts if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<float>> image, bool update_state = true);

    /**
     * Applies dark count correction to an image, modifying it in-place to have
     * reduced horizontal line artifacts.
     *
     * @param[in] image Reference to the image, modified in-place.
     * @param[in] update_state Update dark counts if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<double>> image, bool update_state = true);
};

/**
 * Applies CLAHE-based HDR tone-mapping correction to an RGB image.
 *
 * Computes an exposure correction from the image luminance (weighted sum of
 * R, G, B channels) and applies a uniform affine scale to all three channels,
 * preserving color balance while stretching contrast between 0 and 1.
 * State is updated with smoothing across frames to avoid flickering.
 */
class OUSTER_API_CLASS LocalToneMapper {
    const double lo_percentile_;
    const double hi_percentile_;
    const int update_every_;
    const double damping_;

    double lo_state_ = -1.0;
    double hi_state_ = -1.0;
    double lo_ = -1.0;
    double hi_ = -1.0;

    bool initialized_ = false;
    int counter_ = 0;

    bool compress_dr_ = true;
    bool color_correct_ = true;

    template <typename T>
    void apply(Eigen::TensorMap<rgb_img_t<T>> image, bool update_state);

    template <typename T>
    void apply(Eigen::Ref<img_t<T>> r, Eigen::Ref<img_t<T>> g,
               Eigen::Ref<img_t<T>> b, bool update_state);

   public:
    /** Default constructor using default percentile and update values. */
    OUSTER_API_FUNCTION
    LocalToneMapper();

    /**
     * Constructor specifying update modulo, and using default percentiles.
     *
     * @param[in] update_every update every this number of frames.
     */
    OUSTER_API_FUNCTION
    LocalToneMapper(int update_every);

    /**
     * Constructor specifying low and high percentiles, and update modulo.
     *
     * @param[in] lo_percentile low percentile to use for luminance scaling.
     * @param[in] hi_percentile high percentile to use for luminance scaling.
     * @param[in] update_every update every this number of frames.
     * @param[in] damping update ratio for AE setpoints, 0 = fast, 1 = slow
     * @param[in] compress_dr compress DR in dark scenes such as tunnels
     * @param[in] color_correct correct colors to recover any dampened colors
     */
    OUSTER_API_FUNCTION
    LocalToneMapper(double lo_percentile, double hi_percentile, int update_every, double damping,
                    bool compress_dr, bool color_correct);

    /**
     * Adjusts the RGB image so that contrast is stretched between 0 and 1.
     *
     * @param[in] image RGB image tensor (H x W x 3), modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::TensorMap<rgb_img_t<float>> image, bool update_state = true);

    /**
     * Adjusts the RGB image so that contrast is stretched between 0 and 1.
     *
     * @param[in] image RGB image tensor (H x W x 3), modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::TensorMap<rgb_img_t<double>> image, bool update_state = true);

    /**
     * Adjusts the RGB image so that contrast is stretched between 0 and 1.
     *
     * Operates on planar channels (each stored as a separate H x W image)
     * rather than an interleaved tensor, modifying them in-place. Produces the
     * same end result as the interleaved rgb_img_t overload.
     *
     * @param[in] r Red channel image, modified in-place.
     * @param[in] g Green channel image, modified in-place.
     * @param[in] b Blue channel image, modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<float>> r, Eigen::Ref<img_t<float>> g,
                Eigen::Ref<img_t<float>> b, bool update_state = true);

    /**
     * Adjusts the RGB image so that contrast is stretched between 0 and 1.
     *
     * Operates on planar channels (each stored as a separate H x W image)
     * rather than an interleaved tensor, modifying them in-place. Produces the
     * same end result as the interleaved rgb_img_t overload.
     *
     * @param[in] r Red channel image, modified in-place.
     * @param[in] g Green channel image, modified in-place.
     * @param[in] b Blue channel image, modified in-place.
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::Ref<img_t<double>> r, Eigen::Ref<img_t<double>> g,
                Eigen::Ref<img_t<double>> b, bool update_state = true);

    /**
     * Adjusts the RGB image so that contrast is stretched between 0 and 1.
     *
     * @param[in] input RGB image tensor (H x W x 3) to tone-map
     * @param[out] output RGB image tensor (H x W x 3)
     * @param[in] update_state Update lo/hi luminance percentiles if true.
     */
    OUSTER_API_FUNCTION
    void update(Eigen::TensorMap<const rgb_img_t<float16_t>> input,
                Eigen::TensorMap<rgb_img_t<float>> output, bool update_state = true);
};

}  // namespace image
}  // namespace core
}  // namespace sdk
}  // namespace ouster
