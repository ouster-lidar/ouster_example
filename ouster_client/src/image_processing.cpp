/**
 * Copyright (c) 2018, Ouster, Inc.
 * All rights reserved.
 */

#include "ouster/image_processing.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace ouster {
namespace sdk {
namespace core {
namespace image {

namespace {

/*
 * damping makes the autoexposure smooth and avoids flickering however, it
 * becomes slower to update.
 * 1.0 --> slowest, smoothest
 * 0.0 --> fastest, prone to flickering
 */
const double AE_DEFAULT_DAMPING = 0.90;

/*
 * for performance reasons, we may not want to update every frame but rather
 * every few frames
 */
const int AE_DEFAULT_UPDATE_EVERY = 3;

/* for performance reasons, only consider a subset of points */
const size_t AE_STRIDE = 4;

/* if there are too few points, do nothing */
const size_t AE_MIN_NONZERO_POINTS = 100;

/* default percentile for scaling in autoexposure */
const double AE_DEFAULT_PERCENTILE = 0.1;

/* CLAHE tile grid */
const int CLAHE_TILES_H = 8;
const int CLAHE_TILES_W = 8;
/* number of histogram bins for CLAHE */
const int CLAHE_HIST_BINS = 1024;
/* clip limit as a multiple of the average bin count per tile */
const float CLAHE_CLIP_LIMIT = 1.0f;

inline uint32_t f16_bits_to_f32_bits_fast_nan_zero(uint16_t bits) {
    // Zero (no-return sentinel) must map to f32 zero; the bias-shift formula
    // produces a garbage value for bits==0 so we guard with a cmov (no branch).
    // This also converts quiet nan in f16 to 0
    const uint32_t expanded = static_cast<uint32_t>(bits + 0x1C000u) << 13;
    return bits != 0 ? (bits != 0x7e00 ? expanded : 0u) : 0u;
}

/*
 * Builds per-tile CLAHE lookup tables from a normalised luminance image.
 *
 * Each tile's histogram is clipped at `clip_limit * mean_count` and the
 * excess is redistributed uniformly before computing the CDF. The returned
 * vector has shape [tiles_h * tiles_w * hist_bins] and maps a histogram bin
 * index to an output value in [0, 1].
 */
template <typename T>
std::vector<float> compute_clahe_luts(const Eigen::Tensor<T, 2, Eigen::RowMajor>& lum, int tiles_h,
                                      int tiles_w, int hist_bins, float clip_limit) {
    const int h = static_cast<int>(lum.dimension(0));
    const int w = static_cast<int>(lum.dimension(1));

    std::vector<float> luts(tiles_h * tiles_w * hist_bins, 0.0f);
    for (int ty = 0; ty < tiles_h; ++ty) {
        for (int tx = 0; tx < tiles_w; ++tx) {
            const int y0 = ty * h / tiles_h;
            const int y1 = (ty + 1) * h / tiles_h;
            const int x0 = tx * w / tiles_w;
            const int x1 = (tx + 1) * w / tiles_w;
            const int tile_pixels = (y1 - y0) * (x1 - x0);

            std::vector<float> hist(hist_bins, 0.0f);
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const int bin = std::min(
                        static_cast<int>(static_cast<float>(lum(y, x)) * hist_bins), hist_bins - 1);
                    hist[bin] += 1.0f;
                }
            }

            const float clip_thresh =
                clip_limit * static_cast<float>(tile_pixels) / static_cast<float>(hist_bins);

            float excess = 0.0f;
            for (int bin = 0; bin < hist_bins; ++bin) {
                if (hist[bin] > clip_thresh) {
                    excess += hist[bin] - clip_thresh;
                    hist[bin] = clip_thresh;
                }
            }
            const float redistribute = excess / static_cast<float>(hist_bins);

            float* lut = luts.data() + ((ty * tiles_w) + tx) * hist_bins;

            const float inv_pix = 1.0f / static_cast<float>(tile_pixels);
            float cdf = 0.0f;
            for (int bin = 0; bin < hist_bins; ++bin) {
                cdf += hist[bin] + redistribute;
                lut[bin] = std::min(cdf * inv_pix, 1.0f);
            }
        }
    }
    return luts;
}

/// Applies CLAHE LUTs calculated above to calculate new enhanced luminance
/// values for each pixel.
template <typename T>
Eigen::Tensor<T, 2, Eigen::RowMajor> apply_clahe_luts(
    const Eigen::Tensor<T, 2, Eigen::RowMajor>& lum, const std::vector<float>& luts, int tiles_h,
    int tiles_w, int hist_bins) {
    const int h = static_cast<int>(lum.dimension(0));
    const int w = static_cast<int>(lum.dimension(1));

    Eigen::Tensor<T, 2, Eigen::RowMajor> result(h, w);

    // precompute x-dependent interpolation variables
    struct XCache {
        int tx0, tx1;
        float fx;
    };

    std::vector<XCache> x_cache(w);
    for (int x = 0; x < w; ++x) {
        float tx_f = (static_cast<float>(x) + 0.5f) * tiles_w / w - 0.5f;
        int tx0 = std::max(0, std::min(tiles_w - 1, static_cast<int>(std::floor(tx_f))));
        x_cache[x].tx0 = tx0;
        x_cache[x].tx1 = std::min(tiles_w - 1, tx0 + 1);
        x_cache[x].fx = tx_f - static_cast<float>(tx0);
    }

    const float* luts_ptr = luts.data();

    for (int y = 0; y < h; ++y) {
        // y dependent calculations
        const float ty_f = (static_cast<float>(y) + 0.5f) * tiles_h / h - 0.5f;
        const int ty0 = std::max(0, std::min(tiles_h - 1, static_cast<int>(std::floor(ty_f))));
        const int ty1 = std::min(tiles_h - 1, ty0 + 1);
        const float fy = ty_f - static_cast<float>(ty0);

        const float one_minus_fy = 1.0f - fy;

        // calculate base offsets for the current row's tile-pairs
        const int row_offset_0 = ty0 * tiles_w * hist_bins;
        const int row_offset_1 = ty1 * tiles_w * hist_bins;

        for (int x = 0; x < w; ++x) {
            const int bin = std::min(static_cast<int>(static_cast<float>(lum(y, x)) * hist_bins),
                                     hist_bins - 1);

            const XCache& xc = x_cache[x];
            const float fx = xc.fx;

            const int idx_00 = row_offset_0 + xc.tx0 * hist_bins + bin;
            const int idx_01 = row_offset_0 + xc.tx1 * hist_bins + bin;
            const int idx_10 = row_offset_1 + xc.tx0 * hist_bins + bin;
            const int idx_11 = row_offset_1 + xc.tx1 * hist_bins + bin;

            // bilinear interpolation
            const float mapped =
                one_minus_fy * ((1.0f - fx) * luts_ptr[idx_00] + fx * luts_ptr[idx_01]) +
                fy * ((1.0f - fx) * luts_ptr[idx_10] + fx * luts_ptr[idx_11]);

            result(y, x) = static_cast<T>(mapped);
        }
    }

    return result;
}

}  // namespace

AutoExposure::AutoExposure()
    : lo_percentile_(AE_DEFAULT_PERCENTILE),
      hi_percentile_(AE_DEFAULT_PERCENTILE),
      ae_update_every_(AE_DEFAULT_UPDATE_EVERY),
      damping_(AE_DEFAULT_DAMPING) {}

AutoExposure::AutoExposure(int update_every)
    : lo_percentile_(AE_DEFAULT_PERCENTILE),
      hi_percentile_(AE_DEFAULT_PERCENTILE),
      ae_update_every_(update_every),
      damping_(AE_DEFAULT_DAMPING) {}

AutoExposure::AutoExposure(double lo_percentile, double hi_percentile, int update_every,
                           double damping)
    : lo_percentile_(lo_percentile),
      hi_percentile_(hi_percentile),
      ae_update_every_(update_every),
      damping_(damping) {}

template <typename T>
void AutoExposure::apply(Eigen::Ref<img_t<T>> image, bool update_state) {
    Eigen::Map<Eigen::Array<T, -1, 1>> key_eigen(image.data(), image.size());

    if (counter_ == 0 && update_state) {
        const size_t num_elements = key_eigen.rows();
        std::vector<size_t> indices;
        indices.reserve(num_elements);
        for (size_t i = 0; i < num_elements; i += AE_STRIDE) {
            // ignore 0 values, which are often due to dropped packets etc
            if (key_eigen[i] > 0) {
                indices.push_back(i);
            }
        }
        if (indices.size() < AE_MIN_NONZERO_POINTS) {
            // too few nonzero values, nothing to do
            return;
        }
        auto cmp = [&](const size_t a, const size_t b) { return key_eigen(a) < key_eigen(b); };

        const size_t lo_kth_extreme =
            static_cast<size_t>(static_cast<double>(indices.size()) * lo_percentile_);
        std::nth_element(indices.begin(),
                         indices.begin() + static_cast<std::ptrdiff_t>(lo_kth_extreme),
                         indices.end(), cmp);
        lo_ = key_eigen[*(indices.begin() + static_cast<std::ptrdiff_t>(lo_kth_extreme))];

        const size_t hi_kth_extreme =
            static_cast<size_t>(static_cast<double>(indices.size()) * hi_percentile_);
        std::nth_element(indices.begin() + static_cast<std::ptrdiff_t>(lo_kth_extreme),
                         indices.end() - static_cast<std::ptrdiff_t>(hi_kth_extreme) - 1,
                         indices.end(), cmp);
        hi_ = key_eigen[*(indices.end() - static_cast<std::ptrdiff_t>(hi_kth_extreme) - 1)];

        if (!initialized_) {
            initialized_ = true;
            lo_state_ = lo_;
            hi_state_ = hi_;
        }
    }
    if (!initialized_) {
        return;
    }

    // we use the simplest form of exponential smoothing
    if (update_state) {
        lo_state_ = damping_ * lo_state_ + (1.0 - damping_) * lo_;
        hi_state_ = damping_ * hi_state_ + (1.0 - damping_) * hi_;
    }

    // Apply affine transformation mapping lo_state to lo_percentile and
    // hi_state to 1 - hi_percentile. If it would map 0 to positive number,
    // instead map using only hi_state
    double lo_hi_scale = (1.0 - (lo_percentile_ + hi_percentile_)) / (hi_state_ - lo_state_);

    if (std::isinf(lo_hi_scale) || std::isnan(lo_hi_scale)) {
        // map everything relative to hi_state being 0.5 due to small spread or
        // nan
        key_eigen *= 0.5 / hi_state_;
    } else if (lo_hi_scale * (0.0 - lo_state_) + lo_percentile_ <= 0.00) {
        // apply affine transformation
        key_eigen -= lo_state_;
        key_eigen *= lo_hi_scale;
        key_eigen += lo_percentile_;
    } else {
        // lo_hi_state transformation would map 0 to positive number
        // instead, map using only hi_state
        key_eigen *= (1.0 - hi_percentile_) / (hi_state_);
    }

    // clamp
    key_eigen = key_eigen.max(static_cast<T>(0)).min(static_cast<T>(1));

    if (update_state) {
        counter_ = (counter_ + 1) % ae_update_every_;
    }
}

// use overloads vs templates so implicit conversion to Eigen::Ref still works
void AutoExposure::update(Eigen::Ref<img_t<float>> image, bool update_state) {
    apply(image, update_state);
}

void AutoExposure::update(Eigen::Ref<img_t<double>> image, bool update_state) {
    apply(image, update_state);
}

template <typename T>
void AutoExposure::apply(Eigen::TensorMap<rgb_img_t<T>> image, bool update_state) {
    const Eigen::Index pixel_count_index = image.dimension(0) * image.dimension(1);
    const auto pixel_count = static_cast<size_t>(pixel_count_index);

    if (counter_ == 0 && update_state) {
        Eigen::Tensor<T, 2, Eigen::RowMajor> lum = image.chip(0, 2) * static_cast<T>(0.299) +
                                                   image.chip(1, 2) * static_cast<T>(0.587) +
                                                   image.chip(2, 2) * static_cast<T>(0.114);

        Eigen::Map<Eigen::Array<T, -1, 1>> lum_flat(lum.data(), pixel_count_index);

        std::vector<T> indices;
        size_t reservation = pixel_count / AE_STRIDE + 1;
        indices.reserve(reservation);
        for (size_t i = 0; i < pixel_count; i += AE_STRIDE) {
            if (lum_flat[i] > 0) {
                indices.push_back(lum_flat[i]);
            }
        }
        if (indices.size() < AE_MIN_NONZERO_POINTS) {
            return;
        }

        // validate that we arent somehow resizing
        if (indices.size() > reservation) {
            throw std::runtime_error("Unexpected resize.");
        }

        auto cmp = [&](const T a, const T b) { return a < b; };

        const size_t lo_k =
            static_cast<size_t>(static_cast<double>(indices.size()) * lo_percentile_);
        const auto lo_offset = static_cast<std::ptrdiff_t>(lo_k);
        auto lo_iter = indices.begin() + lo_offset;
        std::nth_element(indices.begin(), lo_iter, indices.end(), cmp);
        lo_ = indices[lo_k];

        const size_t hi_k =
            static_cast<size_t>(static_cast<double>(indices.size()) * hi_percentile_);
        const auto hi_offset = static_cast<std::ptrdiff_t>(hi_k);
        auto hi_iter = indices.end() - hi_offset - 1;
        std::nth_element(lo_iter, hi_iter, indices.end(), cmp);
        hi_ = *hi_iter;

        if (!initialized_) {
            initialized_ = true;
            lo_state_ = lo_;
            hi_state_ = hi_;
        }
    }

    if (!initialized_) {
        return;
    }

    if (update_state) {
        lo_state_ = damping_ * lo_state_ + (1.0 - damping_) * lo_;
        hi_state_ = damping_ * hi_state_ + (1.0 - damping_) * hi_;
    }

    Eigen::Map<Eigen::Array<T, -1, 1>> all_flat(image.data(), image.size());
    double lo_hi_scale = (1.0 - (lo_percentile_ + hi_percentile_)) / (hi_state_ - lo_state_);

    if (std::isinf(lo_hi_scale) || std::isnan(lo_hi_scale)) {
        all_flat *= static_cast<T>(0.5 / hi_state_);
    } else if (lo_hi_scale * (0.0 - lo_state_) + lo_percentile_ <= 0.0) {
        all_flat -= static_cast<T>(lo_state_);
        all_flat *= static_cast<T>(lo_hi_scale);
        all_flat += static_cast<T>(lo_percentile_);
    } else {
        all_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
    }

    all_flat = all_flat.max(static_cast<T>(0)).min(static_cast<T>(1));

    if (update_state) {
        counter_ = (counter_ + 1) % ae_update_every_;
    }
}

void AutoExposure::update(Eigen::TensorMap<rgb_img_t<float>> image, bool update_state) {
    apply(image, update_state);
}

void AutoExposure::update(Eigen::TensorMap<rgb_img_t<double>> image, bool update_state) {
    apply(image, update_state);
}

template <typename T>
void AutoExposure::apply(Eigen::Ref<img_t<T>> r, Eigen::Ref<img_t<T>> g,
                         Eigen::Ref<img_t<T>> b, bool update_state) {
    const Eigen::Index pixel_count_index = r.size();
    const auto pixel_count = static_cast<size_t>(pixel_count_index);

    if (counter_ == 0 && update_state) {
        img_t<T> lum = r * static_cast<T>(0.299) +
                       g * static_cast<T>(0.587) +
                       b * static_cast<T>(0.114);

        Eigen::Map<Eigen::Array<T, -1, 1>> lum_flat(lum.data(),
                                                    pixel_count_index);

        std::vector<size_t> indices;
        indices.reserve(pixel_count);
        for (size_t i = 0; i < pixel_count; i += AE_STRIDE) {
            if (lum_flat[i] > 0) {
                indices.push_back(i);
            }
        }
        if (indices.size() < AE_MIN_NONZERO_POINTS) {
            return;
        }

        auto cmp = [&](const size_t a, const size_t b) {
            return lum_flat[a] < lum_flat[b];
        };

        const size_t lo_k = static_cast<size_t>(
            static_cast<double>(indices.size()) * lo_percentile_);
        const auto lo_offset = static_cast<std::ptrdiff_t>(lo_k);
        auto lo_iter = indices.begin() + lo_offset;
        std::nth_element(indices.begin(), lo_iter, indices.end(), cmp);
        lo_ = lum_flat[indices[lo_k]];

        const size_t hi_k = static_cast<size_t>(
            static_cast<double>(indices.size()) * hi_percentile_);
        const auto hi_offset = static_cast<std::ptrdiff_t>(hi_k);
        auto hi_iter = indices.end() - hi_offset - 1;
        std::nth_element(lo_iter, hi_iter, indices.end(), cmp);
        hi_ = lum_flat[*hi_iter];

        if (!initialized_) {
            initialized_ = true;
            lo_state_ = lo_;
            hi_state_ = hi_;
        }
    }

    if (!initialized_) {
        return;
    }

    if (update_state) {
        lo_state_ = damping_ * lo_state_ + (1.0 - damping_) * lo_;
        hi_state_ = damping_ * hi_state_ + (1.0 - damping_) * hi_;
    }

    Eigen::Map<Eigen::Array<T, -1, 1>> r_flat(r.data(), r.size());
    Eigen::Map<Eigen::Array<T, -1, 1>> g_flat(g.data(), g.size());
    Eigen::Map<Eigen::Array<T, -1, 1>> b_flat(b.data(), b.size());
    double lo_hi_scale =
        (1.0 - (lo_percentile_ + hi_percentile_)) / (hi_state_ - lo_state_);

    if (std::isinf(lo_hi_scale) || std::isnan(lo_hi_scale)) {
        r_flat *= static_cast<T>(0.5 / hi_state_);
        g_flat *= static_cast<T>(0.5 / hi_state_);
        b_flat *= static_cast<T>(0.5 / hi_state_);
    } else if (lo_hi_scale * (0.0 - lo_state_) + lo_percentile_ <= 0.0) {
        r_flat -= static_cast<T>(lo_state_);
        g_flat -= static_cast<T>(lo_state_);
        b_flat -= static_cast<T>(lo_state_);
        r_flat *= static_cast<T>(lo_hi_scale);
        g_flat *= static_cast<T>(lo_hi_scale);
        b_flat *= static_cast<T>(lo_hi_scale);
        r_flat += static_cast<T>(lo_percentile_);
        g_flat += static_cast<T>(lo_percentile_);
        b_flat += static_cast<T>(lo_percentile_);
    } else {
        r_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
        g_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
        b_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
    }

    r_flat = r_flat.max(static_cast<T>(0)).min(static_cast<T>(1));
    g_flat = g_flat.max(static_cast<T>(0)).min(static_cast<T>(1));
    b_flat = b_flat.max(static_cast<T>(0)).min(static_cast<T>(1));

    if (update_state) {
        counter_ = (counter_ + 1) % ae_update_every_;
    }
}

void AutoExposure::update(Eigen::Ref<img_t<float>> r,
                          Eigen::Ref<img_t<float>> g,
                          Eigen::Ref<img_t<float>> b, bool update_state) {
    apply(r, g, b, update_state);
}

void AutoExposure::update(Eigen::Ref<img_t<double>> r,
                          Eigen::Ref<img_t<double>> g,
                          Eigen::Ref<img_t<double>> b, bool update_state) {
    apply(r, g, b, update_state);
}

void AutoExposure::update(Eigen::TensorMap<const rgb_img_t<float16_t>> input,
                          Eigen::TensorMap<rgb_img_t<float>> out, bool update_state) {
    auto in_ptr = reinterpret_cast<const uint16_t*>(input.data());
    const auto count = static_cast<size_t>(input.dimension(0)) *
                       static_cast<size_t>(input.dimension(1)) *
                       static_cast<size_t>(input.dimension(2));
    for (size_t i = 0; i < count; i++) {
        auto result = f16_bits_to_f32_bits_fast_nan_zero(in_ptr[i]);
        memcpy(&out.data()[i], &result, sizeof(float));
    }
    apply(out, update_state);
}

namespace {

/*
 * damping makes the correction smooth and avoids flickering.
 * however, it becomes slower to update.
 * 1.0 --> slowest, smoothest
 * 0.0 --> fastest, prone to flickering
 */
const double BUC_DAMPING = 0.92;

/*
 * for performance reasons, we may not want to update every frame
 * but rather every 8 or so frames.
 */
const int BUC_UPDATE_EVERY = 8;

/*
 * computes the dark count, i.e. an additive offset in the brightness of the
 * image, to smoothe the difference between rows
 */
template <typename T>
Eigen::Array<T, -1, 1> compute_dark_count(const Eigen::Ref<img_t<T>>& image) {
    const size_t image_h = image.rows();
    const size_t image_w = image.cols();

    Eigen::Array<T, -1, 1> tmp = Eigen::Array<T, -1, 1>::Zero(image_w);
    Eigen::Array<T, -1, 1> new_dark_count = Eigen::Array<T, -1, 1>::Zero(image_h);

    // probably computed lazily when used below?
    auto row_diffs = image.bottomRows(image_h - 1) - image.topRows(image_h - 1);

    // to handle azimuth-masked data, only consider columns with nonzero values
    Eigen::Array<bool, -1, 1> col_mask = image.template cast<bool>().colwise().any();
    const size_t n_cols = col_mask.count();

    if (n_cols == 0) {
        return new_dark_count;
    }

    img_t<T> row_diffs_nonzero{image_h - 1, n_cols};
    for (size_t i = 0, j = 0; i < image_w && j < n_cols; i++) {
        if (col_mask[i]) {
            row_diffs_nonzero.col(static_cast<Eigen::Index>(j)) =
                row_diffs.col(static_cast<Eigen::Index>(i));
            j++;
        }
    }

    // compute the median of differences between rows
    for (size_t i = 1; i < image_h; i++) {
        tmp = row_diffs_nonzero.row(i - 1);
        std::nth_element(tmp.data(), tmp.data() + (n_cols / 2), tmp.data() + n_cols);
        new_dark_count[i] = new_dark_count[i - 1] + tmp[n_cols / 2];
    }

    // remove gradients in the entire height of image by doing linear fit
    Eigen::Matrix<T, -1, 2> image_array(image_h, 2);
    for (size_t i = 0; i < image_h; i++) {
        image_array(i, 0) = 1;
        image_array(i, 1) = static_cast<T>(i);
    }
    Eigen::Matrix<T, 2, 1> x = image_array.fullPivLu().solve(new_dark_count.matrix());
    new_dark_count -= (image_array * x).array();

    // subtract minimum value
    new_dark_count -= new_dark_count.minCoeff();
    return new_dark_count;
}

}  // namespace

template <typename T>
void BeamUniformityCorrector::apply(Eigen::Ref<img_t<T>> image, bool update_state) {
    const auto image_h = image.rows();

    // compute dark counts, if necessary
    if (dark_count_.size() != image_h) {
        dark_count_ = compute_dark_count(image).template cast<double>();
    } else if (update_state && counter_ == 0) {
        // if previous state exists, update using exponential smoothing:
        Eigen::ArrayXd new_dark_count = compute_dark_count(image).template cast<double>();
        dark_count_ *= BUC_DAMPING;
        dark_count_ += new_dark_count * (1.0 - BUC_DAMPING);
    }
    counter_ = (counter_ + 1) % BUC_UPDATE_EVERY;

    // apply the dark count correction
    image.colwise() -= dark_count_.cast<T>();

    // clamp any negative values
    image = image.cwiseMax(static_cast<T>(0));
}

void BeamUniformityCorrector::update(Eigen::Ref<img_t<float>> image, bool update_state) {
    apply(image, update_state);
}

void BeamUniformityCorrector::update(Eigen::Ref<img_t<double>> image, bool update_state) {
    apply(image, update_state);
}

LocalToneMapper::LocalToneMapper()
    : lo_percentile_(0.0), hi_percentile_(0.2), update_every_(1), damping_(0.3) {}

LocalToneMapper::LocalToneMapper(int update_every)
    : lo_percentile_(0.0), hi_percentile_(0.2), update_every_(update_every), damping_(0.3) {}

LocalToneMapper::LocalToneMapper(double lo_percentile, double hi_percentile, int update_every,
                                 double damping, bool compress_dr, bool color_correct)
    : lo_percentile_(lo_percentile),
      hi_percentile_(hi_percentile),
      update_every_(update_every),
      damping_(damping),
      compress_dr_(compress_dr),
      color_correct_(color_correct) {}

template <typename T>
void LocalToneMapper::apply(Eigen::TensorMap<rgb_img_t<T>> image, bool update_state) {
    const size_t pixel_count = image.dimension(0) * image.dimension(1);

    // --- Stage 1: AutoExposure scaling ---
    // Update lo/hi luminance percentile state.
    if (counter_ == 0 && update_state) {
        Eigen::Tensor<T, 2, Eigen::RowMajor> lum = image.chip(0, 2) * static_cast<T>(0.299) +
                                                   image.chip(1, 2) * static_cast<T>(0.587) +
                                                   image.chip(2, 2) * static_cast<T>(0.114);

        Eigen::Map<Eigen::Array<T, -1, 1>> lum_flat(lum.data(), pixel_count);

        std::vector<T> indices;
        size_t reservation = pixel_count / AE_STRIDE + 1;
        indices.reserve(reservation);
        for (size_t i = 0; i < pixel_count; i += AE_STRIDE) {
            if (lum_flat[i] > 0) {
                indices.push_back(lum_flat[i]);
            }
        }
        if (indices.size() < AE_MIN_NONZERO_POINTS) {
            return;
        }

        // validate that we arent somehow resizing
        if (indices.size() > reservation) {
            throw std::runtime_error("Unexpected resize.");
        }

        auto cmp = [&](const T a, const T b) { return a < b; };

        const size_t lo_k = static_cast<size_t>(indices.size() * lo_percentile_);
        std::nth_element(indices.begin(), indices.begin() + lo_k, indices.end(), cmp);
        lo_ = indices[lo_k];

        const size_t hi_k = static_cast<size_t>(indices.size() * hi_percentile_);
        std::nth_element(indices.begin() + lo_k, indices.end() - hi_k - 1, indices.end(), cmp);
        hi_ = *(indices.end() - hi_k - 1);

        if (!initialized_) {
            initialized_ = true;
            lo_state_ = lo_;
            hi_state_ = hi_;
        }
    }

    if (!initialized_) {
        return;
    }

    if (update_state) {
        lo_state_ = damping_ * lo_state_ + (1.0 - damping_) * lo_;
        hi_state_ = damping_ * hi_state_ + (1.0 - damping_) * hi_;
    }

    // Apply a uniform affine transform derived from luminance to all channels,
    // mapping the dynamic range into [0, 1] while preserving color balance.
    Eigen::Map<Eigen::Array<T, -1, 1>> all_flat(image.data(), image.size());

    double lo_hi_scale = (1.0 - (lo_percentile_ + hi_percentile_)) / (hi_state_ - lo_state_);

    if (std::isinf(lo_hi_scale) || std::isnan(lo_hi_scale)) {
        all_flat *= static_cast<T>(0.5 / hi_state_);
    } else if (lo_hi_scale * (0.0 - lo_state_) + lo_percentile_ <= 0.0) {
        all_flat -= static_cast<T>(lo_state_);
        all_flat *= static_cast<T>(lo_hi_scale);
        all_flat += static_cast<T>(lo_percentile_);
    } else {
        all_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
    }

    if (update_state) {
        counter_ = (counter_ + 1) % update_every_;
    }

    T thresh = 0.8;
    T scale = 0.05;

    // --- Stage 2: Dynamic Range Compression ---
    // bring down brightness of incredibly bright areas when in dark
    // environments
    if (hi_state_ < 0.2 && compress_dr_) {
        T* b = const_cast<T*>(image.data());
        const auto nb_signed = static_cast<std::ptrdiff_t>(image.size());
        for (std::ptrdiff_t i = 0; i < nb_signed; i += 3) {
            T lum = b[i + 0] * 0.299 + b[i + 1] * 0.587 + b[i + 2] * 0.114;

            if (lum > thresh) {
                auto new_lum = thresh + (lum - thresh) * scale;
                auto scale = new_lum / lum;
                b[i + 0] *= scale;
                b[i + 1] *= scale;
                b[i + 2] *= scale;
            }
        }
    }

    // use Reinhard to saturate at 1 make sure we don't erase bright spots
    all_flat = all_flat.max(static_cast<T>(0));
    all_flat = all_flat / (1.0 + all_flat);

    const Eigen::Tensor<T, 2, Eigen::RowMajor> lum_ae = image.chip(0, 2) * static_cast<T>(0.299) +
                                                        image.chip(1, 2) * static_cast<T>(0.587) +
                                                        image.chip(2, 2) * static_cast<T>(0.114);

    // --- Stage 3: CLAHE on luminance ---
    // Compute CLAHE on the AE-normalised luminance, then scale each channel by
    // lum_clahe / lum_ae to enhance local contrast while preserving hue.
    const int h = static_cast<int>(image.dimension(0));
    const int w = static_cast<int>(image.dimension(1));

    const auto luts =
        compute_clahe_luts(lum_ae, CLAHE_TILES_H, CLAHE_TILES_W, CLAHE_HIST_BINS, CLAHE_CLIP_LIMIT);
    const Eigen::Tensor<T, 2, Eigen::RowMajor> lum_clahe =
        apply_clahe_luts(lum_ae, luts, CLAHE_TILES_H, CLAHE_TILES_W, CLAHE_HIST_BINS);

    const T color_factor = static_cast<T>(0.75);
    if (!color_correct_) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const T lum_old = lum_ae(y, x);
                const T lum_new = lum_clahe(y, x);
                const T scale =
                    (lum_old > static_cast<T>(1e-6)) ? lum_new / lum_old : static_cast<T>(1);
                image(y, x, 0) = std::min(image(y, x, 0) * scale, static_cast<T>(1));
                image(y, x, 1) = std::min(image(y, x, 1) * scale, static_cast<T>(1));
                image(y, x, 2) = std::min(image(y, x, 2) * scale, static_cast<T>(1));
            }
        }
    } else {
        // --- Stage 4: Improve color saturation ---
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const T lum_old = lum_ae(y, x);
                const T lum_new = lum_clahe(y, x);
                const T scale =
                    (lum_old > static_cast<T>(1e-6)) ? lum_new / lum_old : static_cast<T>(1);
                auto r = image(y, x, 0) * scale;
                auto g = image(y, x, 1) * scale;
                auto b = image(y, x, 2) * scale;

                auto grey = lum_new;

                r = -grey * color_factor + r * (1 + color_factor);
                g = -grey * color_factor + g * (1 + color_factor);
                b = -grey * color_factor + b * (1 + color_factor);

                image(y, x, 0) = std::max(static_cast<T>(0), std::min(r, static_cast<T>(1)));
                image(y, x, 1) = std::max(static_cast<T>(0), std::min(g, static_cast<T>(1)));
                image(y, x, 2) = std::max(static_cast<T>(0), std::min(b, static_cast<T>(1)));
            }
        }
    }
}

void LocalToneMapper::update(Eigen::TensorMap<rgb_img_t<float>> image, bool update_state) {
    apply(image, update_state);
}

void LocalToneMapper::update(Eigen::TensorMap<rgb_img_t<double>> image, bool update_state) {
    apply(image, update_state);
}

template <typename T>
void LocalToneMapper::apply(Eigen::Ref<img_t<T>> r, Eigen::Ref<img_t<T>> g,
                            Eigen::Ref<img_t<T>> b, bool update_state) {
    const auto pixel_count = static_cast<size_t>(r.size());

    // --- Stage 1: AutoExposure scaling ---
    // Update lo/hi luminance percentile state.
    if (counter_ == 0 && update_state) {
        img_t<T> lum = r * static_cast<T>(0.299) + g * static_cast<T>(0.587) +
                       b * static_cast<T>(0.114);

        Eigen::Map<Eigen::Array<T, -1, 1>> lum_flat(lum.data(), pixel_count);

        std::vector<T> indices;
        size_t reservation = pixel_count / AE_STRIDE + 1;
        indices.reserve(reservation);
        for (size_t i = 0; i < pixel_count; i += AE_STRIDE) {
            if (lum_flat[i] > 0) {
                indices.push_back(lum_flat[i]);
            }
        }
        if (indices.size() < AE_MIN_NONZERO_POINTS) {
            return;
        }

        // validate that we arent somehow resizing
        if (indices.size() > reservation) {
            throw std::runtime_error("Unexpected resize.");
        }

        auto cmp = [&](const T a, const T b) { return a < b; };

        const size_t lo_k = static_cast<size_t>(indices.size() * lo_percentile_);
        std::nth_element(indices.begin(), indices.begin() + lo_k, indices.end(), cmp);
        lo_ = indices[lo_k];

        const size_t hi_k = static_cast<size_t>(indices.size() * hi_percentile_);
        std::nth_element(indices.begin() + lo_k, indices.end() - hi_k - 1, indices.end(), cmp);
        hi_ = *(indices.end() - hi_k - 1);

        if (!initialized_) {
            initialized_ = true;
            lo_state_ = lo_;
            hi_state_ = hi_;
        }
    }

    if (!initialized_) {
        return;
    }

    if (update_state) {
        lo_state_ = damping_ * lo_state_ + (1.0 - damping_) * lo_;
        hi_state_ = damping_ * hi_state_ + (1.0 - damping_) * hi_;
    }

    // Apply a uniform affine transform derived from luminance to all channels,
    // mapping the dynamic range into [0, 1] while preserving color balance.
    Eigen::Map<Eigen::Array<T, -1, 1>> r_flat(r.data(), r.size());
    Eigen::Map<Eigen::Array<T, -1, 1>> g_flat(g.data(), g.size());
    Eigen::Map<Eigen::Array<T, -1, 1>> b_flat(b.data(), b.size());

    double lo_hi_scale = (1.0 - (lo_percentile_ + hi_percentile_)) / (hi_state_ - lo_state_);

    if (std::isinf(lo_hi_scale) || std::isnan(lo_hi_scale)) {
        r_flat *= static_cast<T>(0.5 / hi_state_);
        g_flat *= static_cast<T>(0.5 / hi_state_);
        b_flat *= static_cast<T>(0.5 / hi_state_);
    } else if (lo_hi_scale * (0.0 - lo_state_) + lo_percentile_ <= 0.0) {
        r_flat -= static_cast<T>(lo_state_);
        g_flat -= static_cast<T>(lo_state_);
        b_flat -= static_cast<T>(lo_state_);
        r_flat *= static_cast<T>(lo_hi_scale);
        g_flat *= static_cast<T>(lo_hi_scale);
        b_flat *= static_cast<T>(lo_hi_scale);
        r_flat += static_cast<T>(lo_percentile_);
        g_flat += static_cast<T>(lo_percentile_);
        b_flat += static_cast<T>(lo_percentile_);
    } else {
        r_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
        g_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
        b_flat *= static_cast<T>((1.0 - hi_percentile_) / hi_state_);
    }

    if (update_state) {
        counter_ = (counter_ + 1) % update_every_;
    }

    T thresh = 0.8;
    T scale = 0.05;

    // --- Stage 2: Dynamic Range Compression ---
    // bring down brightness of incredibly bright areas when in dark
    // environments
    if (hi_state_ < 0.2 && compress_dr_) {
        T* rp = r.data();
        T* gp = g.data();
        T* bp = b.data();
        for (size_t i = 0; i < pixel_count; ++i) {
            T lum = rp[i] * 0.299 + gp[i] * 0.587 + bp[i] * 0.114;

            if (lum > thresh) {
                auto new_lum = thresh + (lum - thresh) * scale;
                auto scale = new_lum / lum;
                rp[i] *= scale;
                gp[i] *= scale;
                bp[i] *= scale;
            }
        }
    }

    // use Reinhard to saturate at 1 make sure we don't erase bright spots
    r_flat = r_flat.max(static_cast<T>(0));
    g_flat = g_flat.max(static_cast<T>(0));
    b_flat = b_flat.max(static_cast<T>(0));
    r_flat = r_flat / (1.0 + r_flat);
    g_flat = g_flat / (1.0 + g_flat);
    b_flat = b_flat / (1.0 + b_flat);

    // --- Stage 3: CLAHE on luminance ---
    // Compute CLAHE on the AE-normalised luminance, then scale each channel by
    // lum_clahe / lum_ae to enhance local contrast while preserving hue.
    const int h = static_cast<int>(r.rows());
    const int w = static_cast<int>(r.cols());

    Eigen::Tensor<T, 2, Eigen::RowMajor> lum_ae(h, w);
    Eigen::Map<Eigen::Array<T, -1, 1>>(lum_ae.data(), pixel_count) =
        r_flat * static_cast<T>(0.299) + g_flat * static_cast<T>(0.587) +
        b_flat * static_cast<T>(0.114);

    const auto luts =
        compute_clahe_luts(lum_ae, CLAHE_TILES_H, CLAHE_TILES_W, CLAHE_HIST_BINS, CLAHE_CLIP_LIMIT);
    const Eigen::Tensor<T, 2, Eigen::RowMajor> lum_clahe =
        apply_clahe_luts(lum_ae, luts, CLAHE_TILES_H, CLAHE_TILES_W, CLAHE_HIST_BINS);

    const T color_factor = static_cast<T>(0.75);
    if (!color_correct_) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const T lum_old = lum_ae(y, x);
                const T lum_new = lum_clahe(y, x);
                const T scale =
                    (lum_old > static_cast<T>(1e-6)) ? lum_new / lum_old : static_cast<T>(1);
                r(y, x) = std::min(r(y, x) * scale, static_cast<T>(1));
                g(y, x) = std::min(g(y, x) * scale, static_cast<T>(1));
                b(y, x) = std::min(b(y, x) * scale, static_cast<T>(1));
            }
        }
    } else {
        // --- Stage 4: Improve color saturation ---
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const T lum_old = lum_ae(y, x);
                const T lum_new = lum_clahe(y, x);
                const T scale =
                    (lum_old > static_cast<T>(1e-6)) ? lum_new / lum_old : static_cast<T>(1);
                auto rv = r(y, x) * scale;
                auto gv = g(y, x) * scale;
                auto bv = b(y, x) * scale;

                auto grey = lum_new;

                rv = -grey * color_factor + rv * (1 + color_factor);
                gv = -grey * color_factor + gv * (1 + color_factor);
                bv = -grey * color_factor + bv * (1 + color_factor);

                r(y, x) = std::max(static_cast<T>(0), std::min(rv, static_cast<T>(1)));
                g(y, x) = std::max(static_cast<T>(0), std::min(gv, static_cast<T>(1)));
                b(y, x) = std::max(static_cast<T>(0), std::min(bv, static_cast<T>(1)));
            }
        }
    }
}

void LocalToneMapper::update(Eigen::Ref<img_t<float>> r, Eigen::Ref<img_t<float>> g,
                             Eigen::Ref<img_t<float>> b, bool update_state) {
    apply(r, g, b, update_state);
}

void LocalToneMapper::update(Eigen::Ref<img_t<double>> r, Eigen::Ref<img_t<double>> g,
                             Eigen::Ref<img_t<double>> b, bool update_state) {
    apply(r, g, b, update_state);
}

void LocalToneMapper::update(Eigen::TensorMap<const rgb_img_t<float16_t>> input,
                             Eigen::TensorMap<rgb_img_t<float>> output, bool update_state) {
    const auto src = reinterpret_cast<const uint16_t*>(input.data());
    size_t count = input.dimension(0) * input.dimension(1) * input.dimension(2);
    // convert to float
    for (size_t i = 0; i < count; i++) {
        auto result = f16_bits_to_f32_bits_fast_nan_zero(src[i]);
        memcpy(&output.data()[i], &result, sizeof(float));
    }
    apply(output, update_state);
}

}  // namespace image
}  // namespace core
}  // namespace sdk
}  // namespace ouster
