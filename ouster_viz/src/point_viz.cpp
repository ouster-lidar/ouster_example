#include "ouster/point_viz.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "camera.h"
#include "cloud.h"
#include "glfw.h"
#include "image.h"
#include "misc.h"
#include "ouster/colormaps.h"

static_assert(std::is_same<GLfloat, float>::value,
              "Platform has unexpected definition of GLfloat");

namespace ouster {
namespace viz {

namespace {

/*
 * Helper for addable / removable drawable objects
 */
template <typename GL, typename T>
class Indexed {
    struct Front {
        std::unique_ptr<GL> gl;
        std::unique_ptr<T> state;
    };
    using Back = std::shared_ptr<T>;

    std::vector<Front> front;
    std::vector<Back> back;

   public:
    Indexed() : front{}, back{} {}

    void add(const std::shared_ptr<T>& t) {
        // find and use first empty slot, or grow
        auto res = std::find_if(back.begin(), back.end(),
                                [](const Back& b) { return !b; });
        if (res == back.end()) {
            back.push_back(t);
        } else {
            *res = t;
        }
    }

    bool remove(const std::shared_ptr<T>& t) {
        auto res = std::find(back.begin(), back.end(), t);
        if (res == back.end())
            return false;
        else {
            res->reset();
            return true;
        }
    }

    void draw(const WindowCtx& ctx, const impl::CameraData& camera) {
        for (auto& f : front) {
            if (!f.state) continue;                   // skip deleted
            if (!f.gl) f.gl.reset(new GL{*f.state});  // init GL for added
            f.gl->draw(ctx, camera, *f.state);
        }
    }

    void swap() {
        assert(front.size() <= back.size());

        // in case back grew
        if (front.size() < back.size()) front.resize(back.size());

        // send updated, added or deleted state to the front
        for (size_t i = 0; i < back.size(); i++) {
            if (back[i] && front[i].state) {
                std::swap(*front[i].state, *back[i]);
            } else if (back[i] && !front[i].state) {
                front[i].state.reset(new T{*back[i]});
                back[i]->clear();
            } else if (!back[i] && front[i].state) {
                front[i].state.reset();
            }
        }
    }
};

}  // namespace

/*
 * PointViz implementation
 */
struct PointViz::Impl {
    GLFWContext glfw;
    GLuint vao;

    // state for drawing
    std::mutex update_mx;
    bool front_changed{false};

    Camera camera_back, camera_front;

    TargetDisplay target;
    impl::GLRings rings;

    Indexed<impl::GLCloud, Cloud> clouds;
    Indexed<impl::GLCuboid, Cuboid> cuboids;
    Indexed<impl::GLLabel3d, Label3d> labels;
    Indexed<impl::GLImage, Image> images;

    template <typename T>
    using Handlers = std::vector<std::function<T>>;

    Handlers<bool(const WindowCtx&, int, int)> key_handlers;
    Handlers<bool(const WindowCtx&, int, int)> mouse_button_handlers;
    Handlers<bool(const WindowCtx&, double, double)> scroll_handlers;
    Handlers<bool(const WindowCtx&, double, double)> mouse_pos_handlers;

    Impl(const std::string& name, bool fix_aspect, int window_width,
         int window_height)
        : glfw{name, fix_aspect, window_width, window_height} {}
};

/*
 * PointViz interface
 */

PointViz::PointViz(const std::string& name, bool fix_aspect, int window_width,
                   int window_height) {
    // TODO initialization (and opengl API usage) still pretty messed up due to
    // single shared vao
    pimpl = std::unique_ptr<Impl>{
        new Impl{name, fix_aspect, window_width, window_height}};

    // top-level gl state for point viz
    glfwMakeContextCurrent(pimpl->glfw.window);
    glGenVertexArrays(1, &pimpl->vao);
    glBindVertexArray(pimpl->vao);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // TODO: need to check if these were already called?
    impl::GLCloud::initialize();
    impl::GLImage::initialize();
    impl::GLRings::initialize();
    impl::GLCuboid::initialize();

    // add user-setable input handlers
    pimpl->glfw.key_handler = [this](const WindowCtx& ctx, int key, int mods) {
        for (auto& f : pimpl->key_handlers)
            if (!f(ctx, key, mods)) break;
    };
    pimpl->glfw.mouse_button_handler = [this](const WindowCtx& ctx, int button,
                                              int mods) {
        for (auto& f : pimpl->mouse_button_handlers)
            if (!f(ctx, button, mods)) break;
    };
    pimpl->glfw.scroll_handler = [this](const WindowCtx& ctx, double x,
                                        double y) {
        for (auto& f : pimpl->scroll_handlers)
            if (!f(ctx, x, y)) break;
    };
    pimpl->glfw.mouse_pos_handler = [this](const WindowCtx& ctx, double x,
                                           double y) {
        for (auto& f : pimpl->mouse_pos_handlers)
            if (!f(ctx, x, y)) break;
    };

    // glfwPollEvents blocks during resize on macos. Keep rendering to avoid
    // artifacts during resize
    pimpl->glfw.resize_handler = [this]() {
        (void)this;
#ifdef __APPLE__
        draw();
#endif
    };
}

PointViz::~PointViz() { glDeleteVertexArrays(1, &pimpl->vao); }

void PointViz::run() {
    pimpl->glfw.running(true);
    pimpl->glfw.visible(true);
    while (running()) run_once();
    pimpl->glfw.visible(false);
}

void PointViz::run_once() {
    glfwMakeContextCurrent(pimpl->glfw.window);
    draw();
    glfwPollEvents();
}

bool PointViz::running() { return pimpl->glfw.running(); }

void PointViz::running(bool state) { pimpl->glfw.running(state); }

void PointViz::visible(bool state) { pimpl->glfw.visible(state); }

bool PointViz::update() {
    std::lock_guard<std::mutex> guard{pimpl->update_mx};

    // propagate camera changes
    pimpl->camera_front = pimpl->camera_back;

    // last frame hasn't been drawn yet
    if (pimpl->front_changed) return false;

    pimpl->clouds.swap();
    pimpl->cuboids.swap();
    pimpl->labels.swap();
    pimpl->images.swap();
    pimpl->rings.update(pimpl->target);

    pimpl->front_changed = true;

    return true;
}

void PointViz::draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(pimpl->vao);

    // draw images
    {
        std::lock_guard<std::mutex> guard{pimpl->update_mx};
        const auto& ctx = pimpl->glfw.window_context;

        // calculate camera matrices
        auto camera_data =
            pimpl->camera_front.matrices(impl::window_aspect(ctx));

        // draw image
        impl::GLImage::beginDraw();
        pimpl->images.draw(ctx, camera_data);
        impl::GLImage::endDraw();

        // draw clouds
        impl::GLCloud::beginDraw();
        pimpl->clouds.draw(ctx, camera_data);
        impl::GLCloud::endDraw();

        // draw rings
        pimpl->rings.draw(ctx, camera_data);

        // draw cuboids
        impl::GLCuboid::beginDraw();
        pimpl->cuboids.draw(ctx, camera_data);
        impl::GLCuboid::endDraw();

        // draw labels
        impl::GLLabel3d::beginDraw();
        pimpl->labels.draw(ctx, camera_data);
        impl::GLLabel3d::endDraw();

        // mark front buffers no longer dirty
        pimpl->front_changed = false;
    }

    glfwSwapBuffers(pimpl->glfw.window);
}

/*
 * Input handling
 */
void PointViz::push_key_handler(
    std::function<bool(const WindowCtx&, int, int)>&& f) {
    // TODO: not thread safe: called in glfwPollEvents()
    pimpl->key_handlers.push_back(std::move(f));
}

void PointViz::push_mouse_button_handler(
    std::function<bool(const WindowCtx&, int, int)>&& f) {
    pimpl->mouse_button_handlers.push_back(std::move(f));
}

void PointViz::push_scroll_handler(
    std::function<bool(const WindowCtx&, double, double)>&& f) {
    pimpl->scroll_handlers.push_back(std::move(f));
}

void PointViz::push_mouse_pos_handler(
    std::function<bool(const WindowCtx&, double, double)>&& f) {
    pimpl->mouse_pos_handlers.push_back(std::move(f));
}

void PointViz::pop_key_handler() { pimpl->key_handlers.pop_back(); }

void PointViz::pop_mouse_button_handler() {
    pimpl->mouse_button_handlers.pop_back();
}
void PointViz::pop_scroll_handler() { pimpl->scroll_handlers.pop_back(); }

void PointViz::pop_mouse_pos_handler() { pimpl->mouse_pos_handlers.pop_back(); }

/*
 * Add / remove / access objects in the scene
 */
Camera& PointViz::camera() { return pimpl->camera_back; }

TargetDisplay& PointViz::target_display() { return pimpl->target; }

void PointViz::add(const std::shared_ptr<Cloud>& cloud) {
    pimpl->clouds.add(cloud);
}

void PointViz::add(const std::shared_ptr<Cuboid>& cuboid) {
    pimpl->cuboids.add(cuboid);
}

void PointViz::add(const std::shared_ptr<Label3d>& label) {
    pimpl->labels.add(label);
}

void PointViz::add(const std::shared_ptr<Image>& image) {
    pimpl->images.add(image);
}

bool PointViz::remove(const std::shared_ptr<Cloud>& cloud) {
    return pimpl->clouds.remove(cloud);
}

bool PointViz::remove(const std::shared_ptr<Cuboid>& cuboid) {
    return pimpl->cuboids.remove(cuboid);
}

bool PointViz::remove(const std::shared_ptr<Label3d>& label) {
    return pimpl->labels.remove(label);
}

bool PointViz::remove(const std::shared_ptr<Image>& image) {
    return pimpl->images.remove(image);
}

/*
 * Drawable types exposed to the user
 */
Cloud::Cloud(size_t w, size_t h, const double* xyz, const double* off,
             const double* extrinsic)
    : n_{w * h},
      w_{w},
      range_data_(n_, 0),
      key_data_(n_, 0),
      mask_data_(4 * n_, 0),
      xyz_data_(3 * n_, 0),
      off_data_(3 * n_, 0),
      transform_data_(12 * w, 0),
      palette_data_(3 * n_, 0) {
    std::copy(extrinsic, extrinsic + 16, extrinsic_.data());

    // initialize per-column poses to identity
    for (size_t v = 0; v < w; v++) {
        transform_data_[3 * v] = 1;
        transform_data_[3 * (v + w) + 1] = 1;
        transform_data_[3 * (v + 2 * w) + 2] = 1;
    }
    transform_changed_ = true;

    Eigen::Map<Eigen::Matrix4d>{map_pose_.data()}.setIdentity();
    map_pose_changed_ = true;

    set_xyz(xyz);
    set_offset(off);
    set_palette(&spezia[0][0], spezia_n);
}

void Cloud::clear() {
    range_changed_ = false;
    key_changed_ = false;
    mask_changed_ = false;
    xyz_changed_ = false;
    offset_changed_ = false;
    transform_changed_ = false;
    palette_changed_ = false;
    map_pose_changed_ = false;
}

void Cloud::set_range(const uint32_t* x) {
    std::copy(x, x + n_, range_data_.begin());
    range_changed_ = true;
}

void Cloud::set_key(const double* key_data) {
    std::copy(key_data, key_data + n_, key_data_.begin());
    key_changed_ = true;
}

void Cloud::set_mask(const float* mask_data) {
    std::copy(mask_data, mask_data + 4 * n_, mask_data_.begin());
    mask_changed_ = true;
}

void Cloud::set_xyz(const double* xyz) {
    for (size_t i = 0; i < n_; i++) {
        for (size_t k = 0; k < 3; k++) {
            xyz_data_[3 * i + k] = static_cast<GLfloat>(xyz[i + n_ * k]);
        }
    }
    xyz_changed_ = true;
}

void Cloud::set_offset(const double* offset) {
    for (size_t i = 0; i < n_; i++) {
        for (size_t k = 0; k < 3; k++) {
            off_data_[3 * i + k] = static_cast<GLfloat>(offset[i + n_ * k]);
        }
    }
    offset_changed_ = true;
}

void Cloud::set_point_size(float size) {
    point_size_ = size;
    point_size_changed_ = true;
}

void Cloud::set_pose(const mat4d& pose) {
    map_pose_ = pose;
    map_pose_changed_ = true;
}

void Cloud::set_column_poses(const double* rotation,
                             const double* translation) {
    for (size_t v = 0; v < w_; v++) {
        for (size_t u = 0; u < 3; u++) {
            for (size_t rgb = 0; rgb < 3; rgb++) {
                transform_data_[(u * w_ + v) * 3 + rgb] =
                    static_cast<GLfloat>(rotation[v + u * w_ + 3 * rgb * w_]);
            }
        }
        for (size_t rgb = 0; rgb < 3; rgb++) {
            transform_data_[9 * w_ + 3 * v + rgb] =
                static_cast<GLfloat>(translation[v + rgb * w_]);
        }
    }
    transform_changed_ = true;
}

void Cloud::set_palette(const float* palette, size_t palette_size) {
    palette_data_.resize(palette_size * 3);
    std::copy(palette, palette + (palette_size * 3), palette_data_.begin());
    palette_changed_ = true;
}

Image::Image() = default;

void Image::clear() {
    position_changed_ = false;
    image_changed_ = false;
    mask_changed_ = false;
}

void Image::set_image(size_t width, size_t height, const float* image_data) {
    const size_t n = width * height;
    image_data_.resize(n);
    image_width_ = width;
    image_height_ = height;
    std::copy(image_data, image_data + n, image_data_.begin());
    image_changed_ = true;
}

void Image::set_mask(size_t width, size_t height, const float* mask_data) {
    size_t n = width * height * 4;
    mask_data_.resize(n);
    mask_width_ = width;
    mask_height_ = height;
    std::copy(mask_data, mask_data + n, mask_data_.begin());
    mask_changed_ = true;
}

void Image::set_position(const std::array<float, 4>& pos) {
    position_ = pos;
    position_changed_ = true;
}

Cuboid::Cuboid(const mat4f& pose, const std::array<float, 4>& rgba) {
    set_pose(pose);
    set_rgba(rgba);
}

void Cuboid::clear() {
    pose_changed_ = false;
    rgba_changed_ = false;
}

void Cuboid::set_pose(const mat4f& pose) {
    pose_ = pose;
    pose_changed_ = true;
}

void Cuboid::set_rgba(const std::array<float, 4>& rgba) {
    rgba_ = rgba;
    rgba_changed_ = true;
}

Label3d::Label3d(const vec3d& position, const std::string& text) {
    set_position(position);
    set_text(text);
}

void Label3d::clear() {
    pos_changed_ = false;
    text_changed_ = false;
}

void Label3d::set_position(const vec3d& position) {
    position_ = position;
    pos_changed_ = true;
}

void Label3d::set_text(const std::string& text) {
    text_ = text;
    text_changed_ = true;
}

void TargetDisplay::enable_rings(bool state) { rings_enabled_ = state; }

void TargetDisplay::set_ring_size(int n) { ring_size_ = n; }

void add_default_controls(viz::PointViz& viz, std::mutex* mx) {
    bool orthographic = false;

    viz.push_key_handler(
        [=, &viz](const WindowCtx&, int key, int mods) mutable {
            auto lock = mx ? std::unique_lock<std::mutex>{*mx}
                           : std::unique_lock<std::mutex>{};
            if (mods == 0) {
                switch (key) {
                    case GLFW_KEY_W:
                        viz.camera().pitch(5);
                        viz.update();
                        break;
                    case GLFW_KEY_S:
                        viz.camera().pitch(-5);
                        viz.update();
                        break;
                    case GLFW_KEY_A:
                        viz.camera().yaw(5);
                        viz.update();
                        break;
                    case GLFW_KEY_D:
                        viz.camera().yaw(-5);
                        viz.update();
                        break;
                    case GLFW_KEY_EQUAL:
                        viz.camera().dolly(5);
                        viz.update();
                        break;
                    case GLFW_KEY_MINUS:
                        viz.camera().dolly(-5);
                        viz.update();
                        break;
                    case GLFW_KEY_0:
                        orthographic = !orthographic;
                        viz.camera().set_orthographic(orthographic);
                        viz.update();
                        break;
                    case GLFW_KEY_ESCAPE:
                        viz.running(false);
                        break;
                    default:
                        break;
                }
            } else if (mods == GLFW_MOD_SHIFT) {
                switch (key) {
                    case GLFW_KEY_R:
                        viz.camera().reset();
                        viz.update();
                        break;
                    default:
                        break;
                }
            }
            return true;
        });

    viz.push_scroll_handler([=, &viz](const WindowCtx&, double, double yoff) {
        auto lock = mx ? std::unique_lock<std::mutex>{*mx}
                       : std::unique_lock<std::mutex>{};
        viz.camera().dolly(yoff * 5);
        viz.update();
        return true;
    });

    viz.push_mouse_pos_handler(
        [=, &viz](const WindowCtx& wc, double xpos, double ypos) {
            auto lock = mx ? std::unique_lock<std::mutex>{*mx}
                           : std::unique_lock<std::mutex>{};
            double dx = (xpos - wc.mouse_x);
            double dy = (ypos - wc.mouse_y);
            // orbit or dolly in xy
            if (wc.lbutton_down) {
                constexpr double sensitivity = 0.3;
                viz.camera().yaw(sensitivity * dx);
                viz.camera().pitch(sensitivity * dy);
            } else if (wc.mbutton_down) {
                // convert from pixels to fractions of window size
                // TODO: factor out conversion?
                const double window_diagonal =
                    std::sqrt(wc.window_width * wc.window_width +
                              wc.window_height * wc.window_height);
                dx *= 2.0 / window_diagonal;
                dy *= 2.0 / window_diagonal;
                viz.camera().dolly_xy(dx, dy);
            }
            viz.update();
            return true;
        });
}

}  // namespace viz
}  // namespace ouster
