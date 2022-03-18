/**
 * @file
 * @brief pybind wrappers for the ouster simple viz library
 *
 * PoC for exposing the opengl visualizer in Python.
 */
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <csignal>
#include <memory>
#include <utility>
#include <vector>

#include "ouster/colormaps.h"
#include "ouster/lidar_scan.h"
#include "ouster/point_viz.h"
#include "ouster/types.h"

namespace py = pybind11;
using namespace ouster;

PYBIND11_PLUGIN(_viz) {
    py::module m("_viz", R"(
    LidarScanViz bindings generated by pybind11.

    This module is generated from the C++ code and not meant to be used directly.
    )");

    // turn off signatures in docstrings: mypy stubs provide better types
    py::options options;
    options.disable_function_signatures();

    py::class_<viz::PointViz>(m, "PointViz")
        .def(py::init<const std::string&, bool, int, int>(), py::arg("name"),
             py::arg("fix_aspect") = false, py::arg("window_width") = 800,
             py::arg("window_height") = 600)

        .def(
            "run",
            [](viz::PointViz& self) {
                // acquire gil every n frames to check for signals
                const int check_every = 10;
                self.running(true);
                self.visible(true);
                while (self.running()) {
                    if (PyErr_CheckSignals() != 0)
                        throw py::error_already_set();
                    py::gil_scoped_release release;
                    for (int i = 0; i < check_every; i++) self.run_once();
                }
                self.visible(false);
            },
            R"(
             Display a visualizer window and run the rendering loop.

             Must be called from the main thread. Will return when ``quit()`` is called from
             another thread or when the visualizer window is closed. Note: this will replace
             the handler for SIGINT for the duration of the method call.
        )")

        .def("running", py::overload_cast<>(&viz::PointViz::running),
             "Check if the rendering loop is running.")

        .def("running", py::overload_cast<bool>(&viz::PointViz::running),
             "Shut down the visualizer and break out of the rendering loop.")

        .def("update", &viz::PointViz::update,
             "Show updated data in the next rendered frame.")

        // misc
        .def(
            "push_key_handler",
            [](viz::PointViz& self,
               std::function<bool(const viz::WindowCtx&, int, int)> f) {
                // pybind11 doesn't seem to deal with the rvalue ref arg
                // pybind11 already handles acquiring the GIL in the callback
                self.push_key_handler(std::move(f));
            },
            "Add a callback for handling keyboard input.")

        // control scene
        .def_property_readonly("camera", &viz::PointViz::camera,
                               py::return_value_policy::reference_internal,
                               "Get a reference to the camera controls.")

        .def_property_readonly("target_display", &viz::PointViz::target_display,
                               py::return_value_policy::reference_internal,
                               "Get a reference to the target display.")

        .def("add",
             py::overload_cast<const std::shared_ptr<viz::Cloud>&>(
                 &viz::PointViz::add),
             R"(
             Add an object to the scene.

             Args:
                 obj: A cloud, label, image or cuboid.)")
        .def("add", py::overload_cast<const std::shared_ptr<viz::Cuboid>&>(
                        &viz::PointViz::add))
        .def("add", py::overload_cast<const std::shared_ptr<viz::Label3d>&>(
                        &viz::PointViz::add))
        .def("add", py::overload_cast<const std::shared_ptr<viz::Image>&>(
                        &viz::PointViz::add))
        .def("remove",
             py::overload_cast<const std::shared_ptr<viz::Cloud>&>(
                 &viz::PointViz::remove),
             R"(
             Remove an object from the scene.

             Args:
                 obj: A cloud, label, image or cuboid.

             Returns:
                 True if the object was in the scene and was removed.
             )")
        .def("remove", py::overload_cast<const std::shared_ptr<viz::Cuboid>&>(
                           &viz::PointViz::add))
        .def("remove", py::overload_cast<const std::shared_ptr<viz::Label3d>&>(
                           &viz::PointViz::add))
        .def("remove", py::overload_cast<const std::shared_ptr<viz::Image>&>(
                           &viz::PointViz::add));

    m.def(
        "add_default_controls",
        [](viz::PointViz& viz) { viz::add_default_controls(viz); },
        "Add default keyboard and mouse bindings to a visualizer instance.");

    py::class_<viz::WindowCtx>(m, "WindowCtx")
        .def_readonly("lbutton_down", &viz::WindowCtx::lbutton_down)
        .def_readonly("mbutton_down", &viz::WindowCtx::mbutton_down)
        .def_readonly("mouse_x", &viz::WindowCtx::mouse_x)
        .def_readonly("mouse_y", &viz::WindowCtx::mouse_y)
        .def_readonly("window_width", &viz::WindowCtx::window_width)
        .def_readonly("window_height", &viz::WindowCtx::window_height);

    py::class_<viz::Camera>(m, "Camera")
        .def("reset", &viz::Camera::reset)
        .def("yaw", &viz::Camera::yaw)
        .def("pitch", &viz::Camera::pitch)
        .def("dolly", &viz::Camera::dolly)
        .def("dolly_xy", &viz::Camera::dolly_xy)
        .def("set_fov", &viz::Camera::set_fov)
        .def("set_orthographic", &viz::Camera::set_orthographic)
        .def("set_proj_offset", &viz::Camera::set_proj_offset);

    py::class_<viz::TargetDisplay>(m, "TargetDisplay")
        .def("enable_rings", &viz::TargetDisplay::enable_rings)
        .def("set_ring_size", &viz::TargetDisplay::set_ring_size);

    py::class_<viz::Cloud, std::shared_ptr<viz::Cloud>>(m, "Cloud")
        .def("__init__",
             [](viz::Cloud& self, const sensor::sensor_info& info) {
                 const auto xyz_lut = make_xyz_lut(info);
                 new (&self) viz::Cloud{
                     info.format.columns_per_frame,
                     info.format.pixels_per_column, xyz_lut.direction.data(),
                     xyz_lut.offset.data(), info.extrinsic.data()};
             })
        .def("set_range",
             [](viz::Cloud& self, py::array_t<uint32_t> range) {
                 if (range.ndim() != 2)
                     throw std::invalid_argument("Expected a 2d array");
                 if (static_cast<size_t>(range.size()) < self.get_size())
                     throw std::invalid_argument("Bad size");
                 self.set_range(range.data());
             })
        .def("set_key",
             [](viz::Cloud& self, py::array_t<double> key) {
                 if (key.ndim() != 2)
                     throw std::invalid_argument("Expected a 2d array");
                 if (static_cast<size_t>(key.size()) < self.get_size())
                     throw std::invalid_argument("Bad size");
                 self.set_key(key.data());
             })
        .def("set_point_size", &viz::Cloud::set_point_size)
        .def("set_palette", [](viz::Cloud& self, py::array_t<float> buf) {
            constexpr size_t palette_size = 256;
            if (static_cast<size_t>(buf.size()) != 3 * palette_size)
                throw std::invalid_argument("Bad size");
            self.set_palette(buf.data(), palette_size);
        });

    py::class_<viz::Image, std::shared_ptr<viz::Image>>(m, "Image")
        .def(py::init<>())
        .def("set_image",
             [](viz::Image& self, py::array_t<float> image) {
                 if (image.ndim() != 2)
                     throw std::invalid_argument("Expected a 2d array");
                 self.set_image(image.shape(1), image.shape(0), image.data());
             })
        .def("set_mask",
             [](viz::Image& self, py::array_t<float> buf) {
                 if (buf.ndim() != 3)
                     throw std::invalid_argument("Expected a 3d array");
                 if (buf.shape(2) != 4)
                     throw std::invalid_argument("Third dimension must be 4");
                 self.set_mask(buf.shape(1), buf.shape(0), buf.data());
             })
        .def("set_position",
             [](viz::Image& self, float x0, float x1, float y0, float y1) {
                 self.set_position({x0, x1, y1, y0});
             });

    py::class_<viz::Cuboid, std::shared_ptr<viz::Cuboid>>(m, "Cuboid")
        .def("__init__",
             [](viz::Cuboid& self, py::array_t<float> pose,
                py::array_t<float> rgba) {
                 // TODO: lots of duplication. std::array may be a poor choice
                 if (pose.size() != 16)
                     throw std::invalid_argument("Expected a 4x4 matrix");
                 viz::mat4f posea;
                 std::copy(pose.data(), pose.data() + 16, posea.data());

                 if (rgba.size() != 4)
                     throw std::invalid_argument("Expected a 4-element vector");
                 viz::vec4f rgbaa;
                 std::copy(rgba.data(), rgba.data() + 4, rgbaa.data());

                 new (&self) viz::Cuboid{posea, rgbaa};
             })
        .def("set_pose",
             [](viz::Cuboid& self, py::array_t<float> pose) {
                 if (pose.size() != 16)
                     throw std::invalid_argument("Expected a 4x4 matrix");
                 viz::mat4f posea;
                 std::copy(pose.data(), pose.data() + 16, posea.data());
                 self.set_pose(posea);
             })
        .def("set_rgba", [](viz::Cuboid& self, py::array_t<float> rgba) {
            if (rgba.size() != 4)
                throw std::invalid_argument("Expected a 4-element vector");
            viz::vec4f rgbaa;
            std::copy(rgba.data(), rgba.data() + 4, rgbaa.data());
            self.set_rgba(rgbaa);
        });

    py::class_<viz::Label3d, std::shared_ptr<viz::Label3d>>(m, "Label3d")
        .def("__init__",
             [](viz::Label3d& self, py::array_t<float> pos,
                const std::string& text) {
                 if (pos.size() != 3)
                     throw std::invalid_argument("Expected a 3-element vector");
                 viz::vec3d posa;
                 std::copy(pos.data(), pos.data() + 3, posa.data());
                 new (&self) viz::Label3d{posa, text};
             })
        .def("set_position",
             [](viz::Label3d& self, py::array_t<float> pos) {
                 if (pos.size() != 3)
                     throw std::invalid_argument("Expected a 3-element vector");
                 viz::vec3d posa;
                 std::copy(pos.data(), pos.data() + 3, posa.data());
                 self.set_position(posa);
             })
        .def("set_text", &viz::Label3d::set_text);

    m.attr("spezia_palette") = py::array_t<float>{{spezia_n, 3}, &spezia[0][0]};
    m.attr("calref_palette") = py::array_t<float>{{calref_n, 3}, &calref[0][0]};

    m.attr("__version__") = VERSION_INFO;

    return m.ptr();
}
