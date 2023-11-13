# How-to setup

### Python plugin
The Python plugin needs to work in the Python environment that PyTorch uses,
which requires Python>3.7, and is currently not supported by the RECAST3D
binary.

If recompilation is not easy, the following is a quick fix.
Download the RECAST3D repo, initialize git submodules, and navigate to the
slicerecon directory. Then replace the `CMakeLists.txt` by the following.

```angular2html
cmake_minimum_required(VERSION 3.0)

include(FindPkgConfig)

project(slicerecon)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ZeroMQ
find_package(ZeroMQ QUIET)

if (ZeroMQ_FOUND)
    add_library(zmq INTERFACE)
    target_include_directories(zmq INTERFACE ${ZeroMQ_INCLUDE_DIR})
    target_link_libraries(zmq INTERFACE ${ZeroMQ_LIBRARY})
else()
    message("'zmq' not installed on the system, building from source...")

    execute_process(COMMAND git submodule update --init -- ../ext/libzmq
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    set(ZMQ_BUILD_TESTS OFF CACHE BOOL "disable tests" FORCE)
    set(WITH_PERF_TOOL OFF CACHE BOOL "disable perf-tools" FORCE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/../ext/libzmq libzmq)
    set(ZMQ_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../ext/libzmq/include)

    # ZeroMQ names their target libzmq, which is inconsistent => create a ghost dependency
    add_library(zmq INTERFACE)
    target_link_libraries(zmq INTERFACE libzmq)
endif()
# --------------------------------------------------------------------------------------------
# cppzmq

find_package(cppzmq QUIET)
if (NOT cppzmq_FOUND)
  execute_process(COMMAND git submodule update --init -- ../ext/cppzmq
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

  add_library(cppzmq INTERFACE)
  target_include_directories(cppzmq INTERFACE ../ext/cppzmq)
  target_link_libraries(cppzmq INTERFACE zmq)
endif()

# --------------------------------------------------------------------------------------------
# tomopackets
set(TOMOP_LIB_ONLY ON CACHE BOOL "build tomopackets as a library" FORCE)
add_subdirectory(../tomopackets tomopackets)
# --------------------------------------------------------------------------------------------

set(
  LIB_NAMES
  "zmq"
  "cppzmq"
  "tomop"
)

# --------------------------------------------------------------------------------------------
add_subdirectory("../ext/pybind11" pybind11)

set(BINDING_NAME "py_slicerecon")
set(BINDING_SOURCES "slicerecon/module.cpp")
pybind11_add_module(${BINDING_NAME} ${BINDING_SOURCES})
target_include_directories(${BINDING_NAME} PUBLIC "include")
target_link_libraries(${BINDING_NAME} PRIVATE tomop zmq cppzmq)
```
```c++
#pragma once                                                                                                                                                        
                                                                                                                                                                    
#include <string>                                                                                                                                                   
#include <thread>                                                                                                                                                   
                                                                                                                                                                    
#include "tomop/tomop.hpp"                                                                                                                                          
#include <zmq.hpp>                                                                                                                                                  
                                                                                                                                                                    
namespace slicerecon {                                                                                                                                              
                                                                                                                                                                    
class plugin {                                                                                                                                                      
  public:                                                                                                                                                           
    using slice_data_callback_type =                                                                                                                                
        std::function<                                                                                                                                              
            std::pair<std::array<int32_t, 2>,std::vector<float>>                                                                                                    
            (std::array<int32_t, 2>, std::vector<float>, int32_t)>;                                                                                                 
                                                                                                                                                                    
    using set_slice_callback_type =                                                                                                                                 
            std::function<void(int32_t, std::array<float, 9>)>;                                                                                                     
                                                                                                                                                                    
    using remove_slice_callback_type =                                                                                                                              
            std::function<void(int32_t)>;                                                                                                                           
                                                                                                                                                                    
    plugin(std::string hostname_in = "tcp://*:5650",                                                                                                                
           std::string hostname_out = "tcp://localhost:5555")                                                                                                       
        : context_(1), socket_in_(context_, ZMQ_REP),                                                                                                               
          socket_out_(context_, ZMQ_REQ) {                                                                                                                          
        std::cout << "Plugin: " << hostname_in                                                                                                                      
                  << " -> " << hostname_out << std::endl;                                                                                                           
                                                                                                                                                                    
        socket_in_.bind(hostname_in);                                                                                                                               
        socket_out_.connect(hostname_out);                                                                                                                          
    }                                                                                                                                                               
                                                                                                                                                                    
    ~plugin() { serve_thread_.join(); }                                                                                                                             
                                                                                                                                                                    
    void serve() {                                                                
        serve_thread_ = std::thread([&] { listen(); });                        
                                                                                  
        serve_thread_.join();
    }                                                                             
                                                                                                                                                                    
    void ack() {     
        zmq::message_t reply(sizeof(int));
        int success = 1;                                                          
        memcpy(reply.data(), &success, sizeof(int));
        socket_in_.send(reply);
    }            
                                                                                  
    void send(const tomop::Packet& packet) {                                 
        packet.send(socket_out_);                                                 
        zmq::message_t reply;
        socket_out_.recv(&reply);                                                 
    }                                                                             
                                         
    void set_slice_callback(set_slice_callback_type callback) {
        set_slice_callback_ = callback;                                           
    }                                                                             
                                                                                  
    void remove_slice_callback(remove_slice_callback_type callback) {
        remove_slice_callback_ = callback;
    }

    void slice_data_callback(slice_data_callback_type callback) {
        slice_data_callback_ = callback;
    }

    void listen() {
            std::cout << "Plugin starts listening"
                  << std::endl;

        while (true) {
            zmq::message_t update;
            bool kill = false;
            if (!socket_in_.recv(&update)) {
                kill = true;
            } else {
                ack();

                auto desc = ((tomop::packet_desc*)update.data())[0];
                auto buffer =
                    tomop::memory_buffer(update.size(), (char*)update.data());

                switch (desc) {
                case tomop::packet_desc::kill_scene: {
                    auto packet = std::make_unique<tomop::KillScenePacket>();
                    packet->deserialize(std::move(buffer));

                    kill = true;

                    // pass it along
                    send(*packet);

                    break;
                }
                case tomop::packet_desc::set_slice: {
                    auto packet = std::make_unique<tomop::SetSlicePacket>();
                    packet->deserialize(std::move(buffer));

                    if (!set_slice_callback_) {
                        throw tomop::server_error("No set_slice_callback set for plugin");
                    }

                    set_slice_callback_(packet->slice_id, packet->orientation);

                    break;
                }
                case tomop::packet_desc::remove_slice: {
                    auto packet = std::make_unique<tomop::RemoveSlicePacket>();
                    packet->deserialize(std::move(buffer));

                    if (!remove_slice_callback_) {
                        throw tomop::server_error("No remove_slice_callback set for plugin");
                    }

                    remove_slice_callback_(packet->slice_id);

                    break;
                }
                case tomop::packet_desc::slice_data: {
                    auto packet = std::make_unique<tomop::SliceDataPacket>();
                    packet->deserialize(std::move(buffer));

                    if (!slice_data_callback_) {
                        throw tomop::server_error("No callback set for plugin");
                    }

                    auto callback_data = slice_data_callback_(
                        packet->slice_size, std::move(packet->data),
                        packet->slice_id);

                    packet->slice_size = std::get<0>(callback_data);
                    packet->data = std::move(std::get<1>(callback_data));

                    send(*packet);
                    break;
                }

                    // TODO add support for registered parameters
                default:
                    break;
                }
            }

            if (kill) {
                std::cout << "Scene closed...\n";
                break;
            }
        }
    }

  private:
    zmq::context_t context_;
    zmq::socket_t socket_in_;
    zmq::socket_t socket_out_;
    std::thread serve_thread_;

    set_slice_callback_type set_slice_callback_;
    remove_slice_callback_type remove_slice_callback_;
    slice_data_callback_type slice_data_callback_;
};

} // namespace slicerecon
```
```c++
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <slicerecon/servers/plugin.hpp>

using namespace slicerecon;

PYBIND11_MODULE(py_slicerecon, m) {
    m.doc() = "bindings for slicerecon";

    py::class_<plugin>(m, "plugin")
        .def(py::init<std::string, std::string>(), "Initialize the plugin",
             py::arg("hostname_in") = "tcp://*:5650",
             py::arg("hostname_out") = "tcp://localhost:5555")
        .def("set_slice_callback", &plugin::set_slice_callback)
        .def("remove_slice_callback", &plugin::remove_slice_callback)
        .def("slice_data_callback", &plugin::slice_data_callback)
        .def("serve", &plugin::serve, py::call_guard<py::gil_scoped_release>())
      .def("listen", &plugin::listen, py::call_guard<py::gil_scoped_release>());
}
```
Modify `slicerecon/slicerecon/module.cpp` and change the `#include` to
`slicerecon/servers/plugin.hpp`. Then edit `plugin.hpp` so that it does not
depend on `util.hpp` anymore, i.e. removing special logging statements.

Afterwards, compile and install in the PyTorch environment with `pip install -e .`.

### SliceRecon
On the GPU cluster, install slicerecon in a python=3.7 environment with the default install instructions.
Start-up slicerecon server with
```
CUDA_VISIBLE_DEVICES=3 ./slicerecon_server --continuous --slice-size 350 \
--group-size 50 --preview-size 10 --pyplugin
```
 - Note: do not use the *=* symbol for arguments. Slicerecon ignores this.

### RECAST3D
On your own machine, install RECAST3D in a python=3.7 environment, again with default instructions.
Open ports on the remote server with:
```bash
ssh -4NR 5555:localhost:5555 scan_server
ssh -4NR 5556:localhost:5556 scan_server
```

Note: if one/both of the ports is not open or working, you will not see errors
in slicerecon. What will happen is that the preview/slices are missing from
the RECAST3D interface.

### Push data
In some environment of choice, install **tomopackets**. First, download the
RECAST3D repo on the machine that pushes. Init git submodules, and navigate
to the tomopackets dir, and run `pip install -e .` there.
```
PYTHONPATH=~/my_python_project_with_data/ \
python slicerecon_push.py \
/export/scratch3/adriaan/bigstore/handsoap_layeredSand/scan_1/ \
--settings=/export/scratch3/adriaan/bigstore/handsoap_layeredSand/pre_scan
```
 - Make sure that the voxel size in the script matches the voxel size of the 
recons during training by setting the appropriate slice-size and geometry.

### Other things

- SliceRecon does change the slice id when moved, so it is not possible to track
which slices is being moved. That is inconvenient, because we would like to know
on which slice to apply the DNN. 

 - Slicerecon must be recompiled with three extra lines in the visualization server
to pass on the packets of the SetSlicePacket to the plugin. Otherwise the orientation
is unknown by the plugin. The easiest fix is to compile from source, and building
only in the `slicerecon` dir. In my case, this required putting an own
`libastra.pc` in the _miniconda3/envs/live/lib/pkgconfig_ dir.

```bash
diff --git a/slicerecon/include/slicerecon/servers/visualization_server.hpp b/slicerecon/include/slicerecon/servers/visualization_server.hpp
index c23249f..a37cae8 100644
--- a/slicerecon/include/slicerecon/servers/visualization_server.hpp
+++ b/slicerecon/include/slicerecon/servers/visualization_server.hpp
@@ -186,6 +186,10 @@ class visualization_server : public listener, public util::bench_listener {
                         packet->deserialize(std::move(buffer));
 
                         make_slice(packet->slice_id, packet->orientation);
+
+                        if (plugin_socket_) {
+                            send(*packet, true);
+                        }
                         break;
                     }
                     case tomop::packet_desc::remove_slice: {

```