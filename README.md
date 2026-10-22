# NeuralT

NeuralT is a neural-network editor with an ImGui-based graphical interface.

## Requirements (Linux)

- `g++` compiler with C++17 support;
- GLFW;
- GLEW;
- OpenGL;
- nlohmann/json headers (`nlohmann-json3-dev` on Debian/Ubuntu);
- X11 system libraries: `dl`, `pthread`, `Xrandr`, `Xi`, and `m`.

The project also requires an `imgui/` directory containing ImGui sources and the GLFW/OpenGL3 backends.

## Build

Run the following command from the project directory:

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp \
  imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
  imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp \
  -Iimgui -Iimgui/backends -DUSE_IMGUI -DIMGUI_IMPL_OPENGL_LOADER_GLEW \
  -lglfw -lGLEW -lGL -ldl -lpthread -lX11 -lXrandr -lXi -lm -o neuralT
```

## Run

After a successful build, start the application with:

```bash
./neuralT
```

Press 'o' to load Neuro NPC example and F5 for more info. 
