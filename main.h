#pragma once
// -*- coding: utf-8 -*-
// Main header for NeuralT + Neural Editor
// Type declarations used in main.cpp + paths and font helper.
// Author: justdieblondie

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstdio>

// ─────────────────────────────────────────
// Paths and version
// ─────────────────────────────────────────
inline constexpr const char* kAppVersion             = "v3.0";
inline constexpr const char* kEngineConfFile         = "engine.conf";
inline constexpr const char* kLessonsFile            = "lessons.json";
inline constexpr const char* kNetworkFile            = "network.json";
inline constexpr const char* kNetworkTrainedFile     = "network_trained.json";

// ─────────────────────────────────────────
// Minimal math
// ─────────────────────────────────────────
struct Vec2{ float x=0, y=0; };
struct Vec3{ float x=0, y=0, z=0; };

struct Mat4
{
    float m[16] = {0};
    static Mat4 identity();
    static Mat4 perspective(float fovy, float aspect, float znear, float zfar);
    static Mat4 ortho(float l, float ri, float b, float t, float n, float f);
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up);
    static Mat4 translate(const Vec3& v);
    static Mat4 rotateY(float a);
    static Mat4 rotateX(float a);
    static Mat4 scale(const Vec3& v);
    Mat4 operator*(const Mat4& o) const;
};

struct Timer
{
    double now=0, last=0, dt=0;
    void update(double currentTime);
};

struct Bounds3
{
    Vec3 min{ 1e9f, 1e9f, 1e9f };
    Vec3 max{-1e9f,-1e9f,-1e9f };
    void expand(const Vec3& p);
};

// ─────────────────────────────────────────
// Shaders / meshes
// ─────────────────────────────────────────
struct Shader
{
    unsigned int program = 0;
    bool compile(const char* vs, const char* fs, const char* gs, std::string* logOut);
    void use() const;
    void destroy();
    int  uniform(const char* name) const; // GLint
};

struct Vertex{ Vec3 pos; Vec3 normal; Vec2 uv; };

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    unsigned int vao=0, vbo=0, ebo=0;
    Bounds3 bounds;
    void upload();
    void draw() const;
    void destroy();
};

namespace Procedural
{
    Mesh makeCylinder(float R, float H, int seg, bool capped);
    Mesh makeBlade(float rootW, float tipW, float L, float T, int slices);
    Mesh makeGear(float innerR, float outerR, float toothDepth, int teeth, float width);
    Mesh merge(const std::vector<Mesh>& meshes);
    Mesh makeLineList(const std::vector<Vec3>& pts, const std::vector<uint32_t>& inds);
}

// ─────────────────────────────────────────
// Camera, UI theme
// ─────────────────────────────────────────
struct OrbitCamera
{
    float yaw=0, pitch=0, distance=3;
    Vec3  target{0,0,0};
    Mat4 view() const;
    void orbit(float dx, float dy);
    void dolly(float dd);
    void pan(float dx, float dy);
};

struct UITheme
{
    std::string name;
    Vec3 bg, panel, accent, grid, text, mesh;
    // Neural editor colors
    Vec3 nInput, nHidden, nOutput, nRing, nHighlight;
    // Edge gradient (start -> end) and style
    Vec3 edgeStart, edgeEnd;
    float edgeAlpha=0.85f;
    float edgeBase=0.6f, edgeScale=0.8f;

    // Five cohesive themes
    static UITheme NeonNoir();
    static UITheme DeepSpace();
    static UITheme SolarFlare();
    static UITheme Sunset();
    static UITheme EmberForge();
    static UITheme Crimson();
    static UITheme Ruby();
    static UITheme Cyberpunk();
};

// ─────────────────────────────────────────
// Turbine model
// ─────────────────────────────────────────
struct TurbineModel
{
    struct Params
    {
        float rpm=60;
        float current_rpm=0.0f;
        int   blades=3;
        float shaftLength=1.6f, shaftRadius=0.08f;
        float hubRadius=0.2f;
        float bladeLength=0.9f;
        float bladeRootWidth=0.25f, bladeTipWidth=0.12f, bladeThickness=0.02f;
    } params;
    Mesh hub, shaft, blades, gear, rivets, rotor, outerGear, outerGearR, pipes, schematicLines;
    int rotorTeeth=24, outerTeeth=36, outerTeethR=36;
    float outerGearOffset=0.0f, outerGearOffsetR=0.0f;
    void rebuild();
    void destroy();
};

// ─────────────────────────────────────────
// Neural network — data and graph
// ─────────────────────────────────────────
enum class LayerKind { Input, Hidden, Output };
enum class Act { Linear, ReLU, Sigmoid, Tanh };
namespace Activation
{
    static constexpr Act Linear  = Act::Linear;
    static constexpr Act ReLU    = Act::ReLU;
    static constexpr Act Sigmoid = Act::Sigmoid;
    static constexpr Act Tanh    = Act::Tanh;
}

const char* ActName(Act a);
Act NextAct(Act a);

struct Layer
{
    LayerKind kind = LayerKind::Hidden;
    std::string name;
    Act act = Act::Linear;
    std::vector<int> nodeIdx;
    float x0=0,y0=0,x1=0,y1=0;
    // Per-layer variation for the wind effect on connections.
    float windStrengthScale=1.0f;
    float windSpeedScale=1.0f;
    float windJitterScale=1.0f;
};

struct Neuron
{
    int id=0;
    bool isInput=false;
    std::string label;
    std::string desc = "default description";
    float value=0, speed=0, bias=0, radius=14;
    float amp=1.0f; // amplitude for live feed (0..1)
    bool live=false; // per-input live feed toggle
    float x=0, y=0;
    float vx=0, vy=0;
    float phase=0;
};

struct Edge{ int a=0, b=0; float w=0; };

struct NeuralGraph
{
    std::vector<Neuron> nodes;
    std::vector<Edge> edges;
    std::vector<Layer> layers;

    int selected=-1;
    int currentLayer=0;
    bool liveInputs=false;

    // Canvas presentation. Inputs are grouped by the prefix before '_' (NPC,
    // Scene, Act, ...); only one group is expanded at a time.
    std::string expandedInputGroup;
    bool collapseHiddenLayers=false;
    char editBuf[256]{};
    float lastCanvasW=0, lastCanvasH=0;

    Neuron* findAt(float mx, float my);
    void buildDemo();
    void layoutLayers(float W, float H);
    void tickLiveInputs(float dt);
    void stepLayout(float dt, float W, float H);
    void drawCanvas(bool& open, float width, float height);
};

// ─────────────────────────────────────────
// Application
// ─────────────────────────────────────────
struct UITheme; // forward above
struct Renderer
{
    UITheme theme;
    Shader meshShader, lineShader;
    unsigned int gridVao=0, gridVbo=0;
    bool init(const UITheme& t);
    void drawGrid(const Mat4& vp) const;
    void drawMesh(const Mesh& m, const Mat4& mvp, const Mat4& model, const Vec3& color, float metallic) const;
    void drawLines(const Mesh& lineList, const Mat4& mvp, const Vec3& color, float thickness) const;
    void destroy();
};

struct InputState
{
    bool mouseDownL=false, mouseDownM=false, mouseDownR=false;
    bool shift=false, alt=false, ctrl=false;
    double mouseX=0, mouseY=0;
};

struct AppConfig
{
    std::string title = "NeuralX";
    int width = 1280, height = 800;
    bool vsync = true;
    bool rotate = false;
    bool showSchematic = false;
    bool showUI = true;
};

struct Engine
{
    void* window = nullptr;
    Renderer renderer;
    TurbineModel turbine;
    OrbitCamera cam;
    Timer timer;
    AppConfig cfg;
    NeuralGraph graph;
    float rotorAngle = 0;

    void loadTurbinConfDefaults(const std::string& path);
    bool init(const AppConfig& cfgIn);
    void update();
    Mat4 proj() const;
    Mat4 view() const;
    void render();
    void shutdown();
    void onResize(int w, int h);
    void onMouseButton(int button, int action, int mods);
    void onCursorPos(double x, double y);
    void onScroll(double dx, double dy);
    void onKey(int key, int scancode, int action, int mods);
    void processCamera();
};

#ifdef USE_IMGUI

// Dear ImGui – load fonts with extended Latin range
#include "imgui.h"
inline void SetupDefaultFonts(float size_px = 16.0f)
{
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig cfg; 
    cfg.OversampleH = 3; 
    cfg.OversampleV = 2; 
    cfg.PixelSnapH = false;
    static const ImWchar ranges[] = { 0x0020, 0x00FF, 0x0100, 0x017F, 0 };
    const char* candidates[] = 
    {
        "DejaVuSans.ttf", "fonts/DejaVuSans.ttf", "NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf"
    };

    bool loaded=false;

    for(const char* p : candidates)
    {
        FILE* fp = std::fopen(p, "rb");
        if(fp)
        { 
            std::fclose(fp); 
            io.Fonts->AddFontFromFileTTF(p, size_px, &cfg, ranges); 
            loaded = true;
            break; 
        }
    }
    
    if(!loaded) io.Fonts->AddFontDefault();
}

#endif // USE_IMGUI
