
// main.cpp  - NeuralT - Neural Editor
// Author: justdieblondie
#include <cfloat>
#include <cstdarg>
#include <cstddef>
#include <filesystem>
#include "main.h"
#include <cmath>

// GL & window
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>
#include <ctime>

// GLEW before GLFW
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef USE_IMGUI
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#endif

#include <stdint.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <optional>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <cerrno>
#include <unistd.h>
#include <unordered_map>
#include <limits>

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────
class Engine;
extern Engine* gEngine;

// ─────────────────────────────────────────────────────────────────────────────
// Small globals for UI
// ─────────────────────────────────────────────────────────────────────────────
#ifdef USE_IMGUI
static bool IsTrainingActive();
static bool gShowHelp = false;
static std::string gToastMsg; 
static double gToastT=0.0; 
static bool gShowTrainingWindow=false;
static bool gEditingLabel = false; // block shortcuts while editing labels
// Visibility toggles for windows/panels (start hidden)
static bool gShowEditorWindow = true;      // F1
static bool gShowNeuralTPanel = false;      // F2 (mirrors cfg.showUI)
static bool gShowLogWindow    = false;      // F3
static bool gShowParamsWindow = false;      // F4
static bool gShowLoadMenu     = false;      // 'O' key or button
static bool gForceRelayoutWindows = false;  // 'Home' key re-arrangement
static bool gShowSaveMenu     = false;      // 'S' key or button
static char gSaveName[256]    = "saved/network.json";

// Lesson targets never replace network predictions.
static std::unordered_map<int, float> gLessonTargets;
static unsigned int gLessonTargetRevision = 0;
static bool gPauseLiveInference = false;
static double gInferenceAccumulator = 0.0;
static bool gInferenceNeedsReset = false;
static constexpr double kInferenceStepSeconds = 1.0 / 30.0;

static void ClearLessonTargets()
{
    gLessonTargets.clear();
    ++gLessonTargetRevision;
}

struct TrainSample
{
    std::vector<float> in;
    std::vector<float> out;
    std::string desc = "Default";
    std::vector<std::vector<float>> sequence;
};
static std::vector<TrainSample> gDataset;
static int gLessonsCount = 0;

static std::vector<std::vector<float>> gSequenceDraft;
struct PendingTopologyChange
{
    NeuralGraph graph;
    std::vector<TrainSample> replacementLessons;
    std::function<void()> afterCommit;
};
static std::optional<PendingTopologyChange> gPendingTopology;
static bool gOpenTopologyPopup=false;
static void TopologyCommitted();

// New: Log window
static std::vector<std::string> gLogLines;
static bool gLogAutoScroll = true;
// Result overlay state
static bool gCelebrationVisible = false;
static bool gTargetReachedVisual = false;
static bool gTrainInterrupted = false;
static double gTrainDoneTime = 0.0;

static void Logf(const char* fmt, ...)
{
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    gLogLines.emplace_back(buf);
}
#else
// Fallback logger when IMGUI is not compiled in
static void Logf(const char* fmt, ...)
{
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "%s\n", buf);
}
#endif

struct WindParams 
{
    float strength = 150.0f;
    float speed = 1.0f;
    float jitter = 0.50f;
};
static WindParams gWind;

static float RandomRange(float minimum, float maximum)
{
    return minimum + (maximum - minimum) * (rand() / (float)RAND_MAX);
}

static void RandomizeLayerWind(Layer& layer)
{
    layer.windStrengthScale = RandomRange(0.65f, 1.35f);
    layer.windSpeedScale = RandomRange(0.70f, 1.30f);
    layer.windJitterScale = RandomRange(0.60f, 1.40f);
}

static void InitializeGRU(Layer& layer, int inputSize)
{
    const int hiddenSize = (int)layer.nodeIdx.size();
    layer.gruInputSize = inputSize;
    const float inputScale = std::sqrt(6.0f / std::max(1, inputSize + hiddenSize));
    const float recurrentScale = std::sqrt(3.0f / std::max(1, hiddenSize));
    auto randomVector = [](size_t count, float scale)
    {
        std::vector<float> values(count);
        for(float& value : values) value = RandomRange(-scale, scale);
        return values;
    };
    layer.gruWz = randomVector((size_t)hiddenSize * inputSize, inputScale);
    layer.gruWr = randomVector((size_t)hiddenSize * inputSize, inputScale);
    layer.gruWh = randomVector((size_t)hiddenSize * inputSize, inputScale);
    layer.gruUz = randomVector((size_t)hiddenSize * hiddenSize, recurrentScale);
    layer.gruUr = randomVector((size_t)hiddenSize * hiddenSize, recurrentScale);
    layer.gruUh = randomVector((size_t)hiddenSize * hiddenSize, recurrentScale);
    layer.gruBz.assign(hiddenSize, 0.0f);
    layer.gruBr.assign(hiddenSize, 0.0f);
    layer.gruBh.assign(hiddenSize, 0.0f);
    layer.gruState.assign(hiddenSize, 0.0f);
}

// Validate every parameter before indexing a GRU matrix or bias vector.
static bool ValidateGRUParameters(const NeuralGraph& graph, std::string* error)
{
    for(size_t li = 0; li < graph.layers.size(); ++li)
    {
        const Layer& layer = graph.layers[li];
        if(layer.kind != LayerKind::GRU) continue;
        const size_t hidden = layer.nodeIdx.size();
        const size_t input = li > 0 ? graph.layers[li - 1].nodeIdx.size() : 0;
        auto fail = [&](const std::string& detail)
        {
            if(error) *error = "GRU layer " + std::to_string(li + 1) + " (" + layer.name + "): " + detail;
            return false;
        };
        if(li == 0) return fail("requires a preceding input layer");
        if(layer.gruInputSize < 0 || (size_t)layer.gruInputSize != input)
            return fail("gru_input_size does not match the preceding layer");
        auto check = [&](const std::vector<float>& values, size_t expected, const char* name)
        {
            if(values.size() != expected)
                return fail(std::string(name) + " has " + std::to_string(values.size()) +
                            " values; expected " + std::to_string(expected));
            for(float value : values)
                if(!std::isfinite(value)) return fail(std::string(name) + " contains a non-finite value");
            return true;
        };
        if(!check(layer.gruWz, hidden * input, "gru_wz") ||
           !check(layer.gruWr, hidden * input, "gru_wr") ||
           !check(layer.gruWh, hidden * input, "gru_wh") ||
           !check(layer.gruUz, hidden * hidden, "gru_uz") ||
           !check(layer.gruUr, hidden * hidden, "gru_ur") ||
           !check(layer.gruUh, hidden * hidden, "gru_uh") ||
           !check(layer.gruBz, hidden, "gru_bz") ||
           !check(layer.gruBr, hidden, "gru_br") ||
           !check(layer.gruBh, hidden, "gru_bh")) return false;
    }
    return true;
}

static void ResetGRUState(NeuralGraph& graph)
{
    for(Layer& layer : graph.layers)
    {
        if(layer.kind == LayerKind::GRU)
        {
            layer.gruState.assign(layer.nodeIdx.size(), 0.0f);
        }
    }
}

static float gWindInterpolant = 0.0f;
static float gWindStrengthPhase = 0.0f;
static float gWindSpeedPhase = 0.0f;
static float gWindJitterPhase = 0.0f;

static bool gStrengthSineEnabled = false;
static bool gSpeedSineEnabled = false;
static bool gJitterSineEnabled = false;

// Live feed wave effect variables
static bool gLiveFeedWaveActive = true;
static float gLiveFeedWavePhase = 0.0f;
static float gLiveFeedWaveSpeed = 2.0f;
static int gLiveFeedCurrentInput = 0;
static float gLiveFeedInputDelay = 0.3f; // delay between inputs in seconds
static float gLiveFeedInputTimer = 0.0f;
static bool gRandomLiveFeedActive = false;

// ─────────────────────────────────────────────────────────────────────────────
// Math impl
// ─────────────────────────────────────────────────────────────────────────────

static inline float deg2rad(float d)
{ 
    return d * 0.017453292519943295769f; 
}

Mat4 Mat4::identity() 
{
    Mat4 r;
    r.m[0]=1; r.m[5]=1; r.m[10]=1; r.m[15]=1;
    return r;
}

Mat4 Mat4::perspective(float fovy, float aspect, float znear, float zfar) 
{
    float f = 1.0f / std::tan(fovy * 0.5f);
    Mat4 r{};
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}

Mat4 Mat4::ortho(float l, float ri, float b, float t, float n, float f) 
{
    Mat4 M = Mat4::identity();
    M.m[0]  = 2.0f/(ri-l);
    M.m[5]  = 2.0f/(t-b);
    M.m[10] = -2.0f/(f-n);
    M.m[12] = -(ri+l)/(ri-l);
    M.m[13] = -(t+b)/(t-b);
    M.m[14] = -(f+n)/(f-n);
    return M;
}

Mat4 Mat4::lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) 
{
    auto sub = [](Vec3 a, Vec3 b){ return Vec3{a.x-b.x,a.y-b.y,a.z-b.z}; };
    auto dot = [](Vec3 a, Vec3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; };
    auto norm = [&](Vec3 a){ float l=std::sqrt(std::max(dot(a,a), 1e-9f)); return Vec3{a.x/l,a.y/l,a.z/l}; };
    auto cross = [](Vec3 a, Vec3 b){ return Vec3{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; };

    Vec3 f = norm(sub(center, eye));
    Vec3 s = norm(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = Mat4::identity();
    r.m[0]=s.x; r.m[4]=s.y; r.m[8] =s.z;
    r.m[1]=u.x; r.m[5]=u.y; r.m[9] =u.z;
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z;
    r.m[12]=-(s.x*eye.x + s.y*eye.y + s.z*eye.z);
    r.m[13]=-(u.x*eye.x + u.y*eye.y + u.z*eye.z);
    r.m[14]= (f.x*eye.x + f.y*eye.y + f.z*eye.z);
    return r;
}

Mat4 Mat4::translate(const Vec3& v){ Mat4 r=Mat4::identity(); r.m[12]=v.x; r.m[13]=v.y; r.m[14]=v.z; return r; }
Mat4 Mat4::rotateY(float a){ Mat4 r=Mat4::identity(); float c=std::cos(a), s=std::sin(a); r.m[0]=c; r.m[2]=s; r.m[8]=-s; r.m[10]=c; return r; }
Mat4 Mat4::rotateX(float a){ Mat4 r=Mat4::identity(); float c=std::cos(a), s=std::sin(a); r.m[5]=c; r.m[6]=s; r.m[9]=-s; r.m[10]=c; return r; }
Mat4 Mat4::scale(const Vec3& v){ Mat4 r=Mat4::identity(); r.m[0]=v.x; r.m[5]=v.y; r.m[10]=v.z; return r; }

Mat4 Mat4::operator*(const Mat4& o) const 
{
    Mat4 r{};
    for(int c=0;c<4;c++){
        for(int rI=0;rI<4;rI++){
            r.m[c*4 + rI] = m[0*4 + rI]*o.m[c*4 + 0] +
                            m[1*4 + rI]*o.m[c*4 + 1] +
                            m[2*4 + rI]*o.m[c*4 + 2] +
                            m[3*4 + rI]*o.m[c*4 + 3];
        }
    }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utils
// ─────────────────────────────────────────────────────────────────────────────
void Timer::update(double currentTime){ now=currentTime; dt=now-last; last=now; }

void Bounds3::expand(const Vec3& p)
{
    min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y); min.z = std::min(min.z, p.z);
    max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y); max.z = std::max(max.z, p.z);
}

// ─────────────────────────────────────────────────────────────────────────────
// GL helpers
// ─────────────────────────────────────────────────────────────────────────────
static GLuint glCreateShaderChecked(GLenum type, const char* src, std::string* log) 
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, 0x8B81/*GL_COMPILE_STATUS*/, &ok);
    if(!ok){
        GLint len=0; glGetShaderiv(s, 0x8B84/*GL_INFO_LOG_LENGTH*/, &len);
        std::string buf(len, '\0'); glGetShaderInfoLog(s, len, nullptr, buf.data());
        if(log) *log += buf;
    }
    return s;
}

bool Shader::compile(const char* vs, const char* fs, const char* gs, std::string* logOut)
{
    std::string log;
    GLuint v = glCreateShaderChecked(0x8B31/*GL_VERTEX_SHADER*/, vs, &log);
    GLuint f = glCreateShaderChecked(0x8B30/*GL_FRAGMENT_SHADER*/, fs, &log);
    GLuint g = 0;
    if(gs) g = glCreateShaderChecked(0x8DD9/*GL_GEOMETRY_SHADER*/, gs, &log);

    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); if(gs) glAttachShader(p, g);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, 0x8B82/*GL_LINK_STATUS*/, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(p, 0x8B84/*GL_INFO_LOG_LENGTH*/, &len);
        std::string buf(len, '\0'); glGetProgramInfoLog(p, len, nullptr, buf.data());
        log += buf; if(logOut) *logOut = log;
        glDeleteProgram(p); glDeleteShader(v); glDeleteShader(f); if(gs) glDeleteShader(g);
        return false;
    }
    glDeleteShader(v); glDeleteShader(f); if(gs) glDeleteShader(g);
    program = p; if(logOut) *logOut = log;
    return true;
}

void Shader::use() const { glUseProgram(program); }
void Shader::destroy(){ if(program){ glDeleteProgram(program); program=0; } }
GLint Shader::uniform(const char* name) const { return glGetUniformLocation(program, name); }

// ─────────────────────────────────────────────────────────────────────────────
// Mesh
// ─────────────────────────────────────────────────────────────────────────────
void Mesh::upload()
{
    if(vao==0) glGenVertexArrays(1, &vao);
    if(vbo==0) glGenBuffers(1, &vbo);
    if(ebo==0) glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(0x8892/*GL_ARRAY_BUFFER*/, vbo);
    glBufferData(0x8892, vertices.size()*sizeof(Vertex), vertices.data(), 0x88E4/*STATIC*/);
    glBindBuffer(0x8893/*GL_ELEMENT_ARRAY_BUFFER*/, ebo);
    glBufferData(0x8893, indices.size()*sizeof(uint32_t), indices.data(), 0x88E4);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, 0x1406/*GL_FLOAT*/, false, sizeof(Vertex), (void*)offsetof(Vertex,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, 0x1406, false, sizeof(Vertex), (void*)offsetof(Vertex,normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, 0x1406, false, sizeof(Vertex), (void*)offsetof(Vertex,uv));
    glBindVertexArray(0);
    bounds = {};
    for(auto& v: vertices) bounds.expand(v.pos);
}

void Mesh::draw() const 
{
    glBindVertexArray(vao);
    glDrawElements(0x0004/*GL_TRIANGLES*/, (GLint)indices.size(), 0x1405/*UNSIGNED_INT*/, 0);
    glBindVertexArray(0);
}

void Mesh::destroy()
{
    if(ebo){ glDeleteBuffers(1,&ebo); ebo=0; }
    if(vbo){ glDeleteBuffers(1,&vbo); vbo=0; }
    if(vao){ glDeleteVertexArrays(1,&vao); vao=0; }
    vertices.clear(); indices.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Procedural geometry
// ─────────────────────────────────────────────────────────────────────────────
static void appendMesh(Mesh& dst, const Mesh& src)
{
    uint32_t base = (uint32_t)dst.vertices.size();
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
    for(auto i: src.indices) dst.indices.push_back(base + i);
}

Mesh Procedural::makeCylinder(float R, float H, int seg, bool capped)
{
    Mesh m;
    for(int i=0;i<=seg;i++)
    {
        float t = (float)i/seg;
        float ang = t * 2.f*3.1415926535f;
        float c=std::cos(ang), s=std::sin(ang);
        Vec3 n{c,0,s}; Vec3 p0{R*c,-H*0.5f,R*s}; Vec3 p1{R*c, H*0.5f,R*s};
        m.vertices.push_back({p0,n,{t,0}});
        m.vertices.push_back({p1,n,{t,1}});
    }

    for(int i=0;i<seg;i++)
    {
        uint32_t i0 = i*2, i1 = i*2+1, i2 = (i+1)*2, i3 = (i+1)*2+1;
        m.indices.insert(m.indices.end(), {i0,i2,i1, i1,i2,i3});
    }

    if(capped)
    {
        uint32_t centerIdx = (uint32_t)m.vertices.size();
        m.vertices.push_back({{0,-H*0.5f,0},{0,-1,0},{0.5f,0.5f}});

        for(int i=0;i<=seg;i++)
        {
            float t = (float)i/seg; float ang = t * 2.f*3.1415926535f;
            float c=std::cos(ang), s=std::sin(ang);
            Vec3 p{R*c,-H*0.5f,R*s};
            m.vertices.push_back({p,{0,-1,0},{(c+1)*0.5f,(s+1)*0.5f}});
        }

        for(int i=0;i<seg;i++) m.indices.insert(m.indices.end(), {centerIdx, centerIdx+1u+i, centerIdx+1u+i+1u});
        centerIdx = (uint32_t)m.vertices.size();
        m.vertices.push_back({{0, H*0.5f,0},{0,1,0},{0.5f,0.5f}});

        for(int i=0;i<=seg;i++)
        {
            float t = (float)i/seg; float ang = -t * 2.f*3.1415926535f;
            float c=std::cos(ang), s=std::sin(ang);
            Vec3 p{R*c, H*0.5f,R*s};
            m.vertices.push_back({p,{0,1,0},{(c+1)*0.5f,(s+1)*0.5f}});
        }
        for(int i=0;i<seg;i++) m.indices.insert(m.indices.end(), {centerIdx, centerIdx+1u+i, centerIdx+1u+i+1u});
    }
    m.upload(); return m;
}

Mesh Procedural::makeBlade(float rootW, float tipW, float L, float T, int slices)
{
    Mesh m;
    slices = std::max(8, slices);
    constexpr int chordSegments = 10;
    const int rowWidth = chordSegments + 1;

    for(int station = 0; station <= slices; ++station)
    {
        const float radial = (float)station / slices;
        const float smooth = radial * radial * (3.0f - 2.0f * radial);
        const float chord = rootW * (1.0f - smooth) + tipW * smooth;
        const float sweep = L * 0.10f * radial * radial;
        const float twist = 0.62f * (1.0f - radial) + 0.12f * radial;
        const float ct = std::cos(twist), st = std::sin(twist);
        for(int side = 0; side < 2; ++side)
        {
            const float sign = side == 0 ? 1.0f : -1.0f;
            for(int chordIndex = 0; chordIndex <= chordSegments; ++chordIndex)
            {
                const float u = (float)chordIndex / chordSegments;
                const float profile = std::sin(3.1415926535f * u);
                const float thickness = T * (0.08f + 0.92f * profile) * (1.0f - 0.35f * radial);
                const float camber = chord * 0.055f * std::sin(3.1415926535f * u) * (1.0f - 0.45f * radial);
                const float localX = (u - 0.43f) * chord + sweep;
                const float localY = camber + sign * thickness * 0.5f;
                const float x = localX * ct - localY * st;
                const float y = localX * st + localY * ct;
                Vec3 normal{-sign * st, sign * ct, 0.0f};
                m.vertices.push_back({{x,y,radial*L},normal,{u,radial}});
            }
        }
    }

    const int stationWidth = rowWidth * 2;
    for(int station = 0; station < slices; ++station)
    {
        for(int side = 0; side < 2; ++side)
        {
            for(int chordIndex = 0; chordIndex < chordSegments; ++chordIndex)
            {
                const uint32_t a = station * stationWidth + side * rowWidth + chordIndex;
                const uint32_t b = a + 1;
                const uint32_t c = a + stationWidth;
                const uint32_t d = c + 1;
                if(side == 0) m.indices.insert(m.indices.end(), {a,c,b, b,c,d});
                else m.indices.insert(m.indices.end(), {a,b,c, b,d,c});
            }
        }
        for(int edge : {0, chordSegments})
        {
            const uint32_t topA = station * stationWidth + edge;
            const uint32_t bottomA = station * stationWidth + rowWidth + edge;
            const uint32_t topB = topA + stationWidth;
            const uint32_t bottomB = bottomA + stationWidth;
            m.indices.insert(m.indices.end(), {topA,bottomA,topB, bottomA,bottomB,topB});
        }
    }
    for(int station : {0, slices})
    {
        const uint32_t base = station * stationWidth;
        for(int chordIndex = 0; chordIndex < chordSegments; ++chordIndex)
        {
            const uint32_t topA = base + chordIndex;
            const uint32_t topB = topA + 1;
            const uint32_t bottomA = base + rowWidth + chordIndex;
            const uint32_t bottomB = bottomA + 1;
            if(station == 0) m.indices.insert(m.indices.end(), {topA,topB,bottomA, topB,bottomB,bottomA});
            else m.indices.insert(m.indices.end(), {topA,bottomA,topB, topB,bottomA,bottomB});
        }
    }
    m.upload();
    return m;
}

Mesh Procedural::merge(const std::vector<Mesh>& meshes){ Mesh out; for(auto& m : meshes) appendMesh(out, m); out.upload(); return out; }

Mesh Procedural::makeLineList(const std::vector<Vec3>& pts, const std::vector<uint32_t>& inds)
{
    Mesh m; m.vertices.reserve(pts.size());
    for(auto& p: pts) m.vertices.push_back({p,{0,1,0},{0,0}});
    m.indices = inds; m.upload(); return m;
}

Mesh Procedural::makeGear(float innerR, float outerR, float toothDepth, int teeth, float width)
{
    Mesh m;
    teeth = std::max(8, teeth);
    constexpr int pointsPerTooth = 8;
    const int segments = teeth * pointsPerTooth;
    const float halfWidth = width * 0.5f;
    const float rootRadius = outerR;
    if(innerR <= 0.0f) innerR = rootRadius * 0.42f;

    auto toothRadius = [&](int segment)
    {
        const int phase = segment % pointsPerTooth;
        static const float shape[pointsPerTooth] = {0.0f,0.0f,0.42f,1.0f,1.0f,0.42f,0.0f,0.0f};
        return rootRadius + toothDepth * shape[phase];
    };
    for(int i = 0; i < segments; ++i)
    {
        const float angle = 2.0f * 3.1415926535f * i / segments;
        const float c = std::cos(angle), s = std::sin(angle);
        const float radius = toothRadius(i);
        m.vertices.push_back({{innerR*c,-halfWidth,innerR*s},{0,-1,0},{0,0}});
        m.vertices.push_back({{radius*c,-halfWidth,radius*s},{0,-1,0},{1,0}});
        m.vertices.push_back({{innerR*c, halfWidth,innerR*s},{0, 1,0},{0,1}});
        m.vertices.push_back({{radius*c, halfWidth,radius*s},{0, 1,0},{1,1}});
    }
    for(int i = 0; i < segments; ++i)
    {
        const int j = (i + 1) % segments;
        const uint32_t a = i*4, b = j*4;
        m.indices.insert(m.indices.end(), {
            a,a+1,b, a+1,b+1,b,
            a+2,b+2,a+3, a+3,b+2,b+3,
            a+1,a+3,b+1, a+3,b+3,b+1,
            a,b,a+2, a+2,b,b+2
        });
    }
    m.upload();
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────
Mat4 OrbitCamera::view() const 
{
    Vec3 eye{
        target.x + distance*std::cos(pitch)*std::sin(yaw),
        target.y + distance*std::sin(pitch),
        target.z + distance*std::cos(pitch)*std::cos(yaw)
    };
    return Mat4::lookAt(eye, target, {0,1,0});
}

void OrbitCamera::orbit(float dx, float dy){ yaw += dx*0.01f; pitch += dy*0.01f; pitch = std::max(-1.5f, std::min(1.5f, pitch)); }
void OrbitCamera::dolly(float dd){ distance = std::max(0.5f, distance * std::pow(0.95f, dd)); }
void OrbitCamera::pan(float dx, float dy){ float s = distance * 0.0015f; target.x += -dx*s; target.y +=  dy*s; }

// ─────────────────────────────────────────────────────────────────────────────
// Theme
// ─────────────────────────────────────────────────────────────────────────────
UITheme UITheme::NeonNoir()
{
    UITheme t{}; t.name="NeonNoir";
    t.bg={0.01f,0.01f,0.01f}; t.panel={0.08f,0.02f,0.02f}; t.accent={1.0f,0.0f,0.0f}; t.grid={0.4f,0.1f,0.1f}; t.text={1.0f,1.0f,1.0f}; t.mesh={0.9f,0.1f,0.1f};
    t.nInput={1.0f,0.2f,0.2f}; t.nHidden={0.8f,0.1f,0.1f}; t.nOutput={1.0f,0.5f,0.5f}; t.nRing={1.0f,1.0f,0.0f}; t.nHighlight={1.0f,0.0f,0.0f};
    t.edgeStart={1.0f,0.0f,0.0f}; t.edgeEnd={0.8f,0.2f,0.2f}; t.edgeAlpha=0.95f; t.edgeBase=0.7f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Cyberpunk()
{
    UITheme t{}; t.name="Cyberpunk";
    // Dark bluish background with neon cyan/magenta accents
    t.bg={0.04f,0.02f,0.07f}; t.panel={0.10f,0.05f,0.14f}; t.accent={0.05f,0.95f,0.95f}; t.grid={0.45f,0.55f,0.85f}; t.text={0.92f,0.96f,0.98f}; t.mesh={0.95f,0.20f,0.75f};
    t.nInput={0.10f,0.90f,0.90f}; t.nHidden={0.90f,0.25f,0.75f}; t.nOutput={0.95f,0.80f,0.20f}; t.nRing={1.0f,0.95f,0.45f}; t.nHighlight={0.95f,0.20f,0.75f};
    // Edge gradient cyan -> magenta
    t.edgeStart={0.05f,0.85f,0.95f}; t.edgeEnd={0.95f,0.20f,0.75f}; t.edgeAlpha=0.90f; t.edgeBase=0.7f; t.edgeScale=0.9f;
    return t;
}

UITheme UITheme::DeepSpace()
{
    UITheme t{}; t.name="DeepSpace";
    t.bg={0.02f, 0.01f, 0.05f}; t.panel={0.1f, 0.05f, 0.15f}; t.accent={0.1f, 0.9f, 0.9f}; t.grid={0.2f, 0.1f, 0.3f}; t.text={0.9f, 0.95f, 1.0f}; t.mesh={0.6f, 0.2f, 0.8f};
    t.nInput={0.1f, 0.9f, 0.9f}; t.nHidden={0.9f, 0.2f, 0.7f}; t.nOutput={0.5f, 0.9f, 0.9f}; t.nRing={1.0f, 1.0f, 0.0f}; t.nHighlight={0.1f, 1.0f, 1.0f};
    t.edgeStart={0.1f, 0.9f, 0.9f}; t.edgeEnd={0.9f, 0.2f, 0.7f}; t.edgeAlpha=1.0f; t.edgeBase=0.8f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::SolarFlare()
{
    UITheme t{}; t.name="SolarFlare";
    t.bg={0.1f,0.05f,0.0f}; t.panel={0.2f,0.1f,0.0f}; t.accent={1.0f,0.8f,0.0f}; t.grid={0.7f,0.3f,0.0f}; t.text={1.0f,0.95f,0.9f}; t.mesh={1.0f,0.5f,0.0f};
    t.nInput={1.0f,0.9f,0.2f}; t.nHidden={1.0f,0.5f,0.0f}; t.nOutput={1.0f,0.7f,0.1f}; t.nRing={1.0f,1.0f,0.5f}; t.nHighlight={1.0f,1.0f,0.0f};
    t.edgeStart={1.0f,0.8f,0.0f}; t.edgeEnd={1.0f,0.4f,0.0f}; t.edgeAlpha=0.9f; t.edgeBase=0.8f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Sunset()
{
    UITheme t{}; t.name="Sunset";
    t.bg={0.1f, 0.05f, 0.0f}; t.panel={0.3f, 0.15f, 0.05f}; t.accent={1.0f, 0.9f, 0.2f}; t.grid={0.5f, 0.2f, 0.1f}; t.text={1.0f, 0.95f, 0.85f}; t.mesh={1.0f, 0.5f, 0.1f};
    t.nInput={1.0f, 0.9f, 0.2f}; t.nHidden={1.0f, 0.3f, 0.1f}; t.nOutput={1.0f, 0.9f, 0.5f}; t.nRing={1.0f, 1.0f, 1.0f}; t.nHighlight={1.0f, 1.0f, 0.0f};
    t.edgeStart={1.0f, 0.9f, 0.2f}; t.edgeEnd={1.0f, 0.3f, 0.1f}; t.edgeAlpha=0.9f; t.edgeBase=0.7f; t.edgeScale=0.9f;
    return t;
}

UITheme UITheme::EmberForge()
{
    UITheme t{}; t.name="EmberForge";
    t.bg={0.02f,0.02f,0.03f}; t.panel={0.15f,0.08f,0.05f}; t.accent={1.0f,0.3f,0.0f}; t.grid={0.5f,0.2f,0.1f}; t.text={1.0f,0.9f,0.85f}; t.mesh={1.0f,0.4f,0.1f};
    t.nInput={1.0f,0.6f,0.2f}; t.nHidden={1.0f,0.2f,0.0f}; t.nOutput={1.0f,0.8f,0.3f}; t.nRing={1.0f,1.0f,0.0f}; t.nHighlight={1.0f,0.5f,0.0f};
    t.edgeStart={1.0f,0.2f,0.0f}; t.edgeEnd={1.0f,0.7f,0.1f}; t.edgeAlpha=1.0f; t.edgeBase=0.7f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Crimson()
{
    UITheme t{}; t.name="Crimson";
    t.bg={0.05f, 0.05f, 0.05f}; t.panel={0.1f, 0.1f, 0.1f}; t.accent={1.0f, 0.1f, 0.1f}; t.grid={0.3f, 0.3f, 0.3f}; t.text={0.9f, 0.9f, 0.9f}; t.mesh={0.8f, 0.8f, 0.8f};
    t.nInput={0.8f, 0.2f, 0.2f}; t.nHidden={0.6f, 0.6f, 0.6f}; t.nOutput={0.9f, 0.9f, 0.9f}; t.nRing={1.0f, 1.0f, 0.0f}; t.nHighlight={1.0f, 0.0f, 0.0f};
    t.edgeStart={0.1f, 0.1f, 0.1f}; t.edgeEnd={1.0f, 0.0f, 0.0f}; t.edgeAlpha=1.0f; t.edgeBase=0.8f; t.edgeScale=1.0f;
    return t;
}

UITheme UITheme::Ruby()
{
    UITheme t{}; t.name="Ruby";
    t.bg={0.08f, 0.02f, 0.02f}; t.panel={0.15f, 0.05f, 0.05f}; t.accent={0.9f, 0.1f, 0.3f}; t.grid={0.4f, 0.15f, 0.15f}; t.text={1.0f, 0.9f, 0.9f}; t.mesh={0.85f, 0.3f, 0.3f};
    t.nInput={1.0f, 0.3f, 0.4f}; t.nHidden={0.7f, 0.2f, 0.3f}; t.nOutput={1.0f, 0.5f, 0.6f}; t.nRing={1.0f, 0.8f, 0.2f}; t.nHighlight={1.0f, 0.2f, 0.4f};
    t.edgeStart={0.8f, 0.1f, 0.2f}; t.edgeEnd={1.0f, 0.4f, 0.6f}; t.edgeAlpha=0.9f; t.edgeBase=0.7f; t.edgeScale=1.0f;
    return t;
}

static Vec3 ColorHex(unsigned int value)
{
    return {
        ((value >> 16) & 0xff) / 255.0f,
        ((value >> 8) & 0xff) / 255.0f,
        (value & 0xff) / 255.0f
    };
}

static const std::array<GRUTheme, 16>& GRUThemes()
{
    // Compact two-colour harmonies adapted from combinations catalogued by
    // Sanzo Wada. Dark column grounds are derived from the same hues so the
    // palettes remain readable over every existing MLP theme.
    static const std::array<GRUTheme, 16> themes = {{
        {"English Red / Cerulean", ColorHex(0x24100d), ColorHex(0xd96629), ColorHex(0x0093a5),
            ColorHex(0xd96629), ColorHex(0x0093a5), ColorHex(0xffefae), ColorHex(0x0093a5)},
        {"Yellow Orange / Tyrian Blue", ColorHex(0x091724), ColorHex(0xf99d1b), ColorHex(0x12354e),
            ColorHex(0xf99d1b), ColorHex(0x4e7891), ColorHex(0xffefae), ColorHex(0x12354e)},
        {"Sienna / Pale Lemon", ColorHex(0x25190b), ColorHex(0xbb7125), ColorHex(0xffefae),
            ColorHex(0xbb7125), ColorHex(0xffefae), ColorHex(0xfff8d5), ColorHex(0xd59a43)},
        {"Isabella / Red Violet", ColorHex(0x1d1021), ColorHex(0xc5a56e), ColorHex(0x59256a),
            ColorHex(0xc5a56e), ColorHex(0x8d4d92), ColorHex(0xffefae), ColorHex(0x59256a)},
        {"Cossack Green / Vandar Blue", ColorHex(0x071b20), ColorHex(0x437742), ColorHex(0x8fc6d1),
            ColorHex(0x437742), ColorHex(0x064f6e), ColorHex(0xb5decc), ColorHex(0x8fc6d1)},
        {"Grenadine / Deep Indigo", ColorHex(0x090d21), ColorHex(0xf48067), ColorHex(0x7189c4),
            ColorHex(0xf48067), ColorHex(0x051230), ColorHex(0xffc2a9), ColorHex(0x7189c4)},
        {"Orange / Glaucous Green", ColorHex(0x12201b), ColorHex(0xf37420), ColorHex(0xb4cdc2),
            ColorHex(0xf37420), ColorHex(0xb4cdc2), ColorHex(0xffefae), ColorHex(0x78aa98)},
        {"Turquoise / Burnt Sienna", ColorHex(0x102322), ColorHex(0xb5decc), ColorHex(0xa34f2c),
            ColorHex(0xb5decc), ColorHex(0xa34f2c), ColorHex(0xf5ecc2), ColorHex(0xd07045)},
        {"Cinnamon Rufous / Lavender", ColorHex(0x20151d), ColorHex(0xa9522d), ColorHex(0xb7a8c7),
            ColorHex(0xa9522d), ColorHex(0xb7a8c7), ColorHex(0xf0d7c5), ColorHex(0x80638f)},
        {"Violet Red / Dull Blue Violet", ColorHex(0x1c1020), ColorHex(0xb0436d), ColorHex(0x71659a),
            ColorHex(0xb0436d), ColorHex(0x71659a), ColorHex(0xe8bfd2), ColorHex(0x9182bd)},
        {"Cinnamon Rufous / Dark Citrine", ColorHex(0x211b0e), ColorHex(0xa9522d), ColorHex(0x7f6b20),
            ColorHex(0xa9522d), ColorHex(0xb29a3a), ColorHex(0xf1d59a), ColorHex(0x7f6b20)},
        {"Ivory Buff / Mineral Gray", ColorHex(0x151a1b), ColorHex(0xebd3a2), ColorHex(0x607477),
            ColorHex(0xebd3a2), ColorHex(0x607477), ColorHex(0xfff2cf), ColorHex(0x8ba0a1)},
        {"Isabella / Green Blue", ColorHex(0x0d2020), ColorHex(0xc5a56e), ColorHex(0x16706f),
            ColorHex(0xc5a56e), ColorHex(0x16706f), ColorHex(0xf1dbad), ColorHex(0x43a09b)},
        {"Raw Sienna / Veronia Purple", ColorHex(0x1d1123), ColorHex(0xc8873e), ColorHex(0x604878),
            ColorHex(0xc8873e), ColorHex(0x82649d), ColorHex(0xf2c77f), ColorHex(0x604878)},
        {"Spinel Red / Naples Yellow", ColorHex(0x24131a), ColorHex(0xbe3b62), ColorHex(0xf0cf75),
            ColorHex(0xbe3b62), ColorHex(0xf0cf75), ColorHex(0xffe9a8), ColorHex(0xd65e7d)},
        {"Benzol Green / Grayish Lavender", ColorHex(0x111c1c), ColorHex(0x267b72), ColorHex(0xb7a8c7),
            ColorHex(0x267b72), ColorHex(0xb7a8c7), ColorHex(0xc9eee5), ColorHex(0x88779d)}
    }};
    return themes;
}

static size_t gGRUThemeIndex = 0;
static GRUTheme gGRUTheme = GRUThemes()[gGRUThemeIndex];
static int gPairedThemeIndex = -1;

static const std::array<size_t, 16>& GRUPairOrder()
{
    // Each existing MLP theme is paired with a contrasting Wada harmony.
    static const std::array<size_t, 16> order = {4, 2, 5, 0, 7, 6, 1, 3, 8, 9, 10, 11, 12, 13, 14, 15};
    return order;
}

static UITheme PairedMLPTheme(size_t index)
{
    if(index < 8)
    {
        switch(index)
        {
            case 0: return UITheme::NeonNoir();
            case 1: return UITheme::DeepSpace();
            case 2: return UITheme::SolarFlare();
            case 3: return UITheme::Sunset();
            case 4: return UITheme::EmberForge();
            case 5: return UITheme::Cyberpunk();
            case 6: return UITheme::Crimson();
            default: return UITheme::Ruby();
        }
    }

    const GRUTheme& harmony = GRUThemes()[GRUPairOrder()[index % 16]];
    UITheme theme{};
    theme.name = "WadaPair" + std::to_string(index + 1);
    theme.bg = {0.025f, 0.028f, 0.032f};
    theme.panel = {0.075f, 0.080f, 0.090f};
    theme.accent = harmony.unitBorder;
    theme.grid = {harmony.unitFill.x * 0.32f, harmony.unitFill.y * 0.32f, harmony.unitFill.z * 0.32f};
    theme.text = {0.94f, 0.93f, 0.88f};
    theme.mesh = harmony.unitFill;
    theme.nInput = harmony.unitBorder;
    theme.nHidden = harmony.unitFill;
    theme.nOutput = harmony.particle;
    theme.nRing = harmony.particle;
    theme.nHighlight = harmony.connectionEnd;
    theme.edgeStart = harmony.unitBorder;
    theme.edgeEnd = harmony.unitFill;
    theme.edgeAlpha = 0.90f;
    theme.edgeBase = 0.65f;
    theme.edgeScale = 0.90f;
    return theme;
}

static void ApplyPairedTheme(Renderer& renderer, size_t pairIndex)
{
    pairIndex %= 16;
    renderer.theme = PairedMLPTheme(pairIndex);
    gGRUThemeIndex = GRUPairOrder()[pairIndex];
    gGRUTheme = GRUThemes()[gGRUThemeIndex];
    gPairedThemeIndex = (int)pairIndex;
}

static Mesh MakeTubePath(const std::vector<Vec3>& path, float radius, int radialSegments)
{
    Mesh mesh;
    if(path.size() < 2) return mesh;
    radialSegments = std::max(8, radialSegments);
    auto normalize = [](Vec3 value)
    {
        const float length = std::sqrt(value.x*value.x + value.y*value.y + value.z*value.z);
        return length > 1e-6f ? Vec3{value.x/length,value.y/length,value.z/length} : Vec3{0,1,0};
    };
    auto cross = [](Vec3 a, Vec3 b)
    {
        return Vec3{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
    };
    for(size_t pointIndex = 0; pointIndex < path.size(); ++pointIndex)
    {
        const Vec3 before = path[pointIndex == 0 ? 0 : pointIndex - 1];
        const Vec3 after = path[pointIndex + 1 < path.size() ? pointIndex + 1 : pointIndex];
        const Vec3 tangent = normalize({after.x-before.x,after.y-before.y,after.z-before.z});
        const Vec3 reference = std::fabs(tangent.y) < 0.88f ? Vec3{0,1,0} : Vec3{1,0,0};
        const Vec3 side = normalize(cross(tangent, reference));
        const Vec3 up = normalize(cross(side, tangent));
        for(int segment = 0; segment < radialSegments; ++segment)
        {
            const float angle = 2.0f * 3.1415926535f * segment / radialSegments;
            const float c = std::cos(angle), s = std::sin(angle);
            const Vec3 normal{side.x*c+up.x*s,side.y*c+up.y*s,side.z*c+up.z*s};
            const Vec3& center = path[pointIndex];
            mesh.vertices.push_back({{center.x+normal.x*radius,center.y+normal.y*radius,center.z+normal.z*radius},
                                     normal,{(float)segment/radialSegments,(float)pointIndex/(path.size()-1)}});
        }
    }
    for(size_t pointIndex = 0; pointIndex + 1 < path.size(); ++pointIndex)
    {
        for(int segment = 0; segment < radialSegments; ++segment)
        {
            const int next = (segment + 1) % radialSegments;
            const uint32_t a = (uint32_t)(pointIndex * radialSegments + segment);
            const uint32_t b = (uint32_t)(pointIndex * radialSegments + next);
            const uint32_t c = a + radialSegments;
            const uint32_t d = b + radialSegments;
            mesh.indices.insert(mesh.indices.end(), {a,c,b, b,c,d});
        }
    }
    mesh.upload();
    return mesh;
}

static std::vector<Vec3> CubicPipe(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, int segments)
{
    std::vector<Vec3> path;
    path.reserve(segments + 1);
    for(int i = 0; i <= segments; ++i)
    {
        const float t = (float)i / segments;
        const float u = 1.0f - t;
        path.push_back({
            u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x,
            u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y,
            u*u*u*p0.z + 3*u*u*t*p1.z + 3*u*t*t*p2.z + t*t*t*p3.z
        });
    }
    return path;
}

static void AddGearHubAndSpokes(Mesh& gear, float innerRadius, float width, int spokeCount)
{
    Mesh hub = Procedural::makeCylinder(innerRadius*0.34f, width*1.18f, 48, true);
    appendMesh(gear, hub);
    hub.destroy();
    for(int spoke = 0; spoke < spokeCount; ++spoke)
    {
        const float angle = 2.0f * 3.1415926535f * spoke / spokeCount;
        const float c = std::cos(angle), s = std::sin(angle);
        Mesh bar = MakeTubePath({{c*innerRadius*0.28f,0,s*innerRadius*0.28f},
                                 {c*innerRadius*0.94f,0,s*innerRadius*0.94f}}, width*0.16f, 12);
        appendMesh(gear, bar);
        bar.destroy();
    }
    gear.upload();
}

// ─────────────────────────────────────────────────────────────────────────────
// Turbine
// ─────────────────────────────────────────────────────────────────────────────
void TurbineModel::rebuild()
{
    destroy();
    const float bodyR = params.hubRadius * 2.4f;
    const float bodyH = params.hubRadius * 2.0f;
    const float shaftR = params.shaftRadius * 1.35f;
    const float shaftH = params.shaftLength * 1.2f;
    shaft = Procedural::makeCylinder(shaftR, shaftH, 72, true);

    rotor.vertices.clear(); rotor.indices.clear(); rotor.vao=rotor.vbo=rotor.ebo=0;
    auto appendTemporary = [&](Mesh& destination, Mesh temporary)
    {
        appendMesh(destination, temporary);
        temporary.destroy();
    };

    // Layered casing, bearing collars and a raised central housing.
    appendTemporary(rotor, Procedural::makeCylinder(bodyR, bodyH, 96, true));
    Mesh lowerFlange = Procedural::makeCylinder(bodyR*1.13f, bodyH*0.13f, 96, true);
    for(Vertex& vertex : lowerFlange.vertices) vertex.pos.y -= bodyH*0.50f;
    appendTemporary(rotor, std::move(lowerFlange));
    Mesh upperFlange = Procedural::makeCylinder(bodyR*1.13f, bodyH*0.13f, 96, true);
    for(Vertex& vertex : upperFlange.vertices) vertex.pos.y += bodyH*0.50f;
    appendTemporary(rotor, std::move(upperFlange));
    appendTemporary(rotor, Procedural::makeCylinder(bodyR*0.42f, bodyH*1.35f, 64, true));

    // Matched pitch circles make the three gears meet at their working radii.
    rotorTeeth = 40; outerTeeth = 52; outerTeethR = 52;
    const float module = bodyR * 2.15f / rotorTeeth;
    const float rotorPitchR = module * rotorTeeth * 0.5f;
    const float outerPitchR = module * outerTeeth * 0.5f;
    const float rotorRootR = rotorPitchR - module * 1.05f;
    const float outerRootR = outerPitchR - module * 1.05f;
    const float toothDepth = module * 2.05f;
    Mesh ringGear = Procedural::makeGear(rotorPitchR*0.48f, rotorRootR, toothDepth,
                                         rotorTeeth, bodyH*0.30f);
    AddGearHubAndSpokes(ringGear, rotorPitchR*0.48f, bodyH*0.30f, 8);
    appendMesh(rotor, ringGear);
    ringGear.destroy();
    outerGear = Procedural::makeGear(outerPitchR*0.43f, outerRootR, toothDepth,
                                     outerTeeth, bodyH*0.28f);
    outerGearR = Procedural::makeGear(outerPitchR*0.43f, outerRootR, toothDepth,
                                      outerTeethR, bodyH*0.28f);
    AddGearHubAndSpokes(outerGear, outerPitchR*0.43f, bodyH*0.28f, 8);
    AddGearHubAndSpokes(outerGearR, outerPitchR*0.43f, bodyH*0.28f, 8);
    outerGearOffset = -(rotorPitchR + outerPitchR);
    outerGearOffsetR = rotorPitchR + outerPitchR;

    // Cambered, tapered and twisted propeller blades with a reinforced hub.
    Mesh blade = Procedural::makeBlade(params.bladeRootWidth*1.35f, params.bladeTipWidth*0.72f,
                                       std::max(params.bladeLength, bodyR*1.35f),
                                       std::max(params.bladeThickness*2.4f, bodyR*0.055f), 24);
    const int bladeCount = std::clamp(params.blades, 2, 10);
    const float bladeHeight = shaftH*0.5f + bodyR*0.16f;
    for(int bladeIndex = 0; bladeIndex < bladeCount; ++bladeIndex)
    {
        const float angle = 2.0f*3.1415926535f*bladeIndex/bladeCount;
        const float c = std::cos(angle), s = std::sin(angle);
        Mesh transformed = blade;
        for(Vertex& vertex : transformed.vertices)
        {
            const float x = vertex.pos.x, z = vertex.pos.z;
            const float nx = vertex.normal.x, nz = vertex.normal.z;
            vertex.pos = {c*x+s*z,vertex.pos.y+bladeHeight,-s*x+c*z};
            vertex.normal = {c*nx+s*nz,vertex.normal.y,-s*nx+c*nz};
        }
        appendMesh(rotor, transformed);
    }
    blade.destroy();
    Mesh propellerHub = Procedural::makeCylinder(bodyR*0.34f, bodyH*0.48f, 72, true);
    for(Vertex& vertex : propellerHub.vertices) vertex.pos.y += bladeHeight;
    appendTemporary(rotor, std::move(propellerHub));
    Mesh spinner = Procedural::makeCylinder(bodyR*0.19f, bodyH*0.62f, 72, true);
    for(Vertex& vertex : spinner.vertices) vertex.pos.y += bladeHeight + bodyH*0.34f;
    appendTemporary(rotor, std::move(spinner));
    rotor.upload();

    // Connected feed system: two manifolds, curved branches, return lines,
    // flanges and valve wheels all terminate at casing ports.
    pipes.vertices.clear(); pipes.indices.clear(); pipes.vao=pipes.vbo=pipes.ebo=0;
    const float pipeRadius = bodyR * 0.075f;
    const float manifoldX = bodyR * 2.65f;
    const float manifoldZ = bodyR * 1.75f;
    auto addTube = [&](const std::vector<Vec3>& path, float radius)
    {
        Mesh tube = MakeTubePath(path, radius, 20);
        appendMesh(pipes, tube);
        tube.destroy();
    };
    addTube({{-manifoldX,-bodyH*1.75f,-manifoldZ},{-manifoldX,bodyH*1.70f,-manifoldZ}},pipeRadius*1.25f);
    addTube({{ manifoldX,-bodyH*1.75f,-manifoldZ},{ manifoldX,bodyH*1.70f,-manifoldZ}},pipeRadius*1.25f);
    addTube({{-manifoldX,-bodyH*1.75f,-manifoldZ},{manifoldX,-bodyH*1.75f,-manifoldZ}},pipeRadius*1.10f);
    addTube({{-manifoldX, bodyH*1.70f,-manifoldZ},{manifoldX, bodyH*1.70f,-manifoldZ}},pipeRadius*1.10f);
    for(int side : {-1,1})
    {
        for(int level = 0; level < 3; ++level)
        {
            const float y = bodyH * (-0.75f + 0.75f*level);
            const Vec3 start{side*manifoldX,y,-manifoldZ};
            const Vec3 end{side*bodyR*0.88f,y*0.45f,-bodyR*0.34f};
            addTube(CubicPipe(start,{side*manifoldX,y,0},{side*bodyR*1.45f,end.y,-bodyR*0.65f},end,18),pipeRadius);
        }
        const Vec3 returnStart{side*manifoldX,-bodyH*1.15f,-manifoldZ};
        const Vec3 returnEnd{side*bodyR*0.62f,-bodyH*0.34f,bodyR*0.72f};
        addTube(CubicPipe(returnStart,{side*bodyR*2.7f,-bodyH*1.35f,bodyR*1.1f},
                               {side*bodyR*1.2f,-bodyH*0.55f,bodyR*1.25f},returnEnd,22),pipeRadius*0.82f);
    }
    addTube(CubicPipe({-manifoldX,bodyH*1.30f,-manifoldZ},{-bodyR*1.8f,bodyH*2.15f,-bodyR*0.4f},
                       {bodyR*1.8f,bodyH*2.15f,-bodyR*0.4f},{manifoldX,bodyH*1.30f,-manifoldZ},26),pipeRadius*0.90f);

    for(int side : {-1,1})
    {
        for(int level = 0; level < 3; ++level)
        {
            Mesh flange = Procedural::makeCylinder(pipeRadius*1.75f, pipeRadius*0.55f, 32, true);
            const float y = bodyH * (-0.75f + 0.75f*level);
            for(Vertex& vertex : flange.vertices)
            {
                vertex.pos.x += side*bodyR*0.94f;
                vertex.pos.y += y*0.45f;
                vertex.pos.z -= bodyR*0.34f;
            }
            appendTemporary(pipes, std::move(flange));
            Mesh valve = Procedural::makeGear(pipeRadius*0.35f,pipeRadius*1.35f,pipeRadius*0.38f,12,pipeRadius*0.30f);
            for(Vertex& vertex : valve.vertices)
            {
                vertex.pos.x += side*manifoldX;
                vertex.pos.y += y;
                vertex.pos.z -= manifoldZ;
            }
            appendTemporary(pipes, std::move(valve));
        }
    }
    pipes.upload();

    // Permanent fine steam traces remain decorative; the old technical-mode
    // duplicate rendering has been removed.
    std::vector<Vec3> steamPoints;
    std::vector<uint32_t> steamIndices;
    for(int side : {-1,1})
    {
        const uint32_t base = (uint32_t)steamPoints.size();
        for(int i=0;i<=18;++i)
        {
            const float t=(float)i/18.0f;
            steamPoints.push_back({side*(manifoldX+pipeRadius*1.4f),bodyH*(0.4f+1.4f*t),
                                   -manifoldZ+0.06f*std::sin(t*12.0f)});
            if(i>0){ steamIndices.push_back(base+i-1); steamIndices.push_back(base+i); }
        }
    }
    schematicLines = Procedural::makeLineList(steamPoints, steamIndices);
}

void TurbineModel::destroy()
{ 
    hub.destroy(); 
    shaft.destroy(); 
    blades.destroy(); 
    gear.destroy(); 
    rivets.destroy(); 
    rotor.destroy(); 
    outerGear.destroy(); 
    outerGearR.destroy(); 
    pipes.destroy(); 
    schematicLines.destroy(); 
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer
// ─────────────────────────────────────────────────────────────────────────────
static const char* kMeshVS = R"GLSL(
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNrm;
layout(location=2) in vec2 inUV;
uniform mat4 uMVP;
uniform mat4 uM;
out vec3 vN;
out vec3 vW;
void main(){
    vec4 w = uM * vec4(inPos,1.0);
    vW = w.xyz;
    vN = mat3(uM) * inNrm;
    gl_Position = uMVP * vec4(inPos,1.0);
}
)GLSL";

static const char* kMeshFS = R"GLSL(
#version 330 core
in vec3 vN;
in vec3 vW;
out vec4 outCol;
uniform vec3 uColor;
uniform float uMetallic;
uniform float uExposure;
void main(){
    vec3 N = normalize(vN);
    vec3 L = normalize(vec3(0.4,0.8,0.6));
    vec3 V = normalize(vec3(0.0,0.0,1.0));
    float ndl = max(dot(N,L), 0.0);
    float spec = pow(max(dot(reflect(-L,N),V),0.0), mix(32.0, 96.0, uMetallic));
    vec3 col = uColor * (0.2 + 0.8*ndl) + vec3(spec)*0.3;
    col = vec3(1.0) - exp(-col * uExposure);
    outCol = vec4(col, 1.0);
}
)GLSL";

static const char* kLineVS = R"GLSL(
#version 330 core
layout(location=0) in vec3 inPos;
uniform mat4 uMVP;
void main(){
    gl_Position = uMVP * vec4(inPos,1.0);
}
)GLSL";

static const char* kLineFS = R"GLSL(
#version 330 core
out vec4 outCol;
uniform vec3 uColor;
void main(){ outCol = vec4(uColor,1.0); }
)GLSL";

bool Renderer::init(const UITheme& t)
{
    theme = t;
    std::string log;
    if(!meshShader.compile(kMeshVS, kMeshFS, nullptr, &log)){ std::fprintf(stderr, "Mesh shader error:\n%s\n", log.c_str()); return false; }
    if(!lineShader.compile(kLineVS, kLineFS, nullptr, &log)){ std::fprintf(stderr, "Line shader error:\n%s\n", log.c_str()); return false; }

    // grid XZ
    std::vector<float> grid; const int N=20; const float S=0.5f;
    for(int i=-N;i<=N;i++)
    {
        grid.push_back((float)i*S); grid.push_back(0.0f); grid.push_back(-N*S);
        grid.push_back((float)i*S); grid.push_back(0.0f); grid.push_back( N*S);
        grid.push_back(-N*S); grid.push_back(0.0f); grid.push_back((float)i*S);
        grid.push_back( N*S); grid.push_back(0.0f); grid.push_back((float)i*S);
    }

    glGenVertexArrays(1,&gridVao);
    glGenBuffers(1,&gridVbo);
    glBindVertexArray(gridVao);
    glBindBuffer(0x8892/*GL_ARRAY_BUFFER*/, gridVbo);
    glBufferData(0x8892, grid.size()*sizeof(float), grid.data(), 0x88E4/*STATIC*/);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,0x1406/*GL_FLOAT*/,false,0,(void*)0);
    glBindVertexArray(0);
    return true;
}

void Renderer::drawGrid(const Mat4& vp) const 
{
    lineShader.use();
    glUniformMatrix4fv(lineShader.uniform("uMVP"), 1, false, vp.m);
    glUniform3f(lineShader.uniform("uColor"), theme.grid.x, theme.grid.y, theme.grid.z);
    glBindVertexArray(gridVao);
    glDrawArrays(0x0001/*GL_LINES*/, 0, (20*2+1)*4);
    glBindVertexArray(0);
}

void Renderer::drawMesh(const Mesh& m, const Mat4& mvp, const Mat4& model, const Vec3& color, float metallic) const 
{
    meshShader.use();
    glUniformMatrix4fv(meshShader.uniform("uMVP"), 1, false, mvp.m);
    glUniformMatrix4fv(meshShader.uniform("uM"),   1, false, model.m);
    glUniform3f(meshShader.uniform("uColor"), color.x, color.y, color.z);
    glUniform1f(meshShader.uniform("uMetallic"), metallic);
    glUniform1f(meshShader.uniform("uExposure"), 1.0f);
    m.draw();
}

void Renderer::drawLines(const Mesh& lineList, const Mat4& mvp, const Vec3& color, float thickness) const 
{
    glLineWidth(thickness);
    lineShader.use();
    glUniformMatrix4fv(lineShader.uniform("uMVP"), 1, false, mvp.m);
    glUniform3f(lineShader.uniform("uColor"), color.x, color.y, color.z);
    glBindVertexArray(lineList.vao);
    glDrawElements(0x0001/*GL_LINES*/, (GLint)lineList.indices.size(), 0x1405/*UNSIGNED_INT*/, 0);
    glBindVertexArray(0);
}

void Renderer::destroy()
{
    meshShader.destroy();
    lineShader.destroy();
    if(gridVbo){ glDeleteBuffers(1,&gridVbo); gridVbo = 0; }
    if(gridVao){ glDeleteVertexArrays(1,&gridVao); gridVao = 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuralGraph helpers
// ─────────────────────────────────────────────────────────────────────────────
#ifdef USE_IMGUI
static bool NG_ValidateTopology(const NeuralGraph& graph, std::string& error)
{
    auto fail=[&](const std::string& text){ error=text; return false; };
    if(graph.layers.size()<2 || graph.layers.front().kind!=LayerKind::Input ||
       graph.layers.back().kind!=LayerKind::Output) return fail("Input and Output layers are required.");
    std::vector<int> membership(graph.nodes.size(),-1);
    std::unordered_map<int,int> idLayer;
    for(size_t li=0;li<graph.layers.size();++li)
    {
        const auto& layer=graph.layers[li];
        if(layer.nodeIdx.empty()) return fail("A layer cannot be empty; delete the whole hidden layer instead.");
        if(li>0 && li+1<graph.layers.size() && layer.kind!=LayerKind::Hidden && layer.kind!=LayerKind::GRU)
            return fail("Internal layers must be Hidden or GRU.");
        for(int index:layer.nodeIdx)
        {
            if(index<0 || (size_t)index>=graph.nodes.size()) return fail("Invalid neuron index.");
            const auto& node=graph.nodes[index];
            if(membership[index]>=0 || !idLayer.emplace(node.id,(int)li).second)
                return fail("Duplicate neuron ID or layer membership.");
            membership[index]=(int)li;
            if(node.isInput!=(layer.kind==LayerKind::Input) || !std::isfinite(node.bias))
                return fail("Invalid neuron type or bias.");
        }
    }
    for(int li:membership) if(li<0) return fail("Neuron without a layer.");
    std::set<std::pair<int,int>> pairs;
    for(const auto& edge:graph.edges)
    {
        auto a=idLayer.find(edge.a),b=idLayer.find(edge.b);
        if(a==idLayer.end() || b==idLayer.end() || a->second>=b->second || !std::isfinite(edge.w))
            return fail("Invalid forward connection.");
        if(!pairs.emplace(edge.a,edge.b).second) return fail("Duplicate connection.");
        if(graph.layers[b->second].kind==LayerKind::GRU && a->second+1!=b->second)
            return fail("GRU inputs must come from the preceding layer.");
    }
    return ValidateGRUParameters(graph,&error);
}

static void NG_ReportTopologyError(const std::string& error)
{
    gToastMsg="Topology change rejected: "+error;
    gToastT=glfwGetTime();
    Logf("%s",gToastMsg.c_str());
}

static bool NG_CanEdit(const NeuralGraph& graph)
{
    if(IsTrainingActive() || gPendingTopology)
    { Logf("Finish training or the pending topology dialog first."); return false; }
    std::string error;
    if(!NG_ValidateTopology(graph,error)) { NG_ReportTopologyError(error); return false; }
    return true;
}

// Initial construction only. Editing uses NG_PreserveTopologyParameters below.
static void NG_RebuildEdges(NeuralGraph& graph)
{
    graph.edges.clear();
    for(size_t li=0;li+1<graph.layers.size();++li)
    {
        const float scale=std::sqrt(6.0f / std::max<size_t>(1,graph.layers[li].nodeIdx.size()+graph.layers[li+1].nodeIdx.size()));
        for(int a:graph.layers[li].nodeIdx) for(int b:graph.layers[li+1].nodeIdx)
            graph.edges.push_back({graph.nodes[a].id,graph.nodes[b].id,RandomRange(-scale,scale)});
    }
    ResetGRUState(graph);
}

static int NG_NextId(const NeuralGraph& graph)
{
    int maximum=-1;
    for(const auto& node:graph.nodes) maximum=std::max(maximum,node.id);
    if(maximum==std::numeric_limits<int>::max()) throw std::runtime_error("No free neuron ID remains.");
    return maximum+1;
}

static std::unordered_map<int,int> NG_LayerById(const NeuralGraph& graph)
{
    std::unordered_map<int,int> result;
    for(size_t li=0;li<graph.layers.size();++li)
        for(int index:graph.layers[li].nodeIdx) result.emplace(graph.nodes[index].id,(int)li);
    return result;
}

static std::unordered_map<int,size_t> NG_Slots(const NeuralGraph& graph,const Layer& layer)
{
    std::unordered_map<int,size_t> result;
    for(size_t i=0;i<layer.nodeIdx.size();++i) result.emplace(graph.nodes[layer.nodeIdx[i]].id,i);
    return result;
}

static void NG_PreserveTopologyParameters(const NeuralGraph& before,NeuralGraph& after)
{
    const auto oldLayers=NG_LayerById(before),newLayers=NG_LayerById(after);
    std::map<std::pair<int,int>,float> weights;
    for(const auto& edge:before.edges) weights.emplace(std::make_pair(edge.a,edge.b),edge.w);
    after.edges.clear();
    for(size_t li=0;li+1<after.layers.size();++li)
    {
        const auto& source=after.layers[li]; const auto& target=after.layers[li+1];
        const float scale=std::sqrt(6.0f / std::max<size_t>(1,source.nodeIdx.size()+target.nodeIdx.size()));
        for(int ai:source.nodeIdx) for(int bi:target.nodeIdx)
        {
            const int a=after.nodes[ai].id,b=after.nodes[bi].id;
            auto weight=weights.find({a,b});
            auto oldA=oldLayers.find(a),oldB=oldLayers.find(b);
            if(weight!=weights.end()) after.edges.push_back({a,b,weight->second});
            // Keep intentionally absent connections in unchanged adjacent layers.
            else if(oldA==oldLayers.end() || oldB==oldLayers.end() || oldA->second+1!=oldB->second)
                after.edges.push_back({a,b,RandomRange(-scale,scale)});
        }
    }
    // Retain explicit skip connections; an inserted layer replaces former adjacent edges.
    for(const auto& edge:before.edges)
    {
        auto a=newLayers.find(edge.a),b=newLayers.find(edge.b);
        if(a!=newLayers.end() && b!=newLayers.end() && a->second+1<b->second &&
           oldLayers.at(edge.a)+1<oldLayers.at(edge.b) && after.layers[b->second].kind!=LayerKind::GRU)
            after.edges.push_back(edge);
    }
    for(size_t li=1;li<after.layers.size();++li)
    {
        Layer& layer=after.layers[li];
        if(layer.kind!=LayerKind::GRU) continue;
        const Layer& input=after.layers[li-1];
        const size_t H=layer.nodeIdx.size(),I=input.nodeIdx.size();
        int oldIndex=-1;
        for(int index:layer.nodeIdx)
        {
            auto found=oldLayers.find(after.nodes[index].id);
            if(found!=oldLayers.end() && before.layers[found->second].kind==LayerKind::GRU)
            { oldIndex=found->second; break; }
        }
        if(oldIndex<0) { InitializeGRU(layer,(int)I); continue; }
        const Layer& old=before.layers[oldIndex];
        const auto oldUnits=NG_Slots(before,old),oldInputs=NG_Slots(before,before.layers[oldIndex-1]);
        const float inputScale=std::sqrt(6.0f/std::max<size_t>(1,I+H));
        const float recurrentScale=std::sqrt(3.0f/std::max<size_t>(1,H));
        auto resizeMatrix=[&](const std::vector<float>& values,bool recurrent)
        {
            const auto& columns=recurrent ? layer : input;
            const auto& oldColumns=recurrent ? oldUnits : oldInputs;
            const size_t width=columns.nodeIdx.size(),oldWidth=recurrent ? old.nodeIdx.size() : (size_t)old.gruInputSize;
            std::vector<float> result(H*width);
            for(size_t i=0;i<H;++i) for(size_t j=0;j<width;++j)
            {
                auto row=oldUnits.find(after.nodes[layer.nodeIdx[i]].id);
                auto column=oldColumns.find(after.nodes[columns.nodeIdx[j]].id);
                result[i*width+j]=row!=oldUnits.end() && column!=oldColumns.end()
                    ? values[row->second*oldWidth+column->second]
                    : RandomRange(-(recurrent ? recurrentScale : inputScale),recurrent ? recurrentScale : inputScale);
            }
            return result;
        };
        auto resizeBias=[&](const std::vector<float>& values)
        {
            std::vector<float> result(H,0.0f);
            for(size_t i=0;i<H;++i)
            {
                auto row=oldUnits.find(after.nodes[layer.nodeIdx[i]].id);
                if(row!=oldUnits.end()) result[i]=values[row->second];
            }
            return result;
        };
        layer.gruWz=resizeMatrix(old.gruWz,false); layer.gruWr=resizeMatrix(old.gruWr,false); layer.gruWh=resizeMatrix(old.gruWh,false);
        layer.gruUz=resizeMatrix(old.gruUz,true); layer.gruUr=resizeMatrix(old.gruUr,true); layer.gruUh=resizeMatrix(old.gruUh,true);
        layer.gruBz=resizeBias(old.gruBz); layer.gruBr=resizeBias(old.gruBr); layer.gruBh=resizeBias(old.gruBh);
        layer.gruInputSize=(int)I;
    }
    ResetGRUState(after);
    for(auto& layer:after.layers)
    {
        layer.gruLastInput.clear(); layer.gruLastPrevious.clear(); layer.gruLastUpdate.clear();
        layer.gruLastReset.clear(); layer.gruLastCandidate.clear();
    }
    for(auto& node:after.nodes) if(!node.isInput) node.value=0;
}

static void NG_CommitTopology(NeuralGraph& destination,PendingTopologyChange change)
{
    destination=std::move(change.graph);
    gDataset=std::move(change.replacementLessons);
    gLessonsCount=(int)gDataset.size();
    gSequenceDraft.clear();
    ClearLessonTargets();
    gInferenceAccumulator=0; gInferenceNeedsReset=true;
    gLiveFeedCurrentInput=0;
    TopologyCommitted();
    if(change.afterCommit) change.afterCommit();
    Logf("Topology changed; previous lessons and sequence draft cleared.");
}

static bool NG_RequestTopology(NeuralGraph& destination,NeuralGraph candidate,
                               std::vector<TrainSample> replacementLessons={},std::function<void()> afterCommit={})
{
    if(IsTrainingActive() || gPendingTopology) return false;
    std::string error;
    if(!NG_ValidateTopology(candidate,error)) { NG_ReportTopologyError(error); return false; }
    candidate.selected=-1; candidate.currentLayer=std::clamp(candidate.currentLayer,0,(int)candidate.layers.size()-1);
    std::memset(candidate.editBuf,0,sizeof(candidate.editBuf));
    candidate.layoutLayers(destination.lastCanvasW>0 ? destination.lastCanvasW : 800.0f,
                           destination.lastCanvasH>0 ? destination.lastCanvasH : 500.0f);
    PendingTopologyChange change{std::move(candidate),std::move(replacementLessons),std::move(afterCommit)};
    if(!gDataset.empty() || !gSequenceDraft.empty())
    {
        gPendingTopology=std::move(change);
        gOpenTopologyPopup=true;
    }
    else NG_CommitTopology(destination,std::move(change));
    return true;
}

static void NG_AppendNeuron(NeuralGraph& graph,int layerIndex)
{
    Layer& layer=graph.layers[layerIndex];
    Neuron neuron;
    neuron.id=NG_NextId(graph); neuron.isInput=layer.kind==LayerKind::Input;
    neuron.label=(neuron.isInput ? "In" : layer.kind==LayerKind::Output ? "Out" : "H")+std::to_string(layer.nodeIdx.size()+1);
    graph.nodes.push_back(std::move(neuron));
    layer.nodeIdx.push_back((int)graph.nodes.size()-1);
}

static void NG_AddNeuron(NeuralGraph& graph,int layerIndex)
{
    if(!NG_CanEdit(graph) || layerIndex<0 || layerIndex>=(int)graph.layers.size()) return;
    try
    {
        NeuralGraph candidate=graph;
        NG_AppendNeuron(candidate,layerIndex);
        NG_PreserveTopologyParameters(graph,candidate);
        NG_RequestTopology(graph,std::move(candidate));
    }
    catch(const std::exception& error) { NG_ReportTopologyError(error.what()); }
}

static void NG_EraseNode(NeuralGraph& graph,int index)
{
    graph.nodes.erase(graph.nodes.begin()+index);
    for(auto& layer:graph.layers) for(int& nodeIndex:layer.nodeIdx) if(nodeIndex>index) --nodeIndex;
}

static void NG_RemoveLastNeuron(NeuralGraph& graph,int layerIndex)
{
    if(!NG_CanEdit(graph) || layerIndex<0 || layerIndex>=(int)graph.layers.size()) return;
    if(graph.layers[layerIndex].nodeIdx.size()<=1)
    { NG_ReportTopologyError("Cannot remove the last neuron. Delete the hidden layer instead."); return; }
    NeuralGraph candidate=graph;
    int index=candidate.layers[layerIndex].nodeIdx.back();
    candidate.layers[layerIndex].nodeIdx.pop_back();
    NG_EraseNode(candidate,index);
    NG_PreserveTopologyParameters(graph,candidate);
    NG_RequestTopology(graph,std::move(candidate));
}

static void NG_InsertLayer(NeuralGraph& graph,LayerKind kind)
{
    if(!NG_CanEdit(graph)) return;
    const int current=graph.currentLayer;
    if(current<0 || current>=(int)graph.layers.size() || graph.layers[current].kind==LayerKind::Output) return;
    try
    {
        NeuralGraph candidate=graph;
        Layer layer; layer.kind=kind; layer.act=kind==LayerKind::GRU ? Act::Tanh : Act::ReLU;
        layer.name=std::string(kind==LayerKind::GRU ? "GRU " : "HL ")+std::to_string(current+1);
        RandomizeLayerWind(layer);
        candidate.layers.insert(candidate.layers.begin()+current+1,std::move(layer));
        candidate.currentLayer=current+1;
        for(int unit=0;unit<(kind==LayerKind::GRU ? 8 : 1);++unit) NG_AppendNeuron(candidate,current+1);
        NG_PreserveTopologyParameters(graph,candidate);
        NG_RequestTopology(graph,std::move(candidate));
    }
    catch(const std::exception& error) { NG_ReportTopologyError(error.what()); }
}

static void NG_InsertHiddenAfterCurrent(NeuralGraph& graph) { NG_InsertLayer(graph,LayerKind::Hidden); }
static void NG_InsertGRUAfterCurrent(NeuralGraph& graph) { NG_InsertLayer(graph,LayerKind::GRU); }

static bool NG_RemoveCurrentHiddenLayer(NeuralGraph& graph)
{
    if(!NG_CanEdit(graph)) return false;
    const int current=graph.currentLayer;
    if(current<0 || current>=(int)graph.layers.size()) return false;
    const LayerKind kind=graph.layers[current].kind;
    if(kind!=LayerKind::Hidden && kind!=LayerKind::GRU) { Logf("Only hidden or GRU layers can be deleted."); return false; }
    NeuralGraph candidate=graph;
    auto removed=candidate.layers[current].nodeIdx;
    candidate.layers.erase(candidate.layers.begin()+current);
    std::sort(removed.begin(),removed.end(),std::greater<int>());
    for(int index:removed) NG_EraseNode(candidate,index);
    candidate.currentLayer=current-1;
    NG_PreserveTopologyParameters(graph,candidate);
    return NG_RequestTopology(graph,std::move(candidate));
}

// Build a specific multi-layer topology
static void NG_BuildTopology(NeuralGraph& g, int inN, const std::vector<int>& hidden, int outN)
{
    if(IsTrainingActive()) { Logf("Stop training before changing the model or lessons."); return; }
    g.nodes.clear(); g.edges.clear(); g.layers.clear();
    g.selected = -1; std::memset(g.editBuf,0,sizeof(g.editBuf));
    g.currentLayer = 0; g.liveInputs = false;
    int id = 0;

    // Inputs
    {
        Layer L; L.kind = LayerKind::Input; L.name = "Inputs"; L.act = Act::Linear;
        RandomizeLayerWind(L);
        for(int i=0;i<inN;i++){ Neuron n; n.id=id++; n.isInput=true; n.label="In"+std::to_string(i+1); n.speed=0.0f; n.amp=1.0f; g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }

    // Hidden layers
    for(size_t li=0; li<hidden.size(); ++li)
    {
        Layer L; L.kind = LayerKind::Hidden; L.name = "HL "+std::to_string((int)li+1); L.act = Act::ReLU;
        RandomizeLayerWind(L);
        for(int i=0;i<hidden[li]; ++i){ Neuron n; n.id=id++; n.isInput=false; n.label="H"+std::to_string((int)li+1)+"."+std::to_string(i+1); g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }

    // Outputs
    {
        Layer L; L.kind = LayerKind::Output; L.name = "Outputs"; L.act = Act::Sigmoid;
        RandomizeLayerWind(L);
        for(int i=0;i<outN;i++){ Neuron n; n.id=id++; n.isInput=false; n.label="Out"+std::to_string(i+1); g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }
    NG_RebuildEdges(g);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f, g.lastCanvasH>0?g.lastCanvasH:500.0f);
}

static void NG_BuildDinosaur(NeuralGraph& g) 
{
    NG_BuildTopology(g, 3, {5, 8}, 3);
    // Custom labels for T-Rex
    if (g.layers.size() >= 4) {
        // Inputs
        g.nodes[g.layers[0].nodeIdx[0]].label = "ProximityToPrey";
        g.nodes[g.layers[0].nodeIdx[1]].label = "HungerLevel";
        g.nodes[g.layers[0].nodeIdx[2]].label = "ThreatLevel";
        // Outputs
        g.nodes[g.layers[3].nodeIdx[0]].label = "Attack";
        g.nodes[g.layers[3].nodeIdx[1]].label = "Flee";
        g.nodes[g.layers[3].nodeIdx[2]].label = "Roar";
    }
}

static void NG_BuildDeepGRUTopology(NeuralGraph& graph)
{
    if(IsTrainingActive()) { Logf("Stop training before changing the model or lessons."); return; }
    graph.nodes.clear();
    graph.edges.clear();
    graph.layers.clear();
    graph.selected = -1;
    graph.currentLayer = 0;
    int nextId = 0;

    auto addLayer = [&](LayerKind kind, int count, const std::string& name, Act activation)
    {
        Layer layer;
        layer.kind = kind;
        layer.name = name;
        layer.act = activation;
        RandomizeLayerWind(layer);
        const int layerNumber = (int)graph.layers.size();
        for(int i = 0; i < count; ++i)
        {
            Neuron neuron;
            neuron.id = nextId++;
            neuron.isInput = kind == LayerKind::Input;
            const char* prefix = kind == LayerKind::Input ? "Sensor" :
                (kind == LayerKind::GRU ? "Memory" : (kind == LayerKind::Output ? "State" : "Deep"));
            neuron.label = std::string(prefix) + std::to_string(layerNumber) + "." + std::to_string(i + 1);
            graph.nodes.push_back(std::move(neuron));
            layer.nodeIdx.push_back((int)graph.nodes.size() - 1);
        }
        graph.layers.push_back(std::move(layer));
    };

    // 12 + 48 + 40 + 32 + 48 + 8 = 188 neurons.
    addLayer(LayerKind::Input, 12, "Factory sensors", Act::Linear);
    addLayer(LayerKind::Hidden, 48, "Deep features", Act::ReLU);
    addLayer(LayerKind::GRU, 40, "Temporal memory A (GRU)", Act::Tanh);
    InitializeGRU(graph.layers.back(), 48);
    addLayer(LayerKind::GRU, 32, "Temporal memory B (GRU)", Act::Tanh);
    InitializeGRU(graph.layers.back(), 40);
    addLayer(LayerKind::Hidden, 48, "Decision features", Act::ReLU);
    addLayer(LayerKind::Output, 8, "Machine state", Act::Sigmoid);
    NG_RebuildEdges(graph);
    for(Edge& edge : graph.edges) edge.w = RandomRange(-0.35f, 0.35f);
    for(Layer& layer : graph.layers)
    {
        if(layer.kind == LayerKind::Hidden || layer.kind == LayerKind::Output)
            for(int index : layer.nodeIdx) graph.nodes[index].bias = RandomRange(-0.12f, 0.12f);
    }
    graph.layoutLayers(graph.lastCanvasW > 0 ? graph.lastCanvasW : 1200.0f,
                       graph.lastCanvasH > 0 ? graph.lastCanvasH : 700.0f);
}
// Extra helpers: save to JSON + randomize hidden (weights & biases)


static float ng_frand(float a, float b){ return a + (b-a) * (std::rand() / (float)RAND_MAX); }

// SFINAE helpers to get/set bias if Neuron has such field
template<typename T>
auto ng_get_bias(const T& n) -> decltype(n.bias) { return n.bias; }
[[maybe_unused]] static float ng_get_bias(...) { return 0.0f; }

template<typename T>
auto ng_set_bias(T& n, float v) -> decltype(n.bias = v, void()) { n.bias = v; }
[[maybe_unused]] static void ng_set_bias(...) { /* no bias present */ }

static void NG_RandomizeHidden(NeuralGraph& g, float wRange=1.0f, float bRange=0.5f)
{
    if(IsTrainingActive()) { Logf("Stop training before changing the model or lessons."); return; }
    for(const auto& L : g.layers)
    {
        if(L.kind != LayerKind::Hidden && L.kind != LayerKind::Output) continue;
        for(int idxN : L.nodeIdx)
        {
            if(idxN < 0 || idxN >= (int)g.nodes.size()) continue;
            Neuron& n = g.nodes[idxN];
            ng_set_bias(n, ng_frand(-bRange, bRange));
            for(auto& e : g.edges)
            {
                if(e.b == n.id)
                {
                    e.w = ng_frand(-wRange, wRange);
                }
            }
        }
    }
}

const char* ActName(Act a)
{
    switch(a){
        case Act::Linear:  return "Linear";
        case Act::ReLU:    return "ReLU";
        case Act::Sigmoid: return "Sigmoid";
        case Act::Tanh:    return "Tanh";
    }
    return "Linear";
}

Act NextAct(Act a)
{
    switch(a)
    {
        case Act::Linear:  return Act::ReLU;
        case Act::ReLU:    return Act::Sigmoid;
        case Act::Sigmoid: return Act::Tanh;
        case Act::Tanh:    return Act::Linear;
    }
    return Act::Linear;
}

// Build preset lessons locally so cancellation leaves the current dataset intact.
static void BuildFactorySequenceLessons(NeuralGraph& graph, std::vector<TrainSample>& lessons)
{
    if(IsTrainingActive()) { Logf("Stop training before changing the model or lessons."); return; }
    static const char* sensorNames[12] = {
        "Vibration", "Temperature", "Pressure", "MotorCurrent", "RPM", "Acoustic",
        "OilQuality", "Flow", "Torque", "Humidity", "Voltage", "Load"
    };
    static const char* stateNames[8] = {
        "Normal", "BearingWear", "Overheat", "PressureLeak",
        "RotorImbalance", "LubricationFault", "ElectricalFault", "EmergencyStop"
    };
    for(size_t i = 0; i < graph.layers.front().nodeIdx.size() && i < 12; ++i)
    {
        Neuron& neuron = graph.nodes[graph.layers.front().nodeIdx[i]];
        neuron.label = sensorNames[i];
        neuron.desc = "Normalized time-series sensor value (0..1)";
    }
    for(size_t i = 0; i < graph.layers.back().nodeIdx.size() && i < 8; ++i)
    {
        Neuron& neuron = graph.nodes[graph.layers.back().nodeIdx[i]];
        neuron.label = stateNames[i];
        neuron.desc = "Predicted machine condition after the observed sequence";
    }

    lessons.clear();
    auto lesson = [&](int target, const char* description, int variant)
    {
        TrainSample sample;
        sample.out.assign(8, 0.0f);
        sample.out[target] = 1.0f;
        sample.desc = description;
        for(int timeStep = 0; timeStep < 6; ++timeStep)
        {
            const float progress = timeStep / 5.0f;
            std::vector<float> frame(12, 0.12f + 0.015f * variant);
            frame[4] = 0.45f;
            frame[6] = 0.85f;
            frame[10] = 0.62f;
            frame[11] = 0.40f;
            switch(target)
            {
                case 0: frame[0] += 0.02f * progress; frame[1] += 0.03f * progress; break;
                case 1: frame[0] += 0.70f * progress; frame[5] += 0.52f * progress; break;
                case 2: frame[1] += 0.78f * progress; frame[3] += 0.35f * progress; break;
                case 3: frame[2] += 0.68f * progress; frame[7] -= 0.10f * timeStep; break;
                case 4: frame[0] += 0.62f * progress; frame[4] += 0.30f * progress; frame[8] += 0.40f * progress; break;
                case 5: frame[6] -= 0.13f * timeStep; frame[1] += 0.36f * progress; break;
                case 6: frame[3] += 0.62f * progress; frame[10] += (timeStep % 2 ? 0.24f : -0.18f); break;
                case 7: frame[0] += 0.75f * progress; frame[1] += 0.75f * progress; frame[2] += 0.65f * progress; break;
            }
            for(float& value : frame) value = std::clamp(value, 0.0f, 1.0f);
            sample.sequence.push_back(frame);
        }
        sample.in = sample.sequence.back();
        lessons.push_back(std::move(sample));
    };

    for(int variant = 0; variant < 4; ++variant)
    {
        lesson(0, "Stable operation over six observations", variant);
        lesson(1, "Vibration and acoustic energy rise: bearing wear", variant);
        lesson(2, "Temperature and current trend upward: overheating", variant);
        lesson(3, "Pressure rises while flow collapses: leak/blockage", variant);
        lesson(4, "Vibration, RPM and torque diverge: rotor imbalance", variant);
        lesson(5, "Oil quality degrades before heat rises", variant);
        lesson(6, "Current and voltage become unstable: electrical fault", variant);
        lesson(7, "Multiple critical trends require emergency stop", variant);
    }
}
static float TrainGetLR();
static float TrainGetL1();
static float TrainGetL2();
static int   TrainGetEpochs();
static float TrainGetTargetMSE();
static void  TrainApplyParams(float lr, float l1, float l2, int epochs, float targetMSE);

using Json = nlohmann::json;

static void RequireJson(bool condition, const std::string& message)
{
    if(!condition) throw std::runtime_error(message);
}

template<class T> static T JsonScalar(const Json& value, const std::string& name)
{
    if constexpr(std::is_same_v<T, float>)
    {
        RequireJson(value.is_number(), name + " must be a number");
        double number = value.get<double>();
        RequireJson(std::isfinite(number) && std::fabs(number) <= std::numeric_limits<float>::max(),
                    name + " is outside the finite float range");
        return (float)number;
    }
    else if constexpr(std::is_same_v<T, int>)
    {
        RequireJson(value.is_number_integer(), name + " must be an integer");
        double number = value.get<double>();
        RequireJson(number >= std::numeric_limits<int>::min() && number <= std::numeric_limits<int>::max(),
                    name + " is outside the integer range");
        return value.get<int>();
    }
    else
    {
        if constexpr(std::is_same_v<T, bool>) RequireJson(value.is_boolean(), name + " must be boolean");
        else RequireJson(value.is_string(), name + " must be text");
        return value.get<T>();
    }
}

template<class T> static T JsonField(const Json& object, const char* key, T fallback)
{
    RequireJson(object.is_object(), "Expected an object for field " + std::string(key));
    return object.contains(key) ? JsonScalar<T>(object.at(key), key) : fallback;
}

static std::vector<float> JsonFloats(const Json& array, const std::string& name)
{
    RequireJson(array.is_array(), name + " must be an array");
    std::vector<float> result;
    result.reserve(array.size());
    for(size_t i=0; i<array.size(); ++i) result.push_back(JsonScalar<float>(array[i], name + "[" + std::to_string(i) + "]"));
    return result;
}

static Vec3 JsonColor(const Json& object, const char* key, Vec3 fallback)
{
    RequireJson(object.is_object(), "Theme must be an object");
    if(!object.contains(key)) return fallback;
    auto values = JsonFloats(object.at(key), key);
    RequireJson(values.size() == 3, std::string(key) + " must contain three components");
    return {values[0], values[1], values[2]};
}

static Json ColorJson(Vec3 color) { return Json::array({color.x, color.y, color.z}); }

static void CheckJsonNumbers(const Json& value)
{
    if(value.is_number_float()) RequireJson(std::isfinite(value.get<double>()), "Cannot save NaN or infinity");
    if(value.is_structured()) for(const auto& child : value) CheckJsonNumbers(child);
}

static Json ReadJson(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if(!input) throw std::runtime_error("Cannot open " + path);
    // Read the whole file: parse() also rejects trailing non-whitespace data.
    std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    RequireJson(!input.bad(), "Read failed: " + path);
    Json document = Json::parse(contents);
    RequireJson(document.is_object(), "JSON root must be an object");
    return document;
}

// POSIX/Linux: a unique sibling file is flushed and closed before atomic replacement.
static void WriteJsonAtomic(const std::string& path, const Json& document)
{
    CheckJsonNumbers(document);
    const std::string contents = document.dump(2) + "\n";
    std::string pattern = path + ".tmp.XXXXXX";
    std::vector<char> temporary(pattern.begin(), pattern.end());
    temporary.push_back('\0');
    int fd = ::mkstemp(temporary.data());
    if(fd < 0) throw std::system_error(errno, std::generic_category(), "Cannot create temporary file for " + path);
    FILE* output = ::fdopen(fd, "wb");
    if(!output)
    {
        const int failure = errno;
        ::close(fd); ::unlink(temporary.data());
        throw std::system_error(failure, std::generic_category(), "Cannot open temporary stream");
    }
    int failure = 0;
    if(std::fwrite(contents.data(), 1, contents.size(), output) != contents.size()) failure = errno ? errno : EIO;
    if(!failure && std::fflush(output) != 0) failure = errno ? errno : EIO;
    if(!failure && ::fsync(fd) != 0) failure = errno ? errno : EIO;
    if(std::fclose(output) != 0 && !failure) failure = errno ? errno : EIO;
    if(!failure && ::rename(temporary.data(), path.c_str()) != 0) failure = errno;
    if(failure)
    {
        ::unlink(temporary.data());
        throw std::system_error(failure, std::generic_category(), "Cannot save " + path);
    }
}

static void ReportFileError(const std::string& action, const std::exception& error)
{
    gToastMsg = action + ": " + error.what();
    gToastT = glfwGetTime();
    Logf("%s", gToastMsg.c_str());
}

struct SavedTrainingParams
{
    float lr=TrainGetLR(), l1=TrainGetL1(), l2=TrainGetL2(), target=TrainGetTargetMSE();
    int epochs=TrainGetEpochs();
    void read(const Json& document)
    {
        lr=JsonField(document,"learning_rate",lr); l1=JsonField(document,"l1",l1);
        l2=JsonField(document,"l2",l2); target=JsonField(document,"target_mse",target);
        epochs=JsonField(document,"epochs",epochs);
        RequireJson(std::isfinite(lr) && std::isfinite(l1) && std::isfinite(l2) && std::isfinite(target) &&
                    lr>0 && l1>=0 && l2>=0 && epochs>0, "Invalid training parameters");
    }
    void write(Json& document) const
    {
        document["learning_rate"]=lr; document["l1"]=l1; document["l2"]=l2;
        document["target_mse"]=target; document["epochs"]=epochs;
    }
    void apply() const { TrainApplyParams(lr,l1,l2,epochs,target); }
};

// Decodes into local objects only; no editor state is changed by validation.
static void DecodeGraph(const Json& document, NeuralGraph& graph, std::vector<TrainSample>& lessons)
{
    const std::string version=JsonField(document,"version",std::string("neuro-graph-1"));
    RequireJson(version=="neuro-graph-1" || version=="neuro-graph-2", "Unsupported network version: " + version);
    const auto& nodes=document.at("nodes"); const auto& layers=document.at("layers");
    RequireJson(nodes.is_array() && layers.is_array() && !nodes.empty() && layers.size()>=2,
                "Network requires nodes and at least input/output layers");
    std::unordered_map<int,int> indices;
    for(const auto& entry:nodes)
    {
        Neuron n;
        n.id=JsonScalar<int>(entry.at("id"),"id");
        RequireJson(indices.emplace(n.id,(int)graph.nodes.size()).second,"Duplicate neuron ID: " + std::to_string(n.id));
        n.label=JsonField(entry,"label",n.label); n.desc=JsonField(entry,"desc",n.desc);
        n.isInput=JsonField(entry,"isInput",false);
        n.value=JsonField(entry,"value",n.value);
        n.speed=JsonField(entry,"speed",n.speed);
        n.bias=JsonField(entry,"bias",n.bias);
        n.amp=JsonField(entry,"amp",n.amp);
        n.phase=JsonField(entry,"phase",n.phase);
        n.x=JsonField(entry,"x",n.x);
        n.y=JsonField(entry,"y",n.y);
        n.live=JsonField(entry,"live",false);
        graph.nodes.push_back(std::move(n));
    }
    std::vector<int> membership(graph.nodes.size(),-1);
    size_t inputCount=0,outputCount=0;
    for(const auto& entry:layers)
    {
        Layer layer;
        const std::string kind=JsonScalar<std::string>(entry.at("kind"),"kind");
        if(kind=="Input") layer.kind=LayerKind::Input;
        else if(kind=="Output") layer.kind=LayerKind::Output;
        else if(kind=="Hidden") layer.kind=LayerKind::Hidden;
        else if(kind=="GRU") layer.kind=LayerKind::GRU;
        else throw std::runtime_error("Unknown layer kind: " + kind);
        layer.name=JsonField(entry,"name",layer.name);
        const std::string act=JsonField(entry,"activation",std::string("Linear"));
        if(act=="Linear") layer.act=Act::Linear;
        else if(act=="ReLU") layer.act=Act::ReLU;
        else if(act=="Sigmoid") layer.act=Act::Sigmoid;
        else if(act=="Tanh") layer.act=Act::Tanh;
        else throw std::runtime_error("Unknown activation: " + act);
        const auto& ids=entry.at("node_ids");
        RequireJson(ids.is_array(),"node_ids must be an array");
        for(const auto& value:ids)
        {
            int id=JsonScalar<int>(value,"node_ids"); auto found=indices.find(id);
            RequireJson(found!=indices.end(),"Unknown neuron ID: " + std::to_string(id));
            int index=found->second;
            RequireJson(membership[index]<0,"Neuron assigned more than once: " + std::to_string(id));
            membership[index]=(int)graph.layers.size();
            RequireJson(graph.nodes[index].isInput==(layer.kind==LayerKind::Input),"isInput disagrees with layer for neuron " + std::to_string(id));
            layer.nodeIdx.push_back(index);
        }
        if(layer.kind==LayerKind::Input) inputCount+=ids.size();
        if(layer.kind==LayerKind::Output) outputCount+=ids.size();
        if(layer.kind==LayerKind::GRU)
        {
            layer.gruInputSize=JsonScalar<int>(entry.at("gru_input_size"),"gru_input_size");
            layer.gruWz=JsonFloats(entry.at("gru_wz"),"gru_wz");
            layer.gruWr=JsonFloats(entry.at("gru_wr"),"gru_wr");
            layer.gruWh=JsonFloats(entry.at("gru_wh"),"gru_wh");
            layer.gruUz=JsonFloats(entry.at("gru_uz"),"gru_uz");
            layer.gruUr=JsonFloats(entry.at("gru_ur"),"gru_ur");
            layer.gruUh=JsonFloats(entry.at("gru_uh"),"gru_uh");
            layer.gruBz=JsonFloats(entry.at("gru_bz"),"gru_bz");
            layer.gruBr=JsonFloats(entry.at("gru_br"),"gru_br");
            layer.gruBh=JsonFloats(entry.at("gru_bh"),"gru_bh");
            layer.gruState.assign(layer.nodeIdx.size(),0.0f);
        }
        graph.layers.push_back(std::move(layer));
    }
    for(int li:membership) RequireJson(li>=0,"Neuron without a layer");
    RequireJson(graph.layers.front().kind==LayerKind::Input && graph.layers.back().kind==LayerKind::Output,
                "First layer must be Input and last layer Output");
    for(size_t li=1;li+1<graph.layers.size();++li)
        RequireJson(graph.layers[li].kind==LayerKind::Hidden || graph.layers[li].kind==LayerKind::GRU,
                    "Internal layers must be Hidden or GRU");
    std::string gruError;
    RequireJson(ValidateGRUParameters(graph,&gruError),gruError);
    if(document.contains("edges"))
    {
        RequireJson(document.at("edges").is_array(),"edges must be an array");
        for(const auto& entry:document.at("edges"))
        {
            Edge edge{JsonScalar<int>(entry.at("a"),"edge.a"),JsonScalar<int>(entry.at("b"),"edge.b"),
                      JsonScalar<float>(entry.at("w"),"edge.w")};
            auto a=indices.find(edge.a),b=indices.find(edge.b);
            RequireJson(a!=indices.end() && b!=indices.end(),"Edge refers to an unknown neuron");
            RequireJson(membership[a->second]<membership[b->second],"Edge must point to a later layer");
            graph.edges.push_back(edge);
        }
    }
    else // Legacy files without edges use full connections; explicit [] stays empty.
        for(size_t li=0;li+1<graph.layers.size();++li)
            for(int a:graph.layers[li].nodeIdx) for(int b:graph.layers[li+1].nodeIdx)
                graph.edges.push_back({graph.nodes[a].id,graph.nodes[b].id,0.5f});
    std::string topologyError;
    RequireJson(NG_ValidateTopology(graph,topologyError),topologyError);
    if(document.contains("lessons"))
    {
        RequireJson(document.at("lessons").is_array(),"lessons must be an array");
        for(const auto& entry:document.at("lessons"))
        {
            const std::string label="Lesson " + std::to_string(lessons.size()+1);
            TrainSample sample;
            sample.in=JsonFloats(entry.at("in"),label+" input");
            sample.out=JsonFloats(entry.at("out"),label+" output");
            sample.desc=JsonField(entry,"desc",sample.desc);
            RequireJson(sample.in.size()==inputCount && sample.out.size()==outputCount,label+" dimensions do not match the model");
            int steps=JsonField(entry,"sequence_steps",0);
            RequireJson(steps>=0,label+" has a negative step count");
            auto flat=entry.contains("sequence_flat") ? JsonFloats(entry.at("sequence_flat"),label+" sequence") : std::vector<float>{};
            RequireJson(steps==0 ? flat.empty() : inputCount>0 && flat.size()/inputCount==(size_t)steps && flat.size()%inputCount==0,
                        label+" sequence size does not match steps and input count");
            for(int ti=0;ti<steps;++ti)
                sample.sequence.emplace_back(flat.begin()+(size_t)ti*inputCount,flat.begin()+((size_t)ti+1)*inputCount);
            lessons.push_back(std::move(sample));
        }
    }
}

struct SavedDisplaySettings
{
    TurbineModel::Params turbine = gEngine ? gEngine->turbine.params : TurbineModel::Params{};
    float rpm = turbine.rpm;
    float startup_time = turbine.startupTime;
    float shutdown_time = turbine.shutdownTime;
    int blades = turbine.blades;
    float wind_strength = gWind.strength;
    float wind_speed = gWind.speed;
    float wind_jitter = gWind.jitter;
    bool strength_sine = gStrengthSineEnabled;
    bool speed_sine = gSpeedSineEnabled;
    bool jitter_sine = gJitterSineEnabled;
    float strength_phase = gWindStrengthPhase;
    float speed_phase = gWindSpeedPhase;
    float jitter_phase = gWindJitterPhase;
    bool live_wave_active = gLiveFeedWaveActive;
    float live_wave_phase = gLiveFeedWavePhase;
    float live_wave_speed = gLiveFeedWaveSpeed;
    int live_current_input = gLiveFeedCurrentInput;
    float live_input_delay = gLiveFeedInputDelay;
    bool random_live_feed = gRandomLiveFeedActive;
    bool all_inputs_live=false;
    int paired_theme=gPairedThemeIndex;
    UITheme theme=gEngine ? gEngine->renderer.theme : UITheme::Cyberpunk();
    GRUTheme gruTheme=gGRUTheme;
    void read(const Json& document)
    {
        paired_theme=JsonField(document,"paired_theme",-1);
        RequireJson(paired_theme>=-1 && paired_theme<16,"Invalid paired_theme");
        if(document.contains("f2_settings"))
        {
            const auto& section=document.at("f2_settings");
            rpm=JsonField(section,"rpm",rpm);
            startup_time=JsonField(section,"startup_time",startup_time);
            shutdown_time=JsonField(section,"shutdown_time",shutdown_time);
            blades=JsonField(section,"blades",blades);
            wind_strength=JsonField(section,"wind_strength",wind_strength);
            wind_speed=JsonField(section,"wind_speed",wind_speed);
            wind_jitter=JsonField(section,"wind_jitter",wind_jitter);
            strength_sine=JsonField(section,"strength_sine",strength_sine);
            speed_sine=JsonField(section,"speed_sine",speed_sine);
            jitter_sine=JsonField(section,"jitter_sine",jitter_sine);
            strength_phase=JsonField(section,"strength_phase",strength_phase);
            speed_phase=JsonField(section,"speed_phase",speed_phase);
            jitter_phase=JsonField(section,"jitter_phase",jitter_phase);
            live_wave_active=JsonField(section,"live_wave_active",live_wave_active);
            live_wave_phase=JsonField(section,"live_wave_phase",live_wave_phase);
            live_wave_speed=JsonField(section,"live_wave_speed",live_wave_speed);
            live_current_input=JsonField(section,"live_current_input",live_current_input);
            live_input_delay=JsonField(section,"live_input_delay",live_input_delay);
            random_live_feed=JsonField(section,"random_live_feed",random_live_feed);
            all_inputs_live=JsonField(section,"all_inputs_live",all_inputs_live);
        }
        RequireJson(rpm>=0 && rpm<=15000 && startup_time>=0.1f && startup_time<=60 &&
                    shutdown_time>=0.1f && shutdown_time<=60 && blades>=1 && blades<=10,"Invalid turbine settings");
        RequireJson(wind_strength>=0 && wind_strength<=300 && wind_speed>=0 && wind_speed<=5 &&
                    wind_jitter>=0 && wind_jitter<=0.5f && live_wave_speed>=0.5f && live_wave_speed<=5 &&
                    live_current_input>=0 && live_input_delay>=0.1f && live_input_delay<=1,"Invalid live input or wind settings");
        if(document.contains("theme"))
        {
            const auto& section=document.at("theme");
            theme.name=JsonField(section,"name",theme.name);
            theme.bg=JsonColor(section,"bg",theme.bg);
            theme.panel=JsonColor(section,"panel",theme.panel);
            theme.accent=JsonColor(section,"accent",theme.accent);
            theme.grid=JsonColor(section,"grid",theme.grid);
            theme.text=JsonColor(section,"text",theme.text);
            theme.mesh=JsonColor(section,"mesh",theme.mesh);
            theme.nInput=JsonColor(section,"nInput",theme.nInput);
            theme.nHidden=JsonColor(section,"nHidden",theme.nHidden);
            theme.nOutput=JsonColor(section,"nOutput",theme.nOutput);
            theme.nRing=JsonColor(section,"nRing",theme.nRing);
            theme.nHighlight=JsonColor(section,"nHighlight",theme.nHighlight);
            theme.edgeStart=JsonColor(section,"edgeStart",theme.edgeStart);
            theme.edgeEnd=JsonColor(section,"edgeEnd",theme.edgeEnd);
            theme.edgeAlpha=JsonField(section,"edgeAlpha",theme.edgeAlpha);
            theme.edgeBase=JsonField(section,"edgeBase",theme.edgeBase);
            theme.edgeScale=JsonField(section,"edgeScale",theme.edgeScale);
        }
        if(document.contains("gru_theme"))
        {
            const auto& section=document.at("gru_theme");
            gruTheme.name=JsonField(section,"name",gruTheme.name);
            gruTheme.columnBackground=JsonColor(section,"column_background",gruTheme.columnBackground);
            gruTheme.unitFill=JsonColor(section,"unit_fill",gruTheme.unitFill);
            gruTheme.unitBorder=JsonColor(section,"unit_border",gruTheme.unitBorder);
            gruTheme.connectionStart=JsonColor(section,"connection_start",gruTheme.connectionStart);
            gruTheme.connectionEnd=JsonColor(section,"connection_end",gruTheme.connectionEnd);
            gruTheme.particle=JsonColor(section,"particle",gruTheme.particle);
            gruTheme.recurrentLoop=JsonColor(section,"recurrent_loop",gruTheme.recurrentLoop);
        }
    }
    void write(Json& document) const
    {
        document["paired_theme"]=paired_theme;
        auto& section=document["f2_settings"];
        section["rpm"]=rpm;
        section["startup_time"]=startup_time;
        section["shutdown_time"]=shutdown_time;
        section["blades"]=blades;
        section["wind_strength"]=wind_strength;
        section["wind_speed"]=wind_speed;
        section["wind_jitter"]=wind_jitter;
        section["strength_sine"]=strength_sine;
        section["speed_sine"]=speed_sine;
        section["jitter_sine"]=jitter_sine;
        section["strength_phase"]=strength_phase;
        section["speed_phase"]=speed_phase;
        section["jitter_phase"]=jitter_phase;
        section["live_wave_active"]=live_wave_active;
        section["live_wave_phase"]=live_wave_phase;
        section["live_wave_speed"]=live_wave_speed;
        section["live_current_input"]=live_current_input;
        section["live_input_delay"]=live_input_delay;
        section["random_live_feed"]=random_live_feed;
        section["all_inputs_live"]=all_inputs_live;
        auto& t=document["theme"]; t["name"]=theme.name;
        t["bg"]=ColorJson(theme.bg);
        t["panel"]=ColorJson(theme.panel);
        t["accent"]=ColorJson(theme.accent);
        t["grid"]=ColorJson(theme.grid);
        t["text"]=ColorJson(theme.text);
        t["mesh"]=ColorJson(theme.mesh);
        t["nInput"]=ColorJson(theme.nInput);
        t["nHidden"]=ColorJson(theme.nHidden);
        t["nOutput"]=ColorJson(theme.nOutput);
        t["nRing"]=ColorJson(theme.nRing);
        t["nHighlight"]=ColorJson(theme.nHighlight);
        t["edgeStart"]=ColorJson(theme.edgeStart);
        t["edgeEnd"]=ColorJson(theme.edgeEnd);
        t["edgeAlpha"]=theme.edgeAlpha;
        t["edgeBase"]=theme.edgeBase;
        t["edgeScale"]=theme.edgeScale;
        auto& gt=document["gru_theme"]; gt["name"]=gruTheme.name;
        gt["column_background"]=ColorJson(gruTheme.columnBackground);
        gt["unit_fill"]=ColorJson(gruTheme.unitFill);
        gt["unit_border"]=ColorJson(gruTheme.unitBorder);
        gt["connection_start"]=ColorJson(gruTheme.connectionStart);
        gt["connection_end"]=ColorJson(gruTheme.connectionEnd);
        gt["particle"]=ColorJson(gruTheme.particle);
        gt["recurrent_loop"]=ColorJson(gruTheme.recurrentLoop);
    }
    void apply()
    {
        gWind.strength=wind_strength;
        gWind.speed=wind_speed;
        gWind.jitter=wind_jitter;
        gStrengthSineEnabled=strength_sine;
        gSpeedSineEnabled=speed_sine;
        gJitterSineEnabled=jitter_sine;
        gWindStrengthPhase=strength_phase;
        gWindSpeedPhase=speed_phase;
        gWindJitterPhase=jitter_phase;
        gLiveFeedWaveActive=live_wave_active;
        gLiveFeedWavePhase=live_wave_phase;
        gLiveFeedWaveSpeed=live_wave_speed;
        gLiveFeedCurrentInput=live_current_input;
        gLiveFeedInputDelay=live_input_delay;
        gRandomLiveFeedActive=random_live_feed;
        gPairedThemeIndex=paired_theme;
        gGRUTheme=std::move(gruTheme);
        gGRUThemeIndex=0;
        for(size_t i=0;i<GRUThemes().size();++i) if(GRUThemes()[i].name==gGRUTheme.name) { gGRUThemeIndex=i; break; }
        gLiveFeedInputTimer=0;
        if(gEngine)
        {
            gEngine->renderer.theme=std::move(theme);
            gEngine->turbine.params.rpm=rpm;
            gEngine->turbine.params.startupTime=startup_time;
            gEngine->turbine.params.shutdownTime=shutdown_time;
            gEngine->turbine.params.blades=blades;
        }
    }
};

static bool NG_SaveJson(const NeuralGraph& graph, const std::string& path)
{
    try
    {
        Json document={{"version","neuro-graph-2"}};
        SavedTrainingParams training; training.write(document);
        SavedDisplaySettings display; display.all_inputs_live=graph.liveInputs; display.write(document);
        document["nodes"]=Json::array(); document["layers"]=Json::array();
        document["edges"]=Json::array(); document["lessons"]=Json::array();
        for(const auto& n:graph.nodes)
            document["nodes"].push_back({{"id",n.id},{"label",n.label},{"desc",n.desc},{"isInput",n.isInput},
                {"value",n.value},{"speed",n.speed},{"bias",n.bias},{"amp",n.amp},{"live",n.live},
                {"phase",n.phase},{"x",n.x},{"y",n.y}});
        for(const auto& layer:graph.layers)
        {
            const char* kind=layer.kind==LayerKind::Input ? "Input" : layer.kind==LayerKind::Output ? "Output" :
                             layer.kind==LayerKind::GRU ? "GRU" : "Hidden";
            Json entry={{"kind",kind},{"name",layer.name},{"activation",ActName(layer.act)},{"node_ids",Json::array()}};
            for(int index:layer.nodeIdx)
            {
                RequireJson(index>=0 && (size_t)index<graph.nodes.size(),"Invalid neuron index during save");
                entry["node_ids"].push_back(graph.nodes[index].id);
            }
            if(layer.kind==LayerKind::GRU)
            {
                entry["gru_input_size"]=layer.gruInputSize;
                entry["gru_wz"]=layer.gruWz;
                entry["gru_wr"]=layer.gruWr;
                entry["gru_wh"]=layer.gruWh;
                entry["gru_uz"]=layer.gruUz;
                entry["gru_ur"]=layer.gruUr;
                entry["gru_uh"]=layer.gruUh;
                entry["gru_bz"]=layer.gruBz;
                entry["gru_br"]=layer.gruBr;
                entry["gru_bh"]=layer.gruBh;
            }
            document["layers"].push_back(std::move(entry));
        }
        for(const auto& edge:graph.edges) document["edges"].push_back({{"a",edge.a},{"b",edge.b},{"w",edge.w}});
        for(const auto& sample:gDataset)
        {
            std::vector<float> flat;
            for(const auto& frame:sample.sequence)
            {
                RequireJson(frame.size()==sample.in.size(),"Sequence frames have inconsistent dimensions");
                flat.insert(flat.end(),frame.begin(),frame.end());
            }
            document["lessons"].push_back({{"in",sample.in},{"out",sample.out},{"desc",sample.desc},
                {"sequence_steps",sample.sequence.size()},{"sequence_flat",flat}});
        }
        // Validate the same schema we accept on load before touching the existing file.
        NeuralGraph checked; std::vector<TrainSample> checkedLessons;
        DecodeGraph(document,checked,checkedLessons);
        training.read(document); display.read(document);
        WriteJsonAtomic(path,document);
        return true;
    }
    catch(const std::exception& error) { ReportFileError("Save failed ("+path+")",error); return false; }
}

static bool NG_LoadJson(NeuralGraph& destination, const std::string& path, std::string* message)
{
    if(IsTrainingActive() || gPendingTopology)
    { if(message) *message="Finish training or the pending topology change before loading a model."; return false; }
    try
    {
        const Json document=ReadJson(path);
        NeuralGraph graph;
        std::vector<TrainSample> lessons;
        SavedTrainingParams training; training.read(document);
        SavedDisplaySettings display; display.all_inputs_live=destination.liveInputs; display.read(document);
        DecodeGraph(document,graph,lessons);
        graph.liveInputs=display.all_inputs_live;
        graph.lastCanvasW=destination.lastCanvasW; graph.lastCanvasH=destination.lastCanvasH;
        graph.collapseHiddenLayers=destination.collapseHiddenLayers;
        graph.layoutLayers(graph.lastCanvasW>0 ? graph.lastCanvasW : 800.0f,
                           graph.lastCanvasH>0 ? graph.lastCanvasH : 500.0f);
        if(!graph.layers.front().nodeIdx.empty()) display.live_current_input%=(int)graph.layers.front().nodeIdx.size();
        else display.live_current_input=0;
        // Prepare potentially allocating geometry before committing any live settings.
        TurbineModel rebuilt;
        const bool rebuild=gEngine && display.blades!=gEngine->turbine.params.blades;
        if(rebuild)
        {
            rebuilt.params=gEngine->turbine.params;
            rebuilt.params.blades=display.blades;
            try { rebuilt.rebuild(); } catch(...) { rebuilt.destroy(); throw; }
        }
        destination=std::move(graph);
        gDataset=std::move(lessons); gLessonsCount=(int)gDataset.size();
        training.apply(); display.apply();
        if(rebuild)
        {
            std::swap(gEngine->turbine,rebuilt);
            // The prepared mesh uses the requested blade count; keep all loaded dynamics.
            gEngine->turbine.params.rpm=display.rpm;
            gEngine->turbine.params.startupTime=display.startup_time;
            gEngine->turbine.params.shutdownTime=display.shutdown_time;
            rebuilt.destroy();
        }
        ClearLessonTargets(); gSequenceDraft.clear(); gInferenceAccumulator=0; gInferenceNeedsReset=true;
        if(message) message->clear();
        return true;
    }
    catch(const std::exception& error)
    {
        if(message) *message=error.what();
        return false;
    }
}

// ===================== Training dataset & Engine config =====================

struct TrainState
{
    bool active=false;
    bool done=false;
    int totalEpochs=5000;
    int epoch=0;
    float lr=0.003f;
    float l1=0.0003f;
    float l2=0.0005f;
    float targetMSE=0.004f; // <=0 means ignore
    float lastMSE=0.0f;
    std::vector<float> mseHistory;
    double avgEpochMs=0.0;
    double startTime=0.0;
};
static TrainState gTrain;
static bool IsTrainingActive() { return gTrain.active; }
static std::string gTrainingError;

static void TopologyCommitted()
{
    gTrain.done=false; gTrain.epoch=0; gTrain.lastMSE=0;
    gTrain.mseHistory.clear(); gTrain.avgEpochMs=0;
    gCelebrationVisible=false; gTargetReachedVisual=false; gShowTrainingWindow=false;
}

// Accessors to avoid forward-dep issues
static float TrainGetLR(){ return gTrain.lr; }
static float TrainGetL1(){ return gTrain.l1; }
static float TrainGetL2(){ return gTrain.l2; }
static int   TrainGetEpochs(){ return gTrain.totalEpochs; }
static float TrainGetTargetMSE(){ return gTrain.targetMSE; }
static void  TrainApplyParams(float lr, float l1, float l2, int epochs, float targetMSE)
{
    if(gTrain.active) return;
    gTrain.lr = lr;
    gTrain.l1 = l1; 
    gTrain.l2 = l2; 
    gTrain.totalEpochs = epochs; 
    gTrain.targetMSE = targetMSE;
}

struct EngineConf
{
    std::string engine_version;
    float lr=0.003f, l1=0.0003f, l2=0.0005f;
    int epochs=5000;

    // Camera on Turbinoes 
    float cam_yaw=0.7f, cam_pitch=0.35f, cam_distance=4.0f;
    float cam_tx=0.0f, cam_ty=0.0f, cam_tz=0.0f;
};
static EngineConf gConf;

static double nowSeconds(){ return glfwGetTime(); }

// Forward declaration to access camera when saving/loading config
extern Engine* gEngine;
static bool SaveEngineConf(const char* path)
{
    try
    {
        Json document={{"engine_version",kAppVersion}};
        SavedTrainingParams training; training.write(document); training.read(document);
        const OrbitCamera camera=gEngine ? gEngine->cam : OrbitCamera{};
        document["cam_yaw"]=camera.yaw; document["cam_pitch"]=camera.pitch;
        document["cam_distance"]=camera.distance;
        document["cam_tx"]=camera.target.x; document["cam_ty"]=camera.target.y; document["cam_tz"]=camera.target.z;
        RequireJson(camera.distance>0,"Camera distance must be positive");
        WriteJsonAtomic(path,document);
        return true;
    }
    catch(const std::exception& error) { ReportFileError("Config save failed",error); return false; }
}

static void LoadEngineConf(const char* path)
{
    try
    {
        if(!std::filesystem::exists(path)) { SaveEngineConf(path); return; }
        const Json document=ReadJson(path);
        SavedTrainingParams training; training.read(document);
        EngineConf config=gConf;
        config.engine_version=JsonField(document,"engine_version",std::string(kAppVersion));
        config.cam_yaw=JsonField(document,"cam_yaw",config.cam_yaw);
        config.cam_pitch=JsonField(document,"cam_pitch",config.cam_pitch);
        config.cam_distance=JsonField(document,"cam_distance",config.cam_distance);
        config.cam_tx=JsonField(document,"cam_tx",config.cam_tx);
        config.cam_ty=JsonField(document,"cam_ty",config.cam_ty);
        config.cam_tz=JsonField(document,"cam_tz",config.cam_tz);
        RequireJson(config.cam_distance>0,"Camera distance must be positive");
        config.lr=training.lr; config.l1=training.l1; config.l2=training.l2; config.epochs=training.epochs;
        gConf=std::move(config); training.apply();
        if(gEngine)
        {
            gEngine->cam.yaw=gConf.cam_yaw; gEngine->cam.pitch=gConf.cam_pitch;
            gEngine->cam.distance=gConf.cam_distance;
            gEngine->cam.target={gConf.cam_tx,gConf.cam_ty,gConf.cam_tz};
        }
    }
    catch(const std::exception& error) { ReportFileError("Config load failed",error); }
}

// Deprecated: lessons are embedded in the saved network file; 

static void CollectIO(const NeuralGraph& g, std::vector<float>& in, std::vector<float>& out,
                      bool lessonTargets = false)
{
    in.clear(); out.clear();
    for(const auto& L : g.layers)
    {
        if(L.kind == LayerKind::Input)
        {
            for(int idx : L.nodeIdx) in.push_back(g.nodes[idx].value);
        } 
        else if(L.kind == LayerKind::Output)
        {
            for(int idx : L.nodeIdx)
            {
                const Neuron& node = g.nodes[idx];
                auto target = gLessonTargets.find(node.id);
                out.push_back(lessonTargets ? (target != gLessonTargets.end() ? target->second : 0.0f)
                                            : node.value);
            }
        }
    }
}

static float ActApply(Act a, float x)
{
    switch(a)
    {
        default: 
        case Act::Linear: return x;
        case Act::ReLU:   return x>0.f?x:0.f;
        case Act::Sigmoid: return 1.0f/(1.0f+std::exp(-x));
        case Act::Tanh:    return std::tanh(x);
    }
}

static float ActDeriv(Act a, float y)
{
    switch(a)
    {
        default: 
        case Act::Linear: return 1.0f;
        case Act::ReLU:   return y>0.f?1.f:0.f;
        case Act::Sigmoid: return y*(1.0f-y);
        case Act::Tanh:    return 1.0f - y*y;
    }
}

// ID->index cache (rebuilt when nodes/edges change)
struct GraphCache
{
    std::unordered_map<int,int> id2idx;
    std::unordered_map<int,std::vector<int>> outEdges; // id -> edge indices where id is src
    std::unordered_map<int,std::vector<int>> inEdges;  // id -> edge indices where id is dst
};

static GraphCache BuildCache(const NeuralGraph& g)
{
    GraphCache c;
    for(size_t i=0;i<g.nodes.size();++i) c.id2idx[g.nodes[i].id] = (int)i;
    for(size_t i=0;i<g.edges.size();++i)
    {
        const auto& e=g.edges[i];
        c.outEdges[e.a].push_back((int)i);
        c.inEdges[e.b].push_back((int)i);
    }
    return c;
}

static float Sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-std::clamp(value, -30.0f, 30.0f)));
}

static void ForwardGRULayer(NeuralGraph& graph, size_t layerIndex)
{
    Layer& layer = graph.layers[layerIndex];
    if(layerIndex == 0 || layer.nodeIdx.empty()) return;
    const Layer& previousLayer = graph.layers[layerIndex - 1];
    const int inputSize = (int)previousLayer.nodeIdx.size();
    const int hiddenSize = (int)layer.nodeIdx.size();

    std::vector<float> input(inputSize, 0.0f);
    for(int j = 0; j < inputSize; ++j) input[j] = graph.nodes[previousLayer.nodeIdx[j]].value;
    if(layer.gruState.size() != (size_t)hiddenSize) layer.gruState.assign(hiddenSize, 0.0f);

    layer.gruLastInput = input;
    layer.gruLastPrevious = layer.gruState;
    layer.gruLastUpdate.assign(hiddenSize, 0.0f);
    layer.gruLastReset.assign(hiddenSize, 0.0f);
    layer.gruLastCandidate.assign(hiddenSize, 0.0f);

    for(int i = 0; i < hiddenSize; ++i)
    {
        float updateSum = layer.gruBz[i];
        float resetSum = layer.gruBr[i];
        for(int j = 0; j < inputSize; ++j)
        {
            updateSum += layer.gruWz[(size_t)i * inputSize + j] * input[j];
            resetSum += layer.gruWr[(size_t)i * inputSize + j] * input[j];
        }
        for(int j = 0; j < hiddenSize; ++j)
        {
            updateSum += layer.gruUz[(size_t)i * hiddenSize + j] * layer.gruLastPrevious[j];
            resetSum += layer.gruUr[(size_t)i * hiddenSize + j] * layer.gruLastPrevious[j];
        }
        layer.gruLastUpdate[i] = Sigmoid(updateSum);
        layer.gruLastReset[i] = Sigmoid(resetSum);
    }

    std::vector<float> nextState(hiddenSize, 0.0f);
    for(int i = 0; i < hiddenSize; ++i)
    {
        float candidateSum = layer.gruBh[i];
        for(int j = 0; j < inputSize; ++j)
        {
            candidateSum += layer.gruWh[(size_t)i * inputSize + j] * input[j];
        }
        for(int j = 0; j < hiddenSize; ++j)
        {
            candidateSum += layer.gruUh[(size_t)i * hiddenSize + j]
                * layer.gruLastReset[j] * layer.gruLastPrevious[j];
        }
        const float candidate = std::tanh(candidateSum);
        layer.gruLastCandidate[i] = candidate;
        const float update = layer.gruLastUpdate[i];
        nextState[i] = (1.0f - update) * layer.gruLastPrevious[i] + update * candidate;
        graph.nodes[layer.nodeIdx[i]].value = nextState[i];
    }
    layer.gruState = std::move(nextState);
}

// A preview recomputes dense layers but holds recurrent memory at its current step.
static bool ForwardPass(NeuralGraph& g, bool advanceMemory = true)
{
    std::string error;
    if(!ValidateGRUParameters(g, &error))
    {
        // Avoid flooding the log on each UI update for the same invalid model.
        static std::string lastError;
        if(error != lastError) { Logf("Inference rejected: %s", error.c_str()); lastError = error; }
        return false;
    }
    GraphCache cache = BuildCache(g);
    for(size_t layerIndex = 0; layerIndex < g.layers.size(); ++layerIndex)
    {
        auto& L = g.layers[layerIndex];
        if(L.kind == LayerKind::Input) continue;
        if(L.kind == LayerKind::GRU)
        {
            if(advanceMemory) ForwardGRULayer(g, layerIndex);
            else
                for(size_t i = 0; i < L.nodeIdx.size(); ++i)
                    g.nodes[L.nodeIdx[i]].value = i < L.gruState.size() ? L.gruState[i] : 0.0f;
            continue;
        }
        for(int idx : L.nodeIdx)
        {
            Neuron& n = g.nodes[idx];
            float sum = ng_get_bias(n);
            auto it = cache.inEdges.find(n.id);
            if(it != cache.inEdges.end())
            {
                for(int ei : it->second)
                {
                    const auto& e = g.edges[ei];
                    auto source = cache.id2idx.find(e.a);
                    if(source != cache.id2idx.end()) sum += e.w * g.nodes[source->second].value;
                }
            }
            n.value = ActApply(L.act, sum);
        }
    }
    return true;
}

static void ResetInferenceMemory(NeuralGraph& graph)
{
    ResetGRUState(graph);
    gInferenceAccumulator = 0.0;
    gInferenceNeedsReset = false;
    ForwardPass(graph, false);
}

static void UpdateInference(NeuralGraph& graph, double dt)
{
    if(gPendingTopology) return;
    if(gTrain.active)
    {
        gInferenceAccumulator = 0.0;
        gInferenceNeedsReset = true;
        return;
    }
    if(gInferenceNeedsReset) ResetInferenceMemory(graph);
    bool hasLiveInput = graph.liveInputs || gLiveFeedWaveActive;
    for(const auto& node : graph.nodes)
        if(node.isInput && node.live) hasLiveInput = true;
    if(gPauseLiveInference || !hasLiveInput)
    {
        gInferenceAccumulator = 0.0;
        ForwardPass(graph, false);
        return;
    }
    // Catch up short frame delays, but do not replay a long UI stall as new observations.
    gInferenceAccumulator += std::clamp(dt, 0.0, 0.25);
    while(gInferenceAccumulator >= kInferenceStepSeconds)
    {
        gInferenceAccumulator -= kInferenceStepSeconds;
        graph.tickLiveInputs((float)kInferenceStepSeconds);
        if(!ForwardPass(graph))
        {
            gPauseLiveInference = true;
            gInferenceAccumulator = 0.0;
            break;
        }
    }
}

static bool ValidateTrainingData(const NeuralGraph& graph, std::string& error)
{
    auto fail = [&](const std::string& message) { error = message; return false; };
    if(gDataset.empty()) return fail("Dataset is empty.");
    if(!std::isfinite(gTrain.lr) || gTrain.lr <= 0) return fail("Learning rate must be finite and positive.");
    if(!std::isfinite(gTrain.l1) || gTrain.l1 < 0) return fail("L1 must be finite and non-negative.");
    if(!std::isfinite(gTrain.l2) || gTrain.l2 < 0) return fail("L2 must be finite and non-negative.");
    if(!std::isfinite(gTrain.targetMSE)) return fail("Target MSE must be finite.");
    if(gTrain.totalEpochs <= 0) return fail("Epoch count must be positive.");
    if(!NG_ValidateTopology(graph,error)) return false;
    const size_t inputs=graph.layers.front().nodeIdx.size();
    const size_t outputs=graph.layers.back().nodeIdx.size();
    for(size_t si = 0; si < gDataset.size(); ++si)
    {
        const auto& sample = gDataset[si];
        const std::string prefix = "Lesson " + std::to_string(si + 1) + ": ";
        auto check = [&](const std::vector<float>& values, size_t expected, const std::string& name)
        {
            if(values.size() != expected)
                return fail(prefix + name + " has " + std::to_string(values.size()) +
                            " values; expected " + std::to_string(expected));
            for(float value : values) if(!std::isfinite(value)) return fail(prefix + name + " contains NaN or infinity.");
            return true;
        };
        if(!check(sample.in, inputs, "input") || !check(sample.out, outputs, "target")) return false;
        for(size_t ti = 0; ti < sample.sequence.size(); ++ti)
            if(!check(sample.sequence[ti], inputs, "frame " + std::to_string(ti + 1))) return false;
    }
    return true;
}

static bool TrainingForward(NeuralGraph& graph, const std::vector<float>& frame, std::string& error)
{
    size_t pos = 0;
    for(const auto& layer : graph.layers) if(layer.kind == LayerKind::Input)
        for(int idx : layer.nodeIdx) graph.nodes[idx].value = frame[pos++];
    if(!ForwardPass(graph)) { error = "Invalid GRU parameters during forward pass."; return false; }
    for(const auto& node : graph.nodes)
        if(!std::isfinite(node.value)) { error = "Non-finite activation in neuron " + std::to_string(node.id); return false; }
    return true;
}

// Evaluate a fixed model; a copy keeps evaluation memory and inputs out of the editor.
static float EvaluateTrainingMSE(const NeuralGraph& graph, std::string& error)
{
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    if(!ValidateTrainingData(graph, error)) return invalid;
    NeuralGraph evaluation = graph;
    std::vector<int> outputs;
    for(const auto& layer : evaluation.layers) if(layer.kind == LayerKind::Output)
        outputs.insert(outputs.end(), layer.nodeIdx.begin(), layer.nodeIdx.end());
    double squaredError = 0.0;
    for(size_t si = 0; si < gDataset.size(); ++si)
    {
        ResetGRUState(evaluation);
        const auto& sample = gDataset[si];
        const std::vector<std::vector<float>> single = {sample.in};
        const auto& sequence = sample.sequence.empty() ? single : sample.sequence;
        for(const auto& frame : sequence)
            if(!TrainingForward(evaluation, frame, error))
            { error = "Evaluation lesson " + std::to_string(si + 1) + ": " + error; return invalid; }
        for(size_t oi = 0; oi < outputs.size(); ++oi)
        {
            const double difference = (double)evaluation.nodes[outputs[oi]].value - sample.out[oi];
            squaredError += difference * difference;
        }
    }
    const double mse = squaredError / ((double)gDataset.size() * outputs.size());
    if(!std::isfinite(mse) || mse > std::numeric_limits<float>::max())
    { error = "MSE is outside the finite numeric range."; return invalid; }
    return (float)mse;
}

static bool StartTraining(NeuralGraph& graph, bool toTarget)
{
    if(gPendingTopology) { Logf("Resolve the topology change first."); return false; }
    if(gTrain.active) { Logf("Training is already running."); return false; }
    gTrainingError.clear();
    if(!ValidateTrainingData(graph, gTrainingError) ||
       (toTarget && gTrain.targetMSE <= 0.0f))
    {
        if(gTrainingError.empty()) gTrainingError = "Train to target requires Target MSE > 0.";
        Logf("Cannot start training: %s", gTrainingError.c_str());
        gToastMsg = gTrainingError; gToastT = nowSeconds();
        return false;
    }
    bool uniform = true;
    const float reference = graph.edges.empty() ? 0.0f : graph.edges.front().w;
    for(const auto& edge : graph.edges) if(std::fabs(edge.w - reference) > 1e-6f) { uniform = false; break; }
    if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
    const float initialMSE = EvaluateTrainingMSE(graph, gTrainingError);
    if(!std::isfinite(initialMSE))
    {
        Logf("Cannot start training: %s", gTrainingError.c_str());
        gToastMsg = gTrainingError; gToastT = nowSeconds();
        return false;
    }
    if(toTarget) gTrain.totalEpochs = std::max(gTrain.totalEpochs, 200000);
    gTrain.active = true; gTrain.done = false; gTrain.epoch = 0;
    gTrain.lastMSE = initialMSE;
    gTrain.mseHistory = {initialMSE};
    gTrain.startTime = nowSeconds(); gTrain.avgEpochMs = 0.0;
    gCelebrationVisible = false; gTargetReachedVisual = false; gTrainInterrupted = false;
    gShowTrainingWindow = true;
    Logf("Training started: epochs=%d, lr=%.6f, L1=%.6f, L2=%.6f, targetMSE=%.6f, initialMSE=%.6f",
         gTrain.totalEpochs, gTrain.lr, gTrain.l1, gTrain.l2, gTrain.targetMSE, initialMSE);
    return true;
}

static float sgn(float x){ return (x>0)-(x<0); }

// Sequence training: cache every time step, propagate gradients through time,
// then apply one consistent, globally clipped update per training sample.
struct GRUStepCache {
    std::vector<float> x, prev, z, r, candidate;
};
struct SequenceStepCache {
    std::vector<float> values;
    std::vector<GRUStepCache> gru;
};
struct GRUGradient {
    std::vector<float> wz, wr, wh, uz, ur, uh, bz, br, bh;
};

static float TrainOneEpoch(NeuralGraph& g, float lr, float l1, float l2)
{
    if(!ValidateTrainingData(g, gTrainingError)) return std::numeric_limits<float>::quiet_NaN();
    const GraphCache cache = BuildCache(g);
    std::vector<int> nodeLayer(g.nodes.size(), -1);
    for(size_t li=0;li<g.layers.size();++li)
        for(int idx:g.layers[li].nodeIdx) nodeLayer[idx]=(int)li;
    std::vector<int> outputs;
    for(const auto& layer:g.layers) if(layer.kind==LayerKind::Output)
        outputs.insert(outputs.end(),layer.nodeIdx.begin(),layer.nodeIdx.end());
    for(const auto& sample:gDataset)
    {
        ResetGRUState(g);
        const std::vector<std::vector<float>> fallback={sample.in};
        const auto& sequence=sample.sequence.empty()?fallback:sample.sequence;
        std::vector<SequenceStepCache> steps;
        steps.reserve(sequence.size());
        for(const auto& frame:sequence)
        {
            if(!TrainingForward(g, frame, gTrainingError)) return std::numeric_limits<float>::quiet_NaN();
            SequenceStepCache step;
            step.values.reserve(g.nodes.size());
            for(const auto& n:g.nodes) step.values.push_back(n.value);
            step.gru.resize(g.layers.size());
            for(size_t li=0;li<g.layers.size();++li) if(g.layers[li].kind==LayerKind::GRU)
            {
                const auto& L=g.layers[li];
                step.gru[li]={L.gruLastInput,L.gruLastPrevious,L.gruLastUpdate,
                              L.gruLastReset,L.gruLastCandidate};
            }
            steps.push_back(std::move(step));
        }
        std::vector<float> edgeGrad(g.edges.size(),0.0f),biasGrad(g.nodes.size(),0.0f);
        std::vector<GRUGradient> gg(g.layers.size());
        for(size_t li=0;li<g.layers.size();++li) if(g.layers[li].kind==LayerKind::GRU)
        {
            const auto& L=g.layers[li]; auto& q=gg[li];
            q.wz.resize(L.gruWz.size());q.wr.resize(L.gruWr.size());q.wh.resize(L.gruWh.size());
            q.uz.resize(L.gruUz.size());q.ur.resize(L.gruUr.size());q.uh.resize(L.gruUh.size());
            q.bz.resize(L.gruBz.size());q.br.resize(L.gruBr.size());q.bh.resize(L.gruBh.size());
        }
        std::vector<std::vector<float>> adj(steps.size(),std::vector<float>(g.nodes.size(),0.0f));
        const size_t last=steps.size()-1;
        for(size_t oi=0;oi<outputs.size();++oi)
        {
            const int idx=outputs[oi];
            const float error=steps[last].values[idx]-sample.out[oi];
            adj[last][idx]+=error; // derivative of 0.5 * squared error
        }
        // Reverse both the time axis and the layer axis. No parameters change here.
        for(int ti=(int)steps.size()-1;ti>=0;--ti)
        {
            const auto& step=steps[ti];
            for(int li=(int)g.layers.size()-1;li>=0;--li)
            {
                const auto& L=g.layers[li];
                if(L.kind==LayerKind::Input) continue;
                if(L.kind==LayerKind::GRU)
                {
                    if(li==0) continue;
                    const auto& q=step.gru[li];
                    const int H=(int)L.nodeIdx.size(), I=(int)q.x.size();
                    if((int)q.prev.size()!=H || (int)q.z.size()!=H ||
                       (int)q.r.size()!=H || (int)q.candidate.size()!=H)
                    {
                        gTrainingError = "Invalid GRU cache during backpropagation.";
                        return std::numeric_limits<float>::quiet_NaN();
                    }
                    std::vector<float> dz(H),dr(H),dh(H),dprev(H),dx(I);
                    for(int i=0;i<H;++i)
                    {
                        const float d=adj[ti][L.nodeIdx[i]];
                        dh[i]=d*q.z[i]*(1-q.candidate[i]*q.candidate[i]);
                        dz[i]=d*(q.candidate[i]-q.prev[i])*q.z[i]*(1-q.z[i]);
                        dprev[i]=d*(1-q.z[i]);
                    }
                    for(int j=0;j<H;++j)
                    {
                        float resetGrad=0;
                        for(int i=0;i<H;++i)
                            resetGrad+=dh[i]*L.gruUh[(size_t)i*H+j]*q.prev[j];
                        dr[j]=resetGrad*q.r[j]*(1-q.r[j]);
                    }
                    for(int j=0;j<H;++j)
                        for(int i=0;i<H;++i)
                            dprev[j]+=L.gruUz[(size_t)i*H+j]*dz[i]
                                     +L.gruUr[(size_t)i*H+j]*dr[i]
                                     +L.gruUh[(size_t)i*H+j]*dh[i]*q.r[j];
                    auto& grad=gg[li];
                    for(int i=0;i<H;++i)
                    {
                        grad.bz[i]+=dz[i];grad.br[i]+=dr[i];grad.bh[i]+=dh[i];
                        for(int j=0;j<I;++j)
                        {
                            const size_t k=(size_t)i*I+j;
                            grad.wz[k]+=dz[i]*q.x[j];grad.wr[k]+=dr[i]*q.x[j];grad.wh[k]+=dh[i]*q.x[j];
                            dx[j]+=L.gruWz[k]*dz[i]+L.gruWr[k]*dr[i]+L.gruWh[k]*dh[i];
                        }
                        for(int j=0;j<H;++j)
                        {
                            const size_t k=(size_t)i*H+j;
                            grad.uz[k]+=dz[i]*q.prev[j];grad.ur[k]+=dr[i]*q.prev[j];
                            grad.uh[k]+=dh[i]*q.r[j]*q.prev[j];
                        }
                    }
                    const auto& previous=g.layers[li-1];
                    for(int j=0;j<I && j<(int)previous.nodeIdx.size();++j)
                        adj[ti][previous.nodeIdx[j]]+=dx[j];
                    if(ti>0) for(int j=0;j<H;++j) adj[ti-1][L.nodeIdx[j]]+=dprev[j];
                    continue;
                }
                for(int idx:L.nodeIdx)
                {
                    const auto& n=g.nodes[idx];
                    const float d=adj[ti][idx]*ActDeriv(L.act,step.values[idx]);
                    biasGrad[idx]+=d;
                    auto it=cache.inEdges.find(n.id);
                    if(it==cache.inEdges.end()) continue;
                    for(int ei:it->second)
                    {
                        const auto& e=g.edges[ei];
                        auto src=cache.id2idx.find(e.a);
                        if(src==cache.id2idx.end()) continue;
                        const int si=src->second;
                        // GRU inputs are represented by their gated matrices, not graph edges.
                        edgeGrad[ei]+=d*step.values[si];
                        adj[ti][si]+=d*e.w;
                    }
                }
            }
        }
        // Regularize weights (not biases) and clip the combined gradient norm.
        double norm2=0.0;
        auto regularize=[&](float grad,float weight){return grad+l2*weight+l1*sgn(weight);};
        std::vector<char> entersGRU(g.edges.size(),0);
        for(size_t ei=0;ei<g.edges.size();++ei)
        {
            auto it=cache.id2idx.find(g.edges[ei].b);
            if(it!=cache.id2idx.end() && nodeLayer[it->second]>=0 &&
               g.layers[nodeLayer[it->second]].kind==LayerKind::GRU) entersGRU[ei]=1;
            if(!entersGRU[ei]) {edgeGrad[ei]=regularize(edgeGrad[ei],g.edges[ei].w);norm2+=(double)edgeGrad[ei]*edgeGrad[ei];}
        }
        for(float v:biasGrad) norm2+=(double)v*v;
        auto regVec=[&](std::vector<float>& grad,const std::vector<float>& weight)
        {
            for(size_t k=0;k<grad.size();++k){grad[k]=regularize(grad[k],weight[k]);norm2+=(double)grad[k]*grad[k];}
        };
        for(size_t li=0;li<g.layers.size();++li) if(g.layers[li].kind==LayerKind::GRU)
        {
            auto& q=gg[li];const auto& L=g.layers[li];
            regVec(q.wz,L.gruWz);regVec(q.wr,L.gruWr);regVec(q.wh,L.gruWh);
            regVec(q.uz,L.gruUz);regVec(q.ur,L.gruUr);regVec(q.uh,L.gruUh);
            for(const auto* v:{&q.bz,&q.br,&q.bh}) for(float x:*v) norm2+=(double)x*x;
        }
        if(!std::isfinite(norm2))
        {
            gTrainingError = "Non-finite gradient; parameter update was rejected.";
            return std::numeric_limits<float>::quiet_NaN();
        }
        const float scale=(float)std::min(1.0,1.0/std::max(1e-12,std::sqrt(norm2)));
        for(size_t ei=0;ei<g.edges.size();++ei) if(!entersGRU[ei]) g.edges[ei].w-=lr*scale*edgeGrad[ei];
        for(size_t li=0;li<g.layers.size();++li)
        {
            auto& L=g.layers[li];
            if(L.kind==LayerKind::Input) continue;
            if(L.kind!=LayerKind::GRU)
            {
                for(int idx:L.nodeIdx) ng_set_bias(g.nodes[idx],ng_get_bias(g.nodes[idx])-lr*scale*biasGrad[idx]);
                continue;
            }
            auto& q=gg[li];
            auto apply=[&](std::vector<float>& w,const std::vector<float>& grad)
            {for(size_t k=0;k<w.size();++k)w[k]-=lr*scale*grad[k];};
            apply(L.gruWz,q.wz);apply(L.gruWr,q.wr);apply(L.gruWh,q.wh);
            apply(L.gruUz,q.uz);apply(L.gruUr,q.ur);apply(L.gruUh,q.uh);
            apply(L.gruBz,q.bz);apply(L.gruBr,q.br);apply(L.gruBh,q.bh);
        }
    }
    return EvaluateTrainingMSE(g, gTrainingError);
}

#endif

// ─────────────────────────────────────────────────────────────────────────────
// Neural Editor implementation
// ─────────────────────────────────────────────────────────────────────────────
#ifdef USE_IMGUI
[[maybe_unused]] static ImU32 colU32(float r,float g,float b,float a=1.f){ return ImGui::GetColorU32(ImVec4(r,g,b,a)); }

// Forward declaration of global engine pointer for theme access
extern Engine* gEngine;

Neuron* NeuralGraph::findAt(float mx, float my)
{
    for(auto& n: nodes)
    {
        float dx=mx-n.x, dy=my-n.y;
        if(dx*dx+dy*dy <= n.radius*n.radius) return &n;
    }
    return nullptr;
}

static std::string InputGroupName(const std::string& label)
{
    const size_t cut = label.find_first_of("_:/");
    std::string key = (cut == std::string::npos || cut == 0) ? "Other" : label.substr(0, cut);
    static const std::unordered_map<std::string, std::string> friendly = 
    {
        {"NPC", "NPC"}, {"Scene", "Scene"}, {"Act", "Actions"},
        {"Src", "Source"}, {"Tgt", "Target"}, {"Focus", "Focus"},
        {"Cand", "Candidate"}, {"In", "Inputs"}
    };
    auto it = friendly.find(key);
    return it == friendly.end() ? key : it->second;
}

static std::vector<std::pair<std::string, std::vector<int>>> InputGroups(const NeuralGraph& graph)
{
    std::vector<std::pair<std::string, std::vector<int>>> result;
    for(const Layer& layer : graph.layers)
    {
        if(layer.kind != LayerKind::Input) continue;
        for(int idx : layer.nodeIdx)
        {
            if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
            const std::string name = InputGroupName(graph.nodes[idx].label);
            auto it = std::find_if(result.begin(), result.end(), [&](const auto& group){ return group.first == name; });
            if(it == result.end()) result.push_back({name, {idx}});
            else it->second.push_back(idx);
        }
    }
    return result;
}

static bool IsVisibleInput(const NeuralGraph& graph, int nodeIndex)
{
    if(nodeIndex < 0 || nodeIndex >= (int)graph.nodes.size() || !graph.nodes[nodeIndex].isInput) return true;
    return graph.expandedInputGroup.empty() || InputGroupName(graph.nodes[nodeIndex].label) == graph.expandedInputGroup;
}

static int InputDisplaySlotCount(const NeuralGraph& graph)
{
    const auto groups = InputGroups(graph);
    int slots = 0;
    for(const auto& group : groups)
        slots += (group.first == graph.expandedInputGroup) ? (int)group.second.size() : 1;
    return std::max(1, slots);
}

static float CompactInputSlotY(int slot, int count, float y0, float y1, float radius)
{
    if(count<=1) return (y0+y1)*0.5f;
    // Keep the block as tight as possible while leaving a small visual gap
    // between neighbouring circles/plots. If necessary, shrink to available
    // height rather than pushing the first and last inputs against the edges.
    const float safeGap=std::max(28.0f,radius*2.0f+8.0f);
    const float blockSpan=std::min(y1-y0,safeGap*(float)(count-1));
    const float blockTop=(y0+y1-blockSpan)*0.5f;
    return blockTop+blockSpan*(float)slot/(float)(count-1);
}

static float CompactNeuronSlotY(int slot, int count, float y0, float y1, float radius, LayerKind kind)
{
    if(count<=1) return (y0+y1)*0.5f;
    // Hidden labels sit below their circles, while output labels sit beside them.
    // Reserve only the space each presentation actually needs.
    const float labelSpace=(kind==LayerKind::Hidden || kind==LayerKind::GRU)
        ? ImGui::GetTextLineHeight()+4.0f : 4.0f;
    const float safeGap=std::max(16.0f,radius*2.0f+labelSpace);
    const float blockSpan=std::min(y1-y0,safeGap*(float)(count-1));
    const float blockTop=(y0+y1-blockSpan)*0.5f;
    return blockTop+blockSpan*(float)slot/(float)(count-1);
}

void NeuralGraph::buildDemo()
{
    if(IsTrainingActive() || gPendingTopology) return;
    NeuralGraph candidate;
    NG_BuildTopology(candidate,3,{5,8,5},3);
    NG_RequestTopology(*this,std::move(candidate));
}

void NeuralGraph::layoutLayers(float W, float H)
{
    const float marginX = 40.f, marginY = 40.f;
    const float colGap  = (W - 2*marginX);
    const size_t expandedColumns = std::max<size_t>(1, layers.size());
    const float colW = colGap / (float)expandedColumns;
    const float layoutStartX = collapseHiddenLayers
        ? (W - colW * 3.0f) * 0.5f
        : marginX;

    // Compute a global radius so that all neurons fit vertically and horizontally
    float headerH = 3.0f * ImGui::GetTextLineHeightWithSpacing() + 6.0f; // 3 header lines
    float y0g = marginY + headerH;
    float y1g = H - marginY;
    float minSpacingY = 1e9f;
    for(const auto& Lchk : layers)
    {
        int N = Lchk.kind == LayerKind::Input ? InputDisplaySlotCount(*this) : (int)Lchk.nodeIdx.size();
        if(N>1)
        {
            float span = (y1g - y0g);
            float sp = span / (float)(N-1);
            if(sp < minSpacingY) minSpacingY = sp;
        }
    }

    if(minSpacingY == 1e9f) minSpacingY = (y1g - y0g); // single-node layers
    float rVert = std::max(6.0f, 0.5f*minSpacingY - 6.0f);

    // Horizontal limit: ensure circle fits within column; leave padding
    float rHorizGeneric = std::max(6.0f, 0.5f*colW - 12.0f);

    // Inputs have a mini plot on the left: ensure it stays within canvas
    const float kMiniPlotW = 48.0f;

    float rHorizInput = std::max(6.0f, 0.5f*colW - (10.0f + kMiniPlotW + 6.0f));
    float rHoriz = std::min(rHorizGeneric, rHorizInput);
    float globalR = std::clamp(std::min(rVert, rHoriz), 6.0f, 38.0f);

    // Inputs and outputs are the equally important endpoints of the graph.
    // Do not let dense hidden layers shrink either endpoint to tiny dots.
    const float endpointR=std::clamp(ImGui::GetFontSize()*0.75f,12.0f,18.0f);

    float x = layoutStartX;
    size_t displayColumn = 0;
    for(size_t li=0; li<layers.size(); ++li)
    {
        Layer& L = layers[li];
        if(collapseHiddenLayers && (L.kind == LayerKind::Hidden || L.kind == LayerKind::GRU))
        {
            L.x0 = layoutStartX + colW + 6.f;
            L.x1 = layoutStartX + colW*2.f - 6.f;
            L.y0 = marginY * 0.5f; L.y1 = H - marginY * 0.3f;
            continue;
        }

        std::vector<int> visible;
        for(int idx : L.nodeIdx)
            if(L.kind != LayerKind::Input || IsVisibleInput(*this, idx)) visible.push_back(idx);
        int N = L.kind == LayerKind::Input ? InputDisplaySlotCount(*this) : (int)visible.size();
        float y0 = y0g;
        float y1 = y1g;
        if(L.kind == LayerKind::Input)
        {
            for(int idx:L.nodeIdx) nodes[idx].radius=endpointR;
            int slot=0;
            for(const auto& group : InputGroups(*this))
            {
                if(group.first == expandedInputGroup)
                {
                    for(int idx : group.second)
                    {
                        float yy=CompactInputSlotY(slot,N,y0,y1,endpointR);
                        nodes[idx].x=x+colW*0.5f; nodes[idx].y=yy;
                        ++slot;
                    }
                } else ++slot; // one row is occupied by the group's pseudoneuron
            }
        } 
        else for(int i=0;i<(int)visible.size();i++)
        {
            int idx = visible[i];
            // Hidden/output layers form a compact block centred vertically in
            const float layerRadius=L.kind==LayerKind::Output?endpointR:globalR;
            float yy=CompactNeuronSlotY(i,N,y0,y1,layerRadius,L.kind);
            nodes[idx].x=x+colW*0.5f; nodes[idx].y=yy; nodes[idx].radius=layerRadius;
        }

        L.x0 = x + 6.f;
        L.x1 = x + colW - 6.f;
        L.y0 = marginY * 0.5f;
        L.y1 = H - marginY * 0.3f;
        ++displayColumn;
        x = layoutStartX + colW*(float)displayColumn;
        if(collapseHiddenLayers && L.kind == LayerKind::Input)
        {
            ++displayColumn;
            x = layoutStartX + colW*(float)displayColumn;
        }
    }
}

void NeuralGraph::tickLiveInputs(float dt)
{
    if(layers.empty()) return;
    
    // Update live feed wave effect
    if(gLiveFeedWaveActive)
    {
        gLiveFeedWavePhase += gLiveFeedWaveSpeed * dt;
        gLiveFeedInputTimer += dt;
        
        // Check if it's time to activate the next input
        if(gLiveFeedInputTimer >= gLiveFeedInputDelay)
        {
            gLiveFeedInputTimer = 0.0f;
            gLiveFeedCurrentInput++;
            
            // Reset to first input if we've gone through all
            if(gLiveFeedCurrentInput >= (int)layers.front().nodeIdx.size())
            {
                gLiveFeedCurrentInput = 0;
            }
        }
    }
    
    for(int i = 0; i < (int)layers.front().nodeIdx.size(); ++i)
    {
        int idx = layers.front().nodeIdx[i];
        Neuron& n = nodes[idx];
        if(!n.isInput) continue;
        
        // Handle wave effect
        if(gLiveFeedWaveActive)
        {
            if(i == gLiveFeedCurrentInput)
            {
                // Current input in wave - activate with sine wave
                n.value = 0.5f + 0.5f * std::sin(gLiveFeedWavePhase);
            } 
            else 
            {
                // Other inputs - gradually fade to 0
                n.value = std::max(0.0f, n.value - dt * 2.0f);
            }
        } else if(liveInputs || n.live)
        {
            // Normal live input behavior
            n.phase += 2.0f * 3.1415926535f * std::max(0.0f, n.speed) * dt;
            float a = std::clamp(n.amp, 0.0f, 1.0f);
            n.value = 0.5f + 0.5f * a * std::sin(n.phase);
        }
    }
}

void NeuralGraph::stepLayout(float dt, float W, float H)
{
    const float headerH=3.0f*ImGui::GetTextLineHeightWithSpacing()+6.0f;
    const float y0=40.0f+headerH;
    const float y1=H-40.0f;
    const float follow=1.0f-std::exp(-std::max(0.0f,dt)*12.0f);
    for(size_t idxA=0; idxA<nodes.size(); ++idxA)
    {
        auto& a = nodes[idxA];
        // Input rows are deliberately static.  With many inputs the force layout
        // used to push them apart immediately after loading, which also made
        // their labels overlap.  Their evenly-spaced positions are assigned by
        // layoutLayers(), so keep that column out of the simulation.
        if(a.isInput)
        {
            a.vx = 0.0f;
            a.vy = 0.0f;
            continue;
        }

        float targetX = a.x;
        float targetY = a.y;
        for(const auto& L: layers)
        {
            auto pos=std::find(L.nodeIdx.begin(),L.nodeIdx.end(),(int)idxA);
            if(pos!=L.nodeIdx.end())
            {
                targetX = (L.x0 + L.x1)*0.5f;
                const int count=(int)L.nodeIdx.size();
                const int order=(int)std::distance(L.nodeIdx.begin(),pos);
                targetY=CompactNeuronSlotY(order,count,y0,y1,a.radius,L.kind);
                break;
            }
        }
        a.x += (targetX-a.x)*follow;
        a.y += (targetY-a.y)*follow;
        a.vx=0.0f; a.vy=0.0f;
        a.x = std::clamp(a.x, 30.0f, W-30.0f);
        a.y = std::clamp(a.y, 30.0f, H-30.0f);
    }
}

void NeuralGraph::drawCanvas(bool& open, float width, float height)
{
    if(!open) return;

    extern bool gForceRelayoutWindows;
    ImGuiCond cond = gForceRelayoutWindows ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
    ImGui::SetNextWindowSize(ImVec2(width, height), cond);
    ImGui::SetNextWindowPos(ImVec2(10,10), cond);
    
    //F1 window transparency
    ImGui::SetNextWindowBgAlpha(0.50f);

    char editorTitle[128];
    std::snprintf(editorTitle, sizeof(editorTitle), "Neural Network (Neurons: %zu)###Neural Editor", nodes.size());
    if(ImGui::Begin(editorTitle, &open, ImGuiWindowFlags_NoCollapse))
    {
        // Block shortcuts while editing labels; reset per frame and set when inputs are focused
        gEditingLabel = false;
        // Split: top=canvas, optional bottom=live feed.  Keep the preference
        // local to this editor so opening a large network does not force a
        // tall feed panel back into view.
        static bool showLiveFeed = false;
        constexpr float preferredLiveFeedH = 220.0f;
        ImVec2 avail = ImGui::GetContentRegionAvail();
        if(avail.x < 100) avail.x = 100;
        if(avail.y < 140) avail.y = 140;

        const auto inputGroups = InputGroups(*this);
        if(!inputGroups.empty() && (expandedInputGroup.empty() ||
           std::none_of(inputGroups.begin(), inputGroups.end(), [&](const auto& g){ return g.first == expandedInputGroup; })))
        {
            expandedInputGroup = inputGroups.front().first;
            layoutLayers(lastCanvasW > 0 ? lastCanvasW : 800.0f, lastCanvasH > 0 ? lastCanvasH : 500.0f);
        }

        ImGui::BeginDisabled(gTrain.active || gPendingTopology.has_value());
        ImGui::Checkbox("Pause live inference", &gPauseLiveInference);
        ImGui::SameLine();
        if(ImGui::Button("Next sample"))
        {
            gPauseLiveInference = true;
            gInferenceAccumulator = 0.0;
            if(gInferenceNeedsReset) ResetInferenceMemory(*this);
            ForwardPass(*this);
        }
        ImGui::SameLine();
        if(ImGui::Button("Reset memory"))
        {
            gPauseLiveInference = true;
            ResetInferenceMemory(*this);
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Checkbox("Live Feed", &showLiveFeed);
        ImGui::SameLine();
        if(ImGui::Button(collapseHiddenLayers ? "Show hidden layers" : "Hide hidden layers"))
        {
            collapseHiddenLayers = !collapseHiddenLayers;
            layoutLayers(lastCanvasW > 0 ? lastCanvasW : 800.0f, lastCanvasH > 0 ? lastCanvasH : 500.0f);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if(ImGui::IsItemHovered())
            ImGui::SetTooltip("Live: 30 samples/s. Manual: set inputs, Apply, then Next sample.");
        avail = ImGui::GetContentRegionAvail();

        // The small control row is the only space retained when the feed is
        // collapsed, allowing the graph to fill all remaining editor space.
        const float controlsH = ImGui::GetFrameHeightWithSpacing();
        const float maxFeedH = std::max(80.0f, avail.y - 120.0f);
        const float feedH = showLiveFeed ? std::clamp(preferredLiveFeedH, 80.0f, maxFeedH) : 0.0f;
        float canvasH = avail.y - controlsH - feedH - (showLiveFeed ? 8.0f : 0.0f) - 28.0f;
        if(canvasH < 80.0f) canvasH = 80.0f;
        ImVec2 canvasSize = ImVec2(avail.x, canvasH);

        // Auto relayout when size changes
        if(std::abs(canvasSize.x - lastCanvasW) > 1.0f || std::abs(canvasSize.y - lastCanvasH) > 1.0f)
        {
            layoutLayers(canvasSize.x, canvasSize.y);
            lastCanvasW = canvasSize.x; lastCanvasH = canvasSize.y;
        }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 canvasMax = ImVec2(origin.x + canvasSize.x, origin.y + canvasSize.y);

        // CANVAS background
        dl->AddRectFilled(origin, canvasMax, ImGui::GetColorU32(ImVec4(0.08f,0.08f,0.095f,0.42f)), 6.0f);
        dl->AddRect(origin, canvasMax, ImGui::GetColorU32(ImVec4(0.25f,0.25f,0.3f,6.0f)), 6.0f);

        ImVec2 _csize = ImVec2(std::max(1.0f, canvasSize.x), std::max(1.0f, canvasSize.y));
        ImGui::InvisibleButton("canvas", _csize);
        bool isHovered = ImGui::IsItemHovered();
        ImVec2 mouse = ImGui::GetIO().MousePos;
        float mx = mouse.x - origin.x, my = mouse.y - origin.y;

        // layout tick + animation
        float dt = ImGui::GetIO().DeltaTime;
        stepLayout(std::max(0.0f, dt), canvasSize.x, canvasSize.y);

        // layer backgrounds + multi‑line headers + click activation
        for(size_t li=0; li<layers.size(); ++li)
        {
            Layer& L = layers[li];
            if(collapseHiddenLayers && (L.kind == LayerKind::Hidden || L.kind == LayerKind::GRU)) continue;
            ImVec2 a(origin.x + L.x0, origin.y + L.y0);
            ImVec2 b(origin.x + L.x1, origin.y + L.y1);
            const bool isGRULayer = L.kind == LayerKind::GRU;
            ImU32 colBg = isGRULayer
                ? ImGui::GetColorU32((int(li)==currentLayer
                    ? ImVec4(gGRUTheme.columnBackground.x*1.35f, gGRUTheme.columnBackground.y*1.35f,
                             gGRUTheme.columnBackground.z*1.35f, 0.86f)
                    : ImVec4(gGRUTheme.columnBackground.x, gGRUTheme.columnBackground.y,
                             gGRUTheme.columnBackground.z, 0.72f)))
                : ((int(li)==currentLayer
                    ? ImGui::GetColorU32(ImVec4(0.18f,0.20f,0.28f,0.84f))
                    : ImGui::GetColorU32(ImVec4(0.12f,0.12f,0.15f,0.66f))));
            dl->AddRectFilled(a, b, colBg, 8.0f);
            dl->AddRect(a, b, ImGui::GetColorU32(isGRULayer
                ? ImVec4(gGRUTheme.unitBorder.x,gGRUTheme.unitBorder.y,gGRUTheme.unitBorder.z,0.95f)
                : ImVec4(0.35f,0.35f,0.40f,0.9f)), 8.0f, 0, isGRULayer ? 3.0f : 2.0f);

            // --- 3 header lines:
            float lh = ImGui::GetTextLineHeightWithSpacing();
            ImVec2 hp(a.x + 8.0f, a.y + 6.0f);

            // 1) layer name
            const std::string shownLayerName = L.name;
            dl->AddText(hp, ImGui::GetColorU32(ImVec4(1,1,0.85f,1)), shownLayerName.c_str());

            // 2) neuron count
            const int shownCount = (int)L.nodeIdx.size();
            char ln[64]; std::snprintf(ln, sizeof(ln), "Neurons: %d", shownCount);
            dl->AddText(ImVec2(hp.x, hp.y + lh), ImGui::GetColorU32(ImVec4(0.85f,0.9f,1,1)), ln);

            // 3) activation - clickable
            const char* actName = ActName(L.act);
            char la[96];
            if(isGRULayer) std::snprintf(la, sizeof(la), "Gates: update / reset / candidate");
            else std::snprintf(la, sizeof(la), "Activation: %s  (click)", actName);
            ImVec2 actPos(hp.x, hp.y + lh*2.0f);
            dl->AddText(actPos, ImGui::GetColorU32(ImVec4(0.9f,0.8f,1,1)), la);

            ImVec2 actMin = actPos;
            ImVec2 actMax = ImVec2(b.x - 8.0f, actPos.y + lh);
            if(!gTrain.active && !gPendingTopology && !isGRULayer && ImGui::IsMouseHoveringRect(actMin, actMax) &&
               ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                L.act = NextAct(L.act);
            }

            // click entire column = select layer (LPM/PPM)
            ImGui::SetCursorScreenPos(a);
            ImGui::InvisibleButton(("layerbtn##"+std::to_string(li)).c_str(),
                                   ImVec2(std::max(1.0f, b.x-a.x), std::max(1.0f, b.y-a.y)));
            if(ImGui::IsItemClicked(ImGuiMouseButton_Left) || ImGui::IsItemClicked(ImGuiMouseButton_Right))
            {
                currentLayer = (int)li;
            }
        }

        // Collapsed input groups stay visible as named pseudoneurons.  They
        // occupy exactly one layout row and expand in-place when clicked.
        std::vector<std::pair<std::string, ImVec2>> collapsedInputHubs;
        bool inputGroupClickConsumed=false;
        std::string requestedInputGroup;
        if(!inputGroups.empty())
        {
            const Layer* inputLayer=nullptr;
            for(const Layer& layer:layers) if(layer.kind==LayerKind::Input){ inputLayer=&layer; break; }
            if(inputLayer)
            {
                const int slots=InputDisplaySlotCount(*this);
                const float headerH=3.0f*ImGui::GetTextLineHeightWithSpacing()+6.0f;
                const float y0=40.0f+headerH, y1=canvasSize.y-40.0f;
                int slot=0;
                for(const auto& group:inputGroups)
                {
                    if(group.first==expandedInputGroup){ slot+=(int)group.second.size(); continue; }
                    float inputRadius=14.0f;
                    if(!inputLayer->nodeIdx.empty()) inputRadius=nodes[inputLayer->nodeIdx.front()].radius;
                    const float yy=CompactInputSlotY(slot,slots,y0,y1,inputRadius);
                    ImVec2 p(origin.x+(inputLayer->x0+inputLayer->x1)*0.5f, origin.y+yy);
                    collapsedInputHubs.push_back({group.first,p});
                    const float radius=inputRadius;
                    dl->AddCircleFilled(p,radius,ImGui::GetColorU32(ImVec4(.12f,.42f,.38f,1.0f)),32);
                    dl->AddCircle(p,radius,ImGui::GetColorU32(ImVec4(.25f,.95f,.82f,1.0f)),32,2.5f);
                    dl->AddText(ImVec2(p.x-4.0f,p.y-ImGui::GetTextLineHeight()*0.5f),
                                ImGui::GetColorU32(ImVec4(.92f,1,.96f,1)),">");
                    std::string title=group.first+" ("+std::to_string(group.second.size())+")";
                    ImVec2 textSize=ImGui::CalcTextSize(title.c_str());
                    dl->AddText(ImVec2(p.x-radius-10.0f-textSize.x,p.y-textSize.y*0.5f),
                                ImGui::GetColorU32(ImVec4(1,1,.86f,1)),title.c_str());
                    const ImVec2 mousePos=ImGui::GetIO().MousePos;
                    const float hitRadius=radius+6.0f;
                    const float hitDx=mousePos.x-p.x, hitDy=mousePos.y-p.y;
                    const bool groupHovered=hitDx*hitDx+hitDy*hitDy <= hitRadius*hitRadius;
                    if(groupHovered)
                    {
                        dl->AddCircle(p,hitRadius,ImGui::GetColorU32(ImVec4(1,.92f,.30f,.95f)),32,2.0f);
                        ImGui::SetTooltip("Expand input group: %s",group.first.c_str());
                        if(ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                        {
                            requestedInputGroup=group.first;
                            inputGroupClickConsumed=true;
                        }
                    }
                    ++slot;
                }
                if(!requestedInputGroup.empty())
                {
                    expandedInputGroup=requestedInputGroup;
                    layoutLayers(canvasSize.x,canvasSize.y);
                }
            }
        }

        // A compact, readable stand-in for the complete hidden part of the
        // network. It keeps the data flow obvious without drawing hundreds of
        // overlapping neurons and labels.
        ImVec2 hiddenHub(origin.x + canvasSize.x*0.5f, origin.y + canvasSize.y*0.5f);
        if(collapseHiddenLayers)
        {
            int hiddenLayerCount=0, hiddenNeuronCount=0;
            for(const Layer& layer : layers) if(layer.kind == LayerKind::Hidden || layer.kind == LayerKind::GRU)
            {
                ++hiddenLayerCount; hiddenNeuronCount += (int)layer.nodeIdx.size();
            }

            const float pulse = 0.5f + 0.5f*std::sin((float)ImGui::GetTime()*2.0f);
            dl->AddCircleFilled(hiddenHub, 37.0f + pulse*3.0f, ImGui::GetColorU32(ImVec4(.18f,.75f,.72f,.12f)), 40);
            dl->AddCircleFilled(hiddenHub, 29.0f, ImGui::GetColorU32(ImVec4(.10f,.16f,.22f,1.0f)), 40);
            dl->AddCircle(hiddenHub, 29.0f, ImGui::GetColorU32(ImVec4(.25f,.95f,.85f,1.0f)), 40, 3.0f);
            dl->AddCircle(hiddenHub, 18.0f, ImGui::GetColorU32(ImVec4(.25f,.95f,.85f,.8f)), 6, 2.0f);
            const char* blackboxLabel="[BLACKBOX]";
            ImVec2 blackboxSize=ImGui::CalcTextSize(blackboxLabel);
            dl->AddText(ImVec2(hiddenHub.x-blackboxSize.x*.5f, hiddenHub.y-54.0f),
                        ImGui::GetColorU32(ImVec4(.92f,1.0f,.96f,1.0f)), blackboxLabel);
            char summary[96]; std::snprintf(summary, sizeof(summary), "%d deep/GRU layers / %d neurons", hiddenLayerCount, hiddenNeuronCount);
            ImVec2 ss=ImGui::CalcTextSize(summary);
            dl->AddText(ImVec2(hiddenHub.x-ss.x*.5f, hiddenHub.y+38.0f), ImGui::GetColorU32(ImVec4(.75f,.95f,.92f,1)), summary);
        }

        // compute top output neuron (for highlight)
        int topOutId = -1; float topOutVal = -FLT_MAX;
        for(const auto& L_ : layers)
        {
            if(L_.kind != LayerKind::Output) continue;
            for(int idx_ : L_.nodeIdx)
            {
                if(idx_>=0 && idx_<(int)nodes.size())
                {
                    const Neuron& nn_ = nodes[idx_];
                    if(nn_.value > topOutVal){ topOutVal = nn_.value; topOutId = nn_.id; }
                }
            }
        }

        // compute top input neuron (for label background)
        int topInId = -1; float topInVal = -FLT_MAX;
        for(const auto& L_ : layers)
        {
            if(L_.kind != LayerKind::Input) continue;
            for(int idx_ : L_.nodeIdx)
            {
                if(idx_>=0 && idx_<(int)nodes.size())
                {
                    const Neuron& nn_ = nodes[idx_];
                    if(nn_.value > topInVal){ topInVal = nn_.value; topInId = nn_.id; }
                }
            }
        }

        // edges
        auto findById = [&](int id)->Neuron*
        {
            for(auto& n: nodes){
                if(n.id==id) return &n;
            }
            return nullptr;
        };
        auto findLayerForNode = [&](int nodeIndex) -> const Layer*
        {
            for(const auto& layer : layers)
            {
                if(std::find(layer.nodeIdx.begin(), layer.nodeIdx.end(), nodeIndex) != layer.nodeIdx.end())
                {
                    return &layer;
                }
            }
            return nullptr;
        };

        const UITheme& _theme = (gEngine ? gEngine->renderer.theme : UITheme::Cyberpunk());
        for(auto& e: edges)
        {
            Neuron* A = findById(e.a);
            Neuron* B = findById(e.b);
            if(!A||!B) continue;
            const int ai = int(A - nodes.data()), bi = int(B - nodes.data());
            if(!IsVisibleInput(*this, ai) || !IsVisibleInput(*this, bi)) continue;
            if(collapseHiddenLayers) continue;
            const Layer* layerA = findLayerForNode(ai);
            const Layer* layerB = findLayerForNode(bi);
            if(layerA && layerB)
            {
                const size_t connectionCount = layerA->nodeIdx.size() * layerB->nodeIdx.size();
                const size_t visualStride = std::max<size_t>(1, connectionCount / 600);
                const size_t visualHash = (size_t)A->id * 131u + (size_t)B->id * 17u;
                if(visualHash % visualStride != 0) continue;
            }
            ImVec2 p1(origin.x+A->x, origin.y+A->y), p2(origin.x+B->x, origin.y+B->y);
            // (thickness computed below)
            // Map weight [-1,1] -> theme gradient and style
            {
                float t = (e.w + 1.0f) * 0.5f; t = t<0?0: (t>1?1:t);
                auto mixf = [](float a, float b, float tt){ return a + (b-a)*tt; };
                float r = mixf(_theme.edgeStart.x, _theme.edgeEnd.x, t);
                float g = mixf(_theme.edgeStart.y, _theme.edgeEnd.y, t);
                float b = mixf(_theme.edgeStart.z, _theme.edgeEnd.z, t);
                const bool gruFlow = (layerA && layerA->kind == LayerKind::GRU) ||
                                     (layerB && layerB->kind == LayerKind::GRU);
                if(gruFlow)
                {
                    r = mixf(gGRUTheme.connectionStart.x, gGRUTheme.connectionEnd.x, t);
                    g = mixf(gGRUTheme.connectionStart.y, gGRUTheme.connectionEnd.y, t);
                    b = mixf(gGRUTheme.connectionStart.z, gGRUTheme.connectionEnd.z, t);
                }
                float a = _theme.edgeAlpha;
                float thick = std::clamp(
                    _theme.edgeBase + _theme.edgeScale * std::fabs(e.w),
                    0.35f,
                    1.60f);

                // Pulsing glow
                float time = (float)ImGui::GetTime();
                float pulse = 0.5f + 0.5f*std::sin(time*2.0f + 0.15f*(float)(A->id + B->id));
                ImU32 glowCol = ImGui::GetColorU32(ImVec4(r,g,b, 0.18f * pulse));
                float w1 = std::min(thick * (2.5f + 2.0f*pulse), 3.00f);
                float w2 = std::min(thick * (1.8f + 1.2f*pulse), 2.40f);
                float w3 = std::min(thick * (1.2f + 0.6f*pulse), 1.90f);

                // Use actual panel values or oscillating values based on sine wave settings
                float currentStrength = gStrengthSineEnabled ? 
                    (150.0f + 150.0f * std::sin(gWindStrengthPhase)) : 
                    gWind.strength;
                float currentSpeed = gSpeedSineEnabled ? 
                    (2.5f + 2.5f * std::sin(gWindSpeedPhase)) : 
                    gWind.speed;
                float currentJitter = gJitterSineEnabled ? 
                    (0.25f + 0.25f * std::sin(gWindJitterPhase)) : 
                    gWind.jitter;

                // A connection inherits the mean wind properties of the two
                // layers it joins, so each layer pair sways independently.
                const float strengthScale = (layerA && layerB)
                    ? (layerA->windStrengthScale + layerB->windStrengthScale) * 0.5f
                    : 1.0f;
                const float speedScale = (layerA && layerB)
                    ? (layerA->windSpeedScale + layerB->windSpeedScale) * 0.5f
                    : 1.0f;
                const float jitterScale = (layerA && layerB)
                    ? (layerA->windJitterScale + layerB->windJitterScale) * 0.5f
                    : 1.0f;
                currentStrength *= strengthScale;
                currentSpeed *= speedScale;
                currentJitter *= jitterScale;

                // Wind sway: stronger turbine => more bend, same direction for all (with slight jitter)
                float rpm = 0.0f;
                if(gEngine)
                {
                    rpm = gEngine->turbine.params.rpm;
                }

                // Normalize RPM to [0,1] using a soft scale; tweakable mapping
                float windK = std::fmin(rpm / 2000.0f, 1.0f);

                // Base amplitude in pixels, smoothly interpolated
                float baseAmp = (currentStrength * windK) * gWindInterpolant;

                // Global wind direction (mostly to the right, slightly up). Add tiny rotation jitter.
                float baseDirX = 1.0f, baseDirY = -0.15f;
                float jitter = currentJitter * std::sin(time*0.8f + 0.07f*(A->id*13 + B->id*17));
                float ca = std::cos(jitter), sa = std::sin(jitter);
                ImVec2 windDir = ImVec2(baseDirX*ca - baseDirY*sa, baseDirX*sa + baseDirY*ca);

                // Mild oscillation so they sway a bit over time (same sense for all)
                float osc = 0.75f + 0.25f*std::sin(time * currentSpeed + 0.05f*(A->id + B->id));
                float amp = baseAmp * osc;
                ImVec2 wind = ImVec2(windDir.x * amp, windDir.y * amp);

                // Bézier control points pushed along wind to create a smooth bend
                ImVec2 d = ImVec2(p2.x - p1.x, p2.y - p1.y);
                ImVec2 c1 = ImVec2(p1.x + d.x*0.33f + wind.x, p1.y + d.y*0.33f + wind.y);
                ImVec2 c2 = ImVec2(p1.x + d.x*0.66f + wind.x, p1.y + d.y*0.66f + wind.y);

                dl->AddBezierCubic(p1, c1, c2, p2, glowCol, w1);
                dl->AddBezierCubic(p1, c1, c2, p2, glowCol, w2);
                dl->AddBezierCubic(p1, c1, c2, p2, glowCol, w3);
                dl->AddBezierCubic(p1, c1, c2, p2, ImGui::GetColorU32(ImVec4(r,g,b,a)), thick);
                if(gruFlow)
                {
                    const float flow = std::fmod(time * 0.42f + 0.013f * (A->id + B->id), 1.0f);
                    const float inv = 1.0f - flow;
                    ImVec2 particle(
                        inv*inv*inv*p1.x + 3*inv*inv*flow*c1.x + 3*inv*flow*flow*c2.x + flow*flow*flow*p2.x,
                        inv*inv*inv*p1.y + 3*inv*inv*flow*c1.y + 3*inv*flow*flow*c2.y + flow*flow*flow*p2.y);
                    dl->AddCircleFilled(particle, 2.8f, ImGui::GetColorU32(ImVec4(
                        gGRUTheme.particle.x,gGRUTheme.particle.y,gGRUTheme.particle.z,0.95f)), 10);
                }
            }
        }

        // A loop around every GRU column visualizes recurrent flow h(t-1) -> h(t).
        if(!collapseHiddenLayers)
        {
            for(const Layer& layer : layers)
            {
                if(layer.kind != LayerKind::GRU) continue;
                ImVec2 top(origin.x + layer.x0 + 10.0f, origin.y + layer.y0 + 82.0f);
                ImVec2 bottom(origin.x + layer.x0 + 10.0f, origin.y + layer.y1 - 12.0f);
                ImU32 recurrentColor = ImGui::GetColorU32(ImVec4(
                    gGRUTheme.recurrentLoop.x,gGRUTheme.recurrentLoop.y,gGRUTheme.recurrentLoop.z,0.82f));
                dl->AddBezierCubic(top, ImVec2(top.x-38.0f,top.y), ImVec2(bottom.x-38.0f,bottom.y), bottom,
                                   recurrentColor, 2.2f);
                dl->AddTriangleFilled(bottom, ImVec2(bottom.x-7.0f,bottom.y-10.0f),
                                      ImVec2(bottom.x+5.0f,bottom.y-9.0f), recurrentColor);
            }
        }

        if(collapseHiddenLayers)
        {
            const ImU32 flowGlow=ImGui::GetColorU32(ImVec4(.20f,.90f,.75f,.18f));
            const ImU32 flow=ImGui::GetColorU32(ImVec4(.25f,.95f,.82f,.72f));
            for(size_t i=0;i<nodes.size();++i) if(nodes[i].isInput && IsVisibleInput(*this, (int)i))
            {
                ImVec2 p(origin.x+nodes[i].x, origin.y+nodes[i].y);
                dl->AddBezierCubic(p, ImVec2(p.x+50,p.y), ImVec2(hiddenHub.x-65,hiddenHub.y), hiddenHub, flowGlow, 5.0f);
                dl->AddBezierCubic(p, ImVec2(p.x+50,p.y), ImVec2(hiddenHub.x-65,hiddenHub.y), hiddenHub, flow, 1.2f);
            }

            for(const Layer& layer: layers) if(layer.kind==LayerKind::Output) for(int idx:layer.nodeIdx)
            {
                ImVec2 p(origin.x+nodes[idx].x, origin.y+nodes[idx].y);
                dl->AddBezierCubic(hiddenHub, ImVec2(hiddenHub.x+65,hiddenHub.y), ImVec2(p.x-50,p.y), p, flowGlow, 5.0f);
                dl->AddBezierCubic(hiddenHub, ImVec2(hiddenHub.x+65,hiddenHub.y), ImVec2(p.x-50,p.y), p, flow, 1.2f);
            }

            for(const auto& groupHub:collapsedInputHubs)
            {
                const ImVec2 p=groupHub.second;
                dl->AddBezierCubic(p,ImVec2(p.x+50,p.y),ImVec2(hiddenHub.x-65,hiddenHub.y),hiddenHub,flowGlow,5.0f);
                dl->AddBezierCubic(p,ImVec2(p.x+50,p.y),ImVec2(hiddenHub.x-65,hiddenHub.y),hiddenHub,flow,1.2f);
            }
        } 
        else if(!collapsedInputHubs.empty())
        {
            // One aggregate connection per collapsed group communicates that
            // the group still participates in the network without restoring
            // every individual edge.
            const Layer* firstHidden=nullptr;
            for(const Layer& layer:layers) if(layer.kind==LayerKind::Hidden){ firstHidden=&layer; break; }
            if(firstHidden && !firstHidden->nodeIdx.empty())
            {
                ImVec2 target(0,0);
                for(int idx:firstHidden->nodeIdx){ target.x+=nodes[idx].x; target.y+=nodes[idx].y; }
                target.x=origin.x+target.x/(float)firstHidden->nodeIdx.size();
                target.y=origin.y+target.y/(float)firstHidden->nodeIdx.size();
                for(const auto& groupHub:collapsedInputHubs)
                {
                    ImVec2 p=groupHub.second;
                    dl->AddBezierCubic(p,ImVec2(p.x+45,p.y),ImVec2(target.x-45,target.y),target,
                        ImGui::GetColorU32(ImVec4(.25f,.95f,.82f,.65f)),1.5f);
                }
            }
        }

        // selection handled below; mini-plot is not clickable

        // neurons + labels + mini-sine + speed slider
        for(auto& n: nodes)
        {
            const int visibleIdx = int(&n - nodes.data());
            if(!IsVisibleInput(*this, visibleIdx)) continue;
            if(collapseHiddenLayers)
            {
                bool hiddenNode=false;
                for(const Layer& layer:layers) if((layer.kind==LayerKind::Hidden || layer.kind==LayerKind::GRU) &&
                    std::find(layer.nodeIdx.begin(), layer.nodeIdx.end(), visibleIdx)!=layer.nodeIdx.end()){ hiddenNode=true; break; }
                if(hiddenNode) continue;
            }
            ImVec2 p(origin.x+n.x, origin.y+n.y);
            bool isSel = (selected==n.id);

            // compute if this neuron is in the output layer
            bool isOutput = false;
            bool isGRU = false;
            {
                int idxN = int(&n - &nodes[0]);
                for(const auto& L: layers)
                {
                    if(L.kind != LayerKind::Output && L.kind != LayerKind::GRU) continue;
                    for(int j : L.nodeIdx)
                    {
                        if(j==idxN)
                        {
                            isOutput = L.kind == LayerKind::Output;
                            isGRU = L.kind == LayerKind::GRU;
                            break;
                        }
                    }
                    if(isOutput || isGRU) break;
                }
            }

            auto colU = [&](const Vec3& v){ return ImGui::GetColorU32(ImVec4(v.x,v.y,v.z,1.0f)); };
            ImU32 baseFill = isGRU ? ImGui::GetColorU32(ImVec4(
                    gGRUTheme.unitFill.x,gGRUTheme.unitFill.y,gGRUTheme.unitFill.z,1.0f))
                : (n.isInput ? colU(_theme.nInput) : (isOutput ? colU(_theme.nOutput) : colU(_theme.nHidden)));
            ImU32 fill = isSel ? ImGui::GetColorU32(ImVec4(0.2f,0.5f,1.0f,1.0f)) : baseFill;
            ImU32 ring = isSel ? colU(_theme.nRing)
                               : ImGui::GetColorU32(isGRU ? ImVec4(gGRUTheme.unitBorder.x,
                                                         gGRUTheme.unitBorder.y,gGRUTheme.unitBorder.z,1.0f)
                                                         : ImVec4(0.9f,0.9f,0.95f,0.9f));

            // highlight top output: theme highlight
            if(isOutput && n.id == topOutId)
            {
                fill = colU(_theme.nHighlight);
            }

            if(isGRU)
            {
                const ImVec2 diamond[4] = {
                    ImVec2(p.x, p.y-n.radius-2.0f), ImVec2(p.x+n.radius+2.0f, p.y),
                    ImVec2(p.x, p.y+n.radius+2.0f), ImVec2(p.x-n.radius-2.0f, p.y)
                };
                dl->AddConvexPolyFilled(diamond, 4, fill);
                dl->AddPolyline(diamond, 4, ring, ImDrawFlags_Closed, 2.5f);
                dl->AddCircle(p, n.radius*0.43f, ImGui::GetColorU32(ImVec4(
                    gGRUTheme.particle.x,gGRUTheme.particle.y,gGRUTheme.particle.z,0.9f)), 16, 1.4f);
            }
            else
            {
                dl->AddCircleFilled(p, n.radius, fill, 32);
                dl->AddCircle(p, n.radius, ring, 32, 2.0f);
            }

            // Determine if this neuron is in a hidden layer
            bool isHidden = false;
            {
                int idxN = int(&n - &nodes[0]);
                for(const auto& L: layers)
                {
                    if(L.kind != LayerKind::Hidden && L.kind != LayerKind::GRU) continue;
                    for(int j : L.nodeIdx){ if(j==idxN){ isHidden = true; break; } }
                    if(isHidden) break;
                }
            }

            // label: hidden BELOW, input on LEFT (leave space for mini-plot), output on RIGHT
            ImVec2 ts = ImGui::CalcTextSize(n.label.c_str());
            ImVec2 tl;
            if(isHidden)
            {
                tl = ImVec2(p.x - ts.x*0.5f, p.y + n.radius + 4.0f);
            } 
            else if(isOutput)
            {
                tl = ImVec2(p.x + n.radius + 8.0f, p.y - ts.y*0.5f);
            } 
            else 
            {
                tl = ImVec2(p.x - n.radius - 8.0f - ts.x - 56.0f, p.y - ts.y*0.5f); // input - space for mini-plot on the left
            }

            // Yellow label background for top input/output by value
            bool isTopIn  = (!isHidden && n.isInput && n.id==topInId);
            bool isTopOut = (isOutput && n.id==topOutId);
            if(isTopIn || isTopOut)
            {
                ImVec2 pad(4.0f, 2.0f);
                ImVec2 r0 = ImVec2(tl.x - pad.x, tl.y - pad.y);
                ImVec2 r1 = ImVec2(tl.x + ts.x + pad.x, tl.y + ts.y + pad.y);
                ImU32 bg = ImGui::GetColorU32(ImVec4(1.0f,0.95f,0.30f,0.90f));
                ImU32 bd = ImGui::GetColorU32(ImVec4(0.85f,0.75f,0.10f,0.95f));
                dl->AddRectFilled(r0, r1, bg, 3.0f);
                dl->AddRect(r0, r1, bd, 3.0f, 0, 1.0f);
            }

            {
                ImU32 labelCol = ImGui::GetColorU32( (isTopIn || isTopOut)
                    ? ImVec4(0.0f,0.0f,0.0f,1.0f)
                    : ImVec4(1.0f,1.0f,0.85f,1.0f) );
                dl->AddText(ImGui::GetFont(),ImGui::GetFontSize(),tl,labelCol,n.label.c_str());
                // Tooltip over label area
                ImVec2 br = ImVec2(tl.x + ts.x, tl.y + ts.y);
                if(ImGui::IsMouseHoveringRect(tl, br))
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(n.desc.empty()? "default description" : n.desc.c_str());
                    ImGui::EndTooltip();
                }
            }

            // Mark a separately configured lesson target, never a prediction override.
            if(isOutput)
            {
                auto target = gLessonTargets.find(n.id);
                if(target != gLessonTargets.end())
                {
                    ImVec2 iconC = ImVec2(tl.x - 10.0f, p.y);
                    ImU32 icoCol = ImGui::GetColorU32(ImVec4(1.0f,0.95f,0.3f,1.0f)); // yellow icon
                    dl->AddCircleFilled(iconC, 4.0f, icoCol, 16);

                    // Tooltip area
                    ImGui::SetCursorScreenPos(ImVec2(iconC.x - 6.0f, iconC.y - 6.0f));
                    ImGui::InvisibleButton((std::string("ovr##")+std::to_string(n.id)).c_str(), ImVec2(12,12));
                    if(ImGui::IsItemHovered()) ImGui::SetTooltip("Lesson target: %.4f (prediction is unchanged)", target->second);
                }
            }

            // show value inside output neurons too
            if(isOutput)
            {
                char bufv[16]; std::snprintf(bufv,sizeof(bufv),"%.2f", n.value);
                ImVec2 tsv2 = ImGui::CalcTextSize(bufv);
                dl->AddText(ImVec2(p.x - tsv2.x*0.5f, p.y - tsv2.y*0.5f),
                            ImGui::GetColorU32(ImVec4(0.05f,0.05f,0.08f,1.0f)), bufv);
            }

            if(n.isInput)
            {
                char buf[16]; std::snprintf(buf,sizeof(buf),"%.2f", n.value);
                ImVec2 tsv = ImGui::CalcTextSize(buf);
                dl->AddText(ImVec2(p.x - tsv.x*0.5f, p.y - tsv.y*0.5f),
                            ImGui::GetColorU32(ImVec4(0.05f,0.05f,0.08f,1.0f)), buf);

                // Mini live-feed plot moved to LEFT of neuron
                const float PW=48.f, PH=22.f;
                ImVec2 pp0(p.x - n.radius - 10.f - PW, p.y - PH*0.5f);
                ImVec2 pp1(pp0.x + PW, pp0.y + PH);
                ImU32 plotBg = ImGui::GetColorU32(ImVec4(0.1f,0.12f,0.14f,1.0f));
                ImU32 plotFg = ImGui::GetColorU32(ImVec4(0.2f,0.85f,0.7f,1.0f));
                if(n.live)
                { 
                    plotBg = ImGui::GetColorU32(ImVec4(0.12f,0.16f,0.12f,1.0f)); 
                    plotFg = ImGui::GetColorU32(ImVec4(0.3f,0.9f,0.3f,1.0f)); 
                }

                dl->AddRectFilled(pp0, pp1, plotBg, 3.0f);
                dl->AddRect(pp0, pp1, ImGui::GetColorU32(ImVec4(0.35f,0.35f,0.40f,1.0f)), 3.0f, 0, 1.0f);
                const int S=32;
                ImVec2 prev = {pp0.x, pp0.y + PH*(1.0f - (0.5f+0.5f*std::sin(n.phase - 2*3.14159f)))};
                for(int i=1;i<=S;i++)
                {
                    float t=(float)i/(float)S;
                    float val = 0.5f + 0.5f*std::sin(n.phase + t*2.0f*3.14159265f);
                    ImVec2 cur = { pp0.x + t*PW, pp0.y + PH*(1.0f - val) };
                    dl->AddLine(prev, cur, plotFg, 1.5f);
                    prev = cur;
                }
                // mini plot is not clickable
            }
        }

        // click elsewhere on canvas selects neuron but doesn't toggle live/value
        if(isHovered && !inputGroupClickConsumed && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
            if(Neuron* hit = findAt(mx,my))
            {
                selected = hit->id;
                std::snprintf(editBuf, sizeof(editBuf), "%s", hit->label.c_str());
            } else 
            {
                selected = -1;
                std::memset(editBuf,0,sizeof(editBuf));
            }
        }

        // ====== LIVE FEED (under the graph) ======
        if(showLiveFeed)
        {
            ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + canvasSize.y + 8.0f));
            ImGui::BeginChild("live_feed", ImVec2(avail.x, feedH), true, ImGuiWindowFlags_HorizontalScrollbar);
            {
            const Layer& L = layers[std::clamp(currentLayer, 0, (int)layers.size()-1)];
            ImGui::Text("Live Feed - %s (neurons: %d, activation: %s)",
                        L.name.c_str(), (int)L.nodeIdx.size(), ActName(L.act));
            if(ImGui::BeginTable("lf", 6, ImGuiTableFlags_Resizable|ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersInnerV)){
                ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 40.f);
                ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.f);
                ImGui::TableSetupColumn("Speed(Hz)", ImGuiTableColumnFlags_WidthFixed, 90.f);
                ImGui::TableSetupColumn("In", ImGuiTableColumnFlags_WidthFixed, 40.f);
                ImGui::TableSetupColumn("Out", ImGuiTableColumnFlags_WidthFixed, 45.f);
                ImGui::TableHeadersRow();

                for(int idx : L.nodeIdx)
                {
                    Neuron& n = nodes[idx];
                    int inCnt=0, outCnt=0;
                    for(auto& e: edges){ if(e.b==n.id) ++inCnt; if(e.a==n.id) ++outCnt; }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("%d", n.id);
                    ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(n.label.c_str());

                    ImGui::TableSetColumnIndex(2);
                    float v = n.value;
                    ImVec4 c = ImVec4(1.0f - v, 0.2f + 0.8f*v, 0.15f, 1.0f);
                    ImGui::TextColored(c, "%.3f", v);

                    ImGui::TableSetColumnIndex(3);
                    if(n.isInput)
                    {
                        ImGui::Text("%.2f", n.speed);
                    } 
                    else 
                    {
                        ImGui::TextDisabled("-");
                    }

                    ImGui::TableSetColumnIndex(4); ImGui::Text("%d", inCnt);
                    ImGui::TableSetColumnIndex(5); ImGui::Text("%d", outCnt);
                }
                ImGui::EndTable();
            }

            // Selected neuron editor
            if(selected != -1)
            {
                Neuron* sel = nullptr;
                for(auto& n : nodes) if(n.id==selected){ sel=&n; break; }
                if(sel)
                {
                    ImGui::Separator();
                    ImGui::Text(u8"Neuron editor (ID %d)", sel->id);
                    ImGui::BeginDisabled(gTrain.active || gPendingTopology.has_value());
                    ImGui::PushItemWidth(320.f);

                    ImGui::InputText(u8"Label", editBuf, IM_ARRAYSIZE(editBuf));
                    bool labelEnter = false; // Enter handled globally below
                    gEditingLabel = ImGui::IsItemActive() || ImGui::IsItemFocused();

                    // Is output?
                    bool isOutput = false;
                    for(const auto& Lchk : layers)
                    {
                        if(Lchk.kind == LayerKind::Output)
                        {
                            for(int idxN : Lchk.nodeIdx)
                            {
                                if(nodes[idxN].id == sel->id){ isOutput = true; break; }
                            }
                        }
                        if(isOutput) break;
                    }

                    static int lastSelForVal = -1;
                    static float pendingVal = 0.0f;
                    static float pendingSpeed = 0.0f;
                    static unsigned int lastTargetRevision = 0;
                    if(lastSelForVal != sel->id || lastTargetRevision != gLessonTargetRevision){
                        lastTargetRevision = gLessonTargetRevision;
                        const auto target = gLessonTargets.find(sel->id);
                        pendingVal = isOutput ? (target != gLessonTargets.end() ? target->second : 0.0f) : sel->value;
                        pendingSpeed = sel->speed;
                        std::snprintf(editBuf, sizeof(editBuf), "%s", sel->label.c_str());
                        lastSelForVal = sel->id;
                    }

                    // Description: reset input when switching selected neuron
                    static char descBuf[256];
                    static int lastSelForDesc = -1;
                    if(lastSelForDesc != sel->id)
                    {
                        // Clear the description field on neuron change (per request)
                        descBuf[0] = '\0';
                        lastSelForDesc = sel->id;
                    }

                    ImGui::InputTextMultiline("Description", descBuf, IM_ARRAYSIZE(descBuf), ImVec2(320, 60));
                    gEditingLabel = gEditingLabel || ImGui::IsItemActive() || ImGui::IsItemFocused();

                    if(isOutput)
                    {
                        ImGui::Text("Prediction: %.4f", sel->value);
                        ImGui::InputFloat("Lesson target", &pendingVal, 0.0f, 0.0f, "%.4f");
                    }
                    else if(sel->isInput)
                    {
                        ImGui::SliderFloat(u8"Value", &pendingVal, 0.0f, 1.0f, "%.4f");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(120.f);
                        ImGui::InputFloat("##val", &pendingVal, 0.0f, 0.0f, "%.4f");
                    }
                    else
                    {
                        ImGui::BeginDisabled();
                        float vtmp = sel->value;
                        ImGui::SliderFloat(u8"Value", &vtmp, -10.0f, 10.0f, "%.4f");
                        ImGui::EndDisabled();
                    }

                    if(sel->isInput)
                    {
                        if(ImGui::SliderFloat(u8"Speed (Hz)", &pendingSpeed, 0.0f, 1.0f, "%.3f"))
                        {
                            // Quantize to 0.001 increments
                            pendingSpeed = std::round(pendingSpeed * 1000.0f) / 1000.0f;
                            float prev = sel->speed;
                            sel->speed = std::max(0.0f, pendingSpeed);

                            // If user sets speed > 0, enable per-neuron live and reset phase for visible motion
                            if(sel->speed > 0.0f) 
                            {
                                if(prev <= 0.0f) sel->phase = 0.0f;
                                sel->live = true;
                            } else 
                            {
                                // speed == 0 stops motion; keep value as-is; live flag can remain
                            }
                        }

                        ImGui::Checkbox(u8"Live feed", &sel->live);
                    } 
                    else 
                    {
                        ImGui::BeginDisabled(); float dummy=0.f;
                        ImGui::SliderFloat(u8"Speed (Hz)", &dummy, 0.0f, 0.0f, "-");
                        ImGui::EndDisabled();
                    }

                    bool doApply = false;
                    doApply |= ImGui::Button(u8"Apply");

                    // Enter anywhere in this window applies changes
                    bool enterPressed = ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter);
                    doApply |= labelEnter || enterPressed;
                    if(doApply && !gTrain.active && !gPendingTopology)
                    {
                        // Commit label
                        sel->label = editBuf;
                        sel->desc = descBuf;
                        // Commit value behavior
                        if(isOutput)
                        {
                            if(std::isfinite(pendingVal)) gLessonTargets[sel->id] = pendingVal;
                        } 
                        else if(sel->isInput)
                        {
                            if(pendingVal < 0.0f) pendingVal = 0.0f; else if(pendingVal > 1.0f) pendingVal = 1.0f;
                            sel->value = pendingVal;
                            sel->speed = std::max(0.0f, pendingSpeed);
                            liveInputs = false;
                            gPauseLiveInference = true;
                            gInferenceAccumulator = 0.0;
                        }
                        if(!gTrain.active) ForwardPass(*this, false);
                    }

                    ImGui::PopItemWidth();
                    ImGui::EndDisabled();
                }
            }
            }
            ImGui::EndChild();
        }

        // Local actions panel
        ImGui::Separator();
        ImGui::Checkbox("Live inputs (sin)", &liveInputs);
        ImGui::SameLine();
        if(ImGui::Button("Relayout")){ layoutLayers(canvasSize.x, canvasSize.y); }
        ImGui::SameLine();
        ImGui::BeginDisabled(gTrain.active || gPendingTopology.has_value());
        if(ImGui::Button("Add GRU layer")){ NG_InsertGRUAfterCurrent(*this); }
        ImGui::SameLine();
        if(ImGui::Button("Add neuron (N)")){ NG_AddNeuron(*this, currentLayer); }
        ImGui::SameLine();
        if(ImGui::Button("Remove last (D)")){ NG_RemoveLastNeuron(*this, currentLayer); }
        ImGui::EndDisabled();
    }
    ImGui::End();
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Engine (GLFW glue + loop)
// ─────────────────────────────────────────────────────────────────────────────
Engine* gEngine = nullptr;

static void cbFramebufferSize(GLFWwindow*, int w, int h){ if(gEngine) gEngine->onResize(w,h); }
static void cbMouseButton(GLFWwindow*, int b, int a, int m){ if(gEngine) gEngine->onMouseButton(b,a,m); }
static void cbCursorPos(GLFWwindow*, double x, double y){ if(gEngine) gEngine->onCursorPos(x,y); }
static void cbScroll(GLFWwindow*, double dx, double dy){ if(gEngine) gEngine->onScroll(dx,dy); }
static void cbKey(GLFWwindow*, int k, int s, int a, int m){ if(gEngine) gEngine->onKey(k,s,a,m); }

static inline std::string trim(const std::string& s)
{
    size_t a=s.find_first_not_of(" \t\r\n"); if(a==std::string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}

void Engine::loadTurbinConfDefaults(const std::string& path)
{
    std::ifstream f(path);
    if(!f) return;
    std::string line;
    auto kv = [](const std::string& ln, std::string& k, std::string& v){
        auto p = ln.find('='); if(p==std::string::npos) return false;
        k = trim(ln.substr(0,p)); v = trim(ln.substr(p+1)); return true;
    };
    while(std::getline(f,line))
    {
        if(line.empty() || line[0]=='#' || line[0]==';') continue;
        std::string k,v; if(!kv(line,k,v)) continue;
        if(k=="rpm") turbine.params.rpm = std::atof(v.c_str());
        else if(k=="blades") turbine.params.blades = std::atoi(v.c_str());
        else if(k=="shaftLength") turbine.params.shaftLength = std::atof(v.c_str());
        else if(k=="shaftRadius") turbine.params.shaftRadius = std::atof(v.c_str());
        else if(k=="hubRadius") turbine.params.hubRadius = std::atof(v.c_str());
        else if(k=="bladeLength") turbine.params.bladeLength = std::atof(v.c_str());
        else if(k=="bladeRootWidth") turbine.params.bladeRootWidth = std::atof(v.c_str());
        else if(k=="bladeTipWidth") turbine.params.bladeTipWidth = std::atof(v.c_str());
        else if(k=="bladeThickness") turbine.params.bladeThickness = std::atof(v.c_str());
    }
}

bool Engine::init(const AppConfig& cfgIn)
{
    cfg = cfgIn;
    if(!glfwInit()){ std::fprintf(stderr,"GLFW init fail\n"); return false; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Start maximized for better UX
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    GLFWwindow* win = glfwCreateWindow(cfg.width, cfg.height, cfg.title.c_str(), nullptr, nullptr);
    if(!win){ std::fprintf(stderr,"GLFW window fail\n"); glfwTerminate(); return false; }
    window = win;
    glfwMakeContextCurrent(win);
    // Fallback: ensure maximized
    glfwMaximizeWindow(win);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){ std::fprintf(stderr,"GLEW init fail\n"); return false; }
    glfwSwapInterval(cfg.vsync?1:0);

    gEngine = this;
    glfwSetFramebufferSizeCallback(win, cbFramebufferSize);
    glfwSetMouseButtonCallback(win, cbMouseButton);
    glfwSetCursorPosCallback(win, cbCursorPos);
    glfwSetScrollCallback(win, cbScroll);
    glfwSetKeyCallback(win, cbKey);

    glEnable(0x0B71/*GL_DEPTH_TEST*/);
    glEnable(0x0BE2/*GL_BLEND*/);
    glBlendFunc(0x0302/*SRC_ALPHA*/, 0x0303/*ONE_MINUS_SRC_ALPHA*/);

    if(!renderer.init(UITheme::Crimson())) return false;

    // Read turbin.conf as defaults
    loadTurbinConfDefaults("turbin.conf");
    cfg.rotate = true;
    turbine.params.rpm = 444.0f;
    turbine.params.current_rpm = 0.0f;
    turbine.rebuild();

    srand((unsigned int)time(NULL));
    gWindStrengthPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindSpeedPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindJitterPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindInterpolant = 0.0f;

    timer.last = glfwGetTime();
    // Initial camera looking slightly from above at origin
    cam.target = {0,0,0};
    cam.distance = 4.0f;
    cam.yaw = 0.7f;
    cam.pitch = 0.35f;

#ifdef USE_IMGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    SetupDefaultFonts(16.0f);
    // Fonts configured by SetupDefaultFonts()

    NG_BuildDinosaur(graph);
#endif
    
    LoadEngineConf("engine.conf");
    Logf("Engine %s initialized.", kAppVersion);
    return true;
}

void Engine::update()
{
    double t = glfwGetTime(); timer.update(t);
#ifdef USE_IMGUI
    UpdateInference(graph, timer.dt);
#endif

    const float targetRpm = cfg.rotate ? turbine.params.rpm : 0.0f;
    const float transitionTime = cfg.rotate
        ? std::max(0.1f, turbine.params.startupTime)
        : std::max(0.1f, turbine.params.shutdownTime);
    const float rpmReference = std::max({1.0f, turbine.params.rpm, turbine.params.current_rpm});
    const float maxRpmStep = rpmReference * (float)timer.dt / transitionTime;
    const float rpmDifference = targetRpm - turbine.params.current_rpm;
    turbine.params.current_rpm += std::clamp(rpmDifference, -maxRpmStep, maxRpmStep);
    if(std::fabs(rpmDifference) <= maxRpmStep) turbine.params.current_rpm = targetRpm;

    gWindInterpolant = turbine.params.rpm > 0.001f
        ? std::clamp(turbine.params.current_rpm / turbine.params.rpm, 0.0f, 1.0f)
        : 0.0f;

    // Update wind parameter phases for oscillation
    gWindStrengthPhase += 0.5f * (float)timer.dt;
    gWindSpeedPhase += 0.4f * (float)timer.dt;
    gWindJitterPhase += 0.6f * (float)timer.dt;

    if (gStrengthSineEnabled) 
    {
        gWind.strength = 150.0f + 150.0f * sin(gWindStrengthPhase);
    }

    if (gSpeedSineEnabled) 
    {
        gWind.speed = 2.5f + 2.5f * sin(gWindSpeedPhase);
    }

    if (gJitterSineEnabled) 
    {
        gWind.jitter = 0.25f + 0.25f * sin(gWindJitterPhase);
    }

    if(turbine.params.current_rpm > 0.001f)
    {
        float rps = turbine.params.current_rpm / 60.0f;
        float radps = rps * 6.28318530718f;
        rotorAngle += (float)(radps * timer.dt);
    }
    processCamera();
}

Mat4 Engine::proj() const 
{
    int w,h; glfwGetFramebufferSize((GLFWwindow*)window, &w, &h);
    float aspect = (h>0) ? (float)w/(float)h : 1.0f;
    return Mat4::perspective(deg2rad(55.0f), aspect, 0.05f, 100.0f);
}

Mat4 Engine::view() const { return cam.view(); }

#ifdef USE_IMGUI
// Keep track of HUD rectangle for overlay hole
static ImVec2 gHUDMin(0,0), gHUDMax(0,0);
static bool gHUDHasRect = false;

// Overlay training HUD in the center while training
static void DrawTrainingHUD()
{
    if(!gTrain.active) return;
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 sz = io.DisplaySize;
    ImVec2 winSz(560, 360);
    ImGui::SetNextWindowPos(ImVec2((sz.x-winSz.x)*0.5f, (sz.y-winSz.y)*0.5f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(winSz, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin("Training HUD", nullptr, flags);

    // capture rect for overlay hole
    ImVec2 pos = ImGui::GetWindowPos();
    ImVec2 siz = ImGui::GetWindowSize();
    gHUDMin = pos; gHUDMax = ImVec2(pos.x+siz.x, pos.y+siz.y); gHUDHasRect = true;
    ImGui::TextColored(ImVec4(0.9f,0.95f,1.0f,1), "Training in progress");
    ImGui::Separator();
    float last = gTrain.mseHistory.empty()? 0.0f : gTrain.mseHistory.back();
    ImGui::Text("Epoch: %d / %d", gTrain.epoch, gTrain.totalEpochs);
    ImGui::Text("MSE:   %.6f  (target: %s)", last, gTrain.targetMSE>0?std::to_string(gTrain.targetMSE).c_str():"-");
    ImGui::Text("Avg epoch time: %.2f ms", gTrain.avgEpochMs);
    double elapsed = (gTrain.startTime>0.0) ? (nowSeconds()-gTrain.startTime) : 0.0;
    ImGui::Text("Elapsed: %.1f s", elapsed);
    ImGui::Spacing();
    ImGui::PlotLines("MSE", gTrain.mseHistory.data(), (int)gTrain.mseHistory.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-1,200));
            // Axes and ticks for MSE plot
            {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 p0 = ImGui::GetItemRectMin();
                ImVec2 p1 = ImGui::GetItemRectMax();
                // y axis at left, x axis at bottom
                dl->AddLine(ImVec2(p0.x, p0.y), ImVec2(p0.x, p1.y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.8f,0.6f)), 1.0f);
                dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p1.x, p1.y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.8f,0.6f)), 1.0f);

                // ticks
                const int xticks = 10;
                const int yticks = 5;
                for(int i=0;i<=xticks;i++)
                {
                    float t = (float)i/(float)xticks;
                    float x = p0.x + t*(p1.x-p0.x);
                    dl->AddLine(ImVec2(x, p1.y), ImVec2(x, p1.y-5.0f), ImGui::GetColorU32(ImVec4(0.7f,0.7f,0.7f,0.6f)), 1.0f);
                }

                // Estimate y-range from recent history
                float ymin = 1e9f, ymax = -1e9f;
                int n = (int)gTrain.mseHistory.size();
                int wnd = n;
                for(int i=0;i<wnd;i++){ float v = gTrain.mseHistory[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
                if(!(ymax>ymin)) { ymin = 0.0f; ymax = 1.0f; }
                for(int i=0;i<=yticks;i++)
                {
                    float t = (float)i/(float)yticks;
                    float y = p1.y - t*(p1.y-p0.y);
                    dl->AddLine(ImVec2(p0.x, y), ImVec2(p0.x+5.0f, y), ImGui::GetColorU32(ImVec4(0.7f,0.7f,0.7f,0.6f)), 1.0f);

                    // optional labels: small text with value
                    float v = ymin + t*(ymax-ymin);
                    char buf[32]; std::snprintf(buf, sizeof(buf), "%.3f", v);
                    dl->AddText(ImVec2(p0.x+6.0f, y-8.0f), ImGui::GetColorU32(ImVec4(0.7f,0.7f,0.7f,0.9f)), buf);
                }
            }
    // Progress bar under the plot: percent of epochs completed
    {
        float pct = 0.0f;
        if(gTrain.totalEpochs > 0) pct = std::clamp(gTrain.epoch / (float)std::max(1, gTrain.totalEpochs), 0.0f, 1.0f);
        ImGui::Spacing();
        char pbuf[32]; std::snprintf(pbuf, sizeof(pbuf), "%d%%", (int)std::round(pct*100.0f));
        ImGui::ProgressBar(pct, ImVec2(-1, 0), pbuf);
    }
        
    ImGui::End();
}
#endif

void Engine::render()
{
    int w,h; glfwGetFramebufferSize((GLFWwindow*)window, &w, &h);
    glViewport(0,0,w,h);
    glClearColor(renderer.theme.bg.x, renderer.theme.bg.y, renderer.theme.bg.z, 1.0f);
    glClear(0x00004000/*GL_COLOR_BUFFER_BIT*/ | 0x00000100/*GL_DEPTH_BUFFER_BIT*/);

    Mat4 P = proj();
    Mat4 V = view();
    Mat4 VP = P * V;

    renderer.drawGrid(VP);

    // Fixed industrial materials: these deliberately do not follow UI themes.
    const Vec3 shaftSteel{0.34f, 0.38f, 0.41f};
    const Vec3 rotorBronze{0.58f, 0.34f, 0.14f};
    const Vec3 pipeCopper{0.55f, 0.25f, 0.10f};
    const Vec3 gearSteel{0.28f, 0.31f, 0.33f};
    const Vec3 steamColor{0.72f, 0.84f, 0.88f};

    Mat4 Mshaft = Mat4::translate({0,0,0});
    renderer.drawMesh(turbine.shaft, VP*Mshaft, Mshaft, shaftSteel, 0.32f);

    // Draw full rotor as one rotating assembly around Y axis at origin
    Mat4 Mrotor = Mat4::rotateY(rotorAngle);
    renderer.drawMesh(turbine.rotor, VP*Mrotor, Mrotor, rotorBronze, 0.52f);

    // Draw static pipes
    Mat4 Mpipes = Mat4::translate({0,0,0});
    renderer.drawMesh(turbine.pipes, VP*Mpipes, Mpipes, pipeCopper, 0.42f);

    // Steam lines (stylized) - always visible
    renderer.drawLines(turbine.schematicLines, VP, steamColor, 2.0f);

    // Draw meshing outer gears (counter-rotating), translated on X (left and right)
    float ratioL = (turbine.outerTeeth>0)? (turbine.rotorTeeth / (float)turbine.outerTeeth) : 1.0f;
    const float meshPhaseL = 3.1415926535f / std::max(1, turbine.outerTeeth);
    Mat4 MouterL = Mat4::translate({turbine.outerGearOffset, 0, 0}) * Mat4::rotateY(-rotorAngle*ratioL + meshPhaseL);
    renderer.drawMesh(turbine.outerGear, VP*MouterL, MouterL, gearSteel, 0.58f);
    float ratioR = (turbine.outerTeethR>0)? (turbine.rotorTeeth / (float)turbine.outerTeethR) : 1.0f;
    const float meshPhaseR = 3.1415926535f / std::max(1, turbine.outerTeethR);
    Mat4 MouterR = Mat4::translate({turbine.outerGearOffsetR, 0, 0}) * Mat4::rotateY(-rotorAngle*ratioR + meshPhaseR);
    renderer.drawMesh(turbine.outerGearR, VP*MouterR, MouterR, gearSteel, 0.58f);

#ifdef USE_IMGUI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

    // Auto-layout windows each frame (deterministic positions)
    {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 disp = io.DisplaySize;

        // Left top: Neural Editor (big)
        float neW = std::max(400.0f, disp.x * 0.62f);
        float neH = std::max(300.0f, disp.y * 0.62f);
        if(gShowEditorWindow)
        {
            bool openNe = true;
            graph.drawCanvas(openNe, neW - 20.0f, neH - 20.0f);
        }

        // Top-right: NeuralT control panel
        ImGuiCond layCond = gForceRelayoutWindows ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
        ImGui::SetNextWindowPos(ImVec2(disp.x - 360.0f, 10.0f), layCond);
        ImGui::SetNextWindowSize(ImVec2(350.0f, 270.0f), layCond);
        if(cfg.showUI && gShowNeuralTPanel)
        {
            ImGui::Begin("Steam Punk Uber Crap Turbine", nullptr, ImGuiWindowFlags_NoCollapse);
            char title[128];
            std::snprintf(title, sizeof(title), "NeuralT %s | dt: %.3f ms", kAppVersion, timer.dt*1000.0);
            ImGui::TextUnformatted(title);
            ImGui::Text("Total Neurons: %zu", graph.nodes.size());
            ImGui::Checkbox("VSync (V)", &cfg.vsync);
            ImGui::SameLine(); ImGui::Checkbox("Rotate (R)", &cfg.rotate);
            ImGui::SliderFloat("RPM", &turbine.params.rpm, 0.0f, 15000.0f);
            ImGui::SliderFloat("Startup Time", &turbine.params.startupTime, 0.1f, 60.0f, "%.1f s");
            ImGui::SliderFloat("Shutdown Time", &turbine.params.shutdownTime, 0.1f, 60.0f, "%.1f s");
            ImGui::Text("Current RPM: %.1f", turbine.params.current_rpm);
            int blades = turbine.params.blades;
            if(ImGui::SliderInt("Blades", &blades, 1, 10)){ turbine.params.blades = blades; turbine.rebuild(); }
            ImGui::Separator();
            ImGui::Text("Wind Effect");

            auto draw_sine_plot = [](const char* id, float phase) 
            {
                ImGui::PushID(id);
                ImVec2 plotSize(ImGui::GetContentRegionAvail().x - 60.f, 30);
                ImGui::InvisibleButton("plot", plotSize);
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                ImVec2 p0 = ImGui::GetItemRectMin();
                ImVec2 p1 = ImGui::GetItemRectMax();

                drawList->AddRectFilled(p0, p1, IM_COL32(30, 30, 35, 255));
                drawList->AddRect(p0, p1, IM_COL32(150, 150, 160, 255));

                const int num_segments = 50;
                ImVec2 points[num_segments + 1];
                for (int i = 0; i <= num_segments; ++i) 
                {
                    float t = (float)i / (float)num_segments;
                    float x = p0.x + t * (p1.x - p0.x);
                    float y = p0.y + (p1.y - p0.y) / 2.0f - (p1.y - p0.y) / 2.2f * sin(t * 6.0f + phase);
                    points[i] = ImVec2(x, y);
                }
                for (int i = 0; i < num_segments; ++i) 
                {
                    drawList->AddLine(points[i], points[i + 1], IM_COL32(255, 80, 80, 255), 1.5f);
                }
                ImGui::PopID();
            };

            if (gStrengthSineEnabled) ImGui::BeginDisabled();
            ImGui::SliderFloat("Strength", &gWind.strength, 0.0f, 300.0f, "%.2f");
            if (gStrengthSineEnabled) ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("##StrengthSine", &gStrengthSineEnabled);
            draw_sine_plot("StrengthPlot", gWindStrengthPhase);

            if (gSpeedSineEnabled) ImGui::BeginDisabled();
            ImGui::SliderFloat("Speed", &gWind.speed, 0.0f, 5.0f, "%.2f");
            if (gSpeedSineEnabled) ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("##SpeedSine", &gSpeedSineEnabled);
            draw_sine_plot("SpeedPlot", gWindSpeedPhase);

            if (gJitterSineEnabled) ImGui::BeginDisabled();
            ImGui::SliderFloat("Jitter", &gWind.jitter, 0.0f, 0.5f, "%.2f");
            if (gJitterSineEnabled) ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("##JitterSine", &gJitterSineEnabled);
            draw_sine_plot("JitterPlot", gWindJitterPhase);

            ImGui::Separator();
            ImGui::Text("Live Feed Wave");
            if(ImGui::Button(gLiveFeedWaveActive ? "Disable Wave (W)" : "Enable Wave (W)"))
            {
                gLiveFeedWaveActive = !gLiveFeedWaveActive;
                if(gLiveFeedWaveActive)
                {
                    gLiveFeedCurrentInput = 0;
                    gLiveFeedInputTimer = 0.0f;
                    gLiveFeedWavePhase = 0.0f;
                    Logf("Live feed wave effect activated");
                } 
                else 
                {
                    Logf("Live feed wave effect deactivated");
                }
            }
            if(gLiveFeedWaveActive)
            {
                ImGui::SliderFloat("Wave Speed", &gLiveFeedWaveSpeed, 0.5f, 5.0f, "%.1f");
                ImGui::SliderFloat("Input Delay", &gLiveFeedInputDelay, 0.1f, 1.0f, "%.2f s");
                ImGui::Text("Current Input: %d", gLiveFeedCurrentInput + 1);
            }
            
            ImGui::Separator();
            ImGui::Text("Input Controls");
            
            // Random Live Feed toggle button
            if(ImGui::Button(gRandomLiveFeedActive ? "Disable Random Feed (Q)" : "Enable Random Feed (Q)"))
            {
                gRandomLiveFeedActive = !gRandomLiveFeedActive;
                int inputCount = 0;
                for(const auto& L : graph.layers)
                {
                    if(L.kind != LayerKind::Input) continue;
                    for(int idx : L.nodeIdx)
                    {
                        if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                        auto& n = graph.nodes[idx];
                        if(!n.isInput) continue;
                        
                        if(gRandomLiveFeedActive)
                        {
                            // Set random amplitude (0.1 to 1.0)
                            n.amp = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;

                            // Set random speed (0.1 to 1.0 Hz)
                            n.speed = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;

                            // Enable live feed
                            n.live = true;

                            // Reset phase for immediate effect
                            n.phase = 0.0f;
                        } 
                        else 
                        {
                            // Disable live feed
                            n.live = false;
                            n.speed = 0.0f;
                        }
                        inputCount++;
                    }
                }
                if(gRandomLiveFeedActive)
                {
                    Logf("Random live feed applied to %d input neurons", inputCount);
                } 
                else 
                {
                    Logf("Random live feed disabled for %d input neurons", inputCount);
                }
            }
            
            // Zero All Inputs button
            if(ImGui::Button("Zero All Inputs"))
            {
                int inputCount = 0;
                for(const auto& L : graph.layers)
                {
                    if(L.kind != LayerKind::Input) continue;
                    for(int idx : L.nodeIdx)
                    {
                        if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                        auto& n = graph.nodes[idx];
                        if(!n.isInput) continue;
                        
                        // Set all input values to zero
                        n.value = 0.0f;

                        // Disable live feed
                        n.live = false;
                        n.speed = 0.0f;
                        n.phase = 0.0f;
                        inputCount++;
                    }
                }
                // Also disable global live inputs
                graph.liveInputs = false;

                // Disable wave effect
                gLiveFeedWaveActive = false;
                gRandomLiveFeedActive = false;
                Logf("All %d input neurons set to zero and live feeds disabled", inputCount);
            }

            ImGui::End();
        }

        // Left-bottom: Log window
        float logW = neW - 20.0f;
        float logH = disp.y - neH - 30.0f;
        if(logH < 150.0f) logH = 150.0f;
        if(gShowLogWindow)
        {
        ImGui::SetNextWindowPos(ImVec2(10.0f, neH), layCond);
        ImGui::SetNextWindowSize(ImVec2(logW, logH), layCond);
            ImGui::Begin("Log", nullptr, ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoSavedSettings);
            ImGui::Text("Lessons: %d", gLessonsCount);
            ImGui::Separator();
            ImGui::BeginChild("log_scroller", ImVec2(0, -28), true, ImGuiWindowFlags_HorizontalScrollbar);
            for(const auto& s : gLogLines) ImGui::TextUnformatted(s.c_str());
            if(gLogAutoScroll) ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
            ImGui::Checkbox("Auto scroll", &gLogAutoScroll);
            ImGui::SameLine();
            if(ImGui::Button("Clear")) gLogLines.clear();
            ImGui::End();
        }

        // Right-bottom: Training parameters
        float paramsW = disp.x - (10.0f + logW) - 20.0f;
        if(paramsW < 320.0f) paramsW = 320.0f;
        float paramsH = std::max(220.0f, disp.y * 0.38f);
        if(gShowParamsWindow)
        {
        ImGui::SetNextWindowPos(ImVec2(disp.x - paramsW - 10.0f, disp.y - paramsH - 10.0f), layCond);
        ImGui::SetNextWindowSize(ImVec2(paramsW, paramsH), layCond);
            ImGui::Begin("Training parameters", nullptr, ImGuiWindowFlags_NoCollapse|ImGuiWindowFlags_NoSavedSettings);
        if(!gTrain.active)
        {
            ImGui::BeginDisabled(gPendingTopology.has_value());
            ImGui::SliderInt("Epochs", &gTrain.totalEpochs, 10, 5000);
            ImGui::SliderFloat("Learning rate", &gTrain.lr, 1e-5f, 1.0f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("L1", &gTrain.l1, 0.0f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("L2", &gTrain.l2, 0.0f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::InputFloat("Target MSE (<=0 off)", &gTrain.targetMSE, 0.0f, 0.0f, "%.6f");
            ImGui::Separator();
            // Row 1: Rabbit Example, Help
            if(ImGui::Button("Rabbit Example"))
            {
                NeuralGraph candidate;
                std::vector<TrainSample> lessons;
                // Build 6-10-8-5 network for rabbit scenario
                NG_BuildTopology(candidate, 6, {10,8}, 5);
                // Label inputs/outputs with descriptions
                {
                    static const char* kInputs[6]  = 
                    {
                        "CarrotProximity", "FoxProximity", "HungerLevel",
                        "ThreatLevel", "BushProximity", "SafetyLevel"
                    };

                    static const char* kInputDescs[6] = 
                    {
                        "Proximity to nearest carrot (1=close, 0=far)",
                        "Proximity to nearest fox (1=close, 0=far)",
                        "Current hunger level (1=very hungry, 0=full)",
                        "Perceived threat level (1=high danger, 0=safe)",
                        "Proximity to nearest bush (1=close, 0=far)",
                        "Current safety level (1=very safe, 0=exposed)"
                    };

                    static const char* kActions[5] = 
                    {
                        "Flee", "Hide", "MoveToFood", "Eat", "Rest"
                    };

                    static const char* kActionDescs[5] = 
                    {
                        "Action to flee from danger",
                        "Action to hide in nearby cover",
                        "Action to move towards food",
                        "Action to eat available food",
                        "Action to rest and recover"
                    };

                    if(!candidate.layers.empty() && candidate.layers.front().kind==LayerKind::Input)
                    {
                        auto& L = candidate.layers.front();
                        for(size_t i=0;i<L.nodeIdx.size() && i<6;i++)
                        { 
                            candidate.nodes[L.nodeIdx[i]].label = std::string(kInputs[i]);
                            candidate.nodes[L.nodeIdx[i]].desc = std::string(kInputDescs[i]);
                        }
                    }
                    if(!candidate.layers.empty())
                    {
                        auto& L = candidate.layers.back();
                        if(L.kind==LayerKind::Output)
                        {
                            for(size_t i=0;i<L.nodeIdx.size() && i<5;i++)
                            { 
                                candidate.nodes[L.nodeIdx[i]].label = std::string(kActions[i]);
                                candidate.nodes[L.nodeIdx[i]].desc = std::string(kActionDescs[i]);
                            }
                        }
                    }
                }

                // Randomize hidden layers for a fresh start
                NG_RandomizeHidden(candidate, 0.8f, 0.3f);

                // Dataset from specification (6 inputs, 5 outputs one-hot) with descriptions
                lessons.clear();
                auto make_sample = [&](const std::array<float,6>& in, const std::array<float,5>& out, const std::string& desc)
                {
                    TrainSample s; s.in.assign(in.begin(), in.end()); s.out.assign(out.begin(), out.end()); s.desc = desc; return s;
                };

                lessons.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.8f, 0.3f, 0.6f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Fox nearby, no food → Flee"));
                lessons.push_back(make_sample({0.0f, 0.8f, 0.0f, 0.7f, 0.9f, 0.5f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Fox nearby, bush close → Hide"));
                lessons.push_back(make_sample({0.0f, 0.0f, 0.9f, 0.0f, 0.2f, 0.1f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Hungry, sees carrot, no threat → Move to food"));
                lessons.push_back(make_sample({0.8f, 0.0f, 0.9f, 0.0f, 0.1f, 0.2f}, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, "At carrot, hungry, safe → Eat"));
                lessons.push_back(make_sample({0.1f, 0.0f, 0.0f, 0.0f, 0.2f, 0.9f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Full, no threat → Rest"));
                lessons.push_back(make_sample({0.7f, 0.8f, 0.6f, 0.7f, 0.2f, 0.3f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Fox and food nearby, low hunger → Flee"));
                lessons.push_back(make_sample({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Tired, full, no signals → Rest"));
                lessons.push_back(make_sample({0.0f, 0.2f, 0.9f, 0.1f, 0.1f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Very hungry, sees carrot, fox far → Move to food"));
                lessons.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.7f, 0.0f, 0.2f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Low energy, smells fox, no shelter → Flee"));
                lessons.push_back(make_sample({0.6f, 0.0f, 0.9f, 0.0f, 0.1f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Full, sees carrot, no fox → Rest"));
                lessons.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.6f, 0.5f, 0.4f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Strong fox smell, no shelter → Hide"));
                lessons.push_back(make_sample({0.0f, 0.2f, 0.0f, 0.4f, 0.9f, 0.6f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Fox far, bush close → Hide"));
                lessons.push_back(make_sample({0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.6f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Slightly hungry, safe → Rest"));
                lessons.push_back(make_sample({0.7f, 0.1f, 0.0f, 0.2f, 0.0f, 0.3f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Fox very far, smells carrot → Move to food"));
                lessons.push_back(make_sample({0.5f, 0.2f, 0.4f, 0.1f, 0.0f, 0.9f}, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, "Full, slight fox smell → Eat (safe, enough energy)"));
                lessons.push_back(make_sample({0.6f, 0.0f, 0.7f, 0.0f, 0.1f, 0.7f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "High energy, senses carrot and safety → Rest"));
                lessons.push_back(make_sample({0.0f, 0.0f, 0.7f, 0.0f, 0.8f, 0.3f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Carrot visible behind bush, no fox → Move to food"));
                NG_RequestTopology(graph,std::move(candidate),std::move(lessons),[]()
                {
                    gTrain.lr=0.01f; gTrain.l1=0; gTrain.l2=0.0001f;
                    gTrain.totalEpochs=3000; gTrain.targetMSE=0.02f;
                    SaveEngineConf("engine.conf");
                    Logf("Rabbit Example initialized: lessons=%d",gLessonsCount);
                });
            }

            ImGui::SameLine();
            if(ImGui::Button("Deep GRU Factory"))
            {
                NeuralGraph candidate;
                std::vector<TrainSample> lessons;
                NG_BuildDeepGRUTopology(candidate);
                BuildFactorySequenceLessons(candidate,lessons);
                NG_RequestTopology(graph,std::move(candidate),std::move(lessons),[]()
                {
                    gTrain.lr=0.001f; gTrain.l1=0; gTrain.l2=0.0001f;
                    gTrain.totalEpochs=5000; gTrain.targetMSE=0.015f;
                    SaveEngineConf("engine.conf");
                    Logf("Deep GRU Factory initialized: sequence lessons=%d",gLessonsCount);
                });
            }

            ImGui::SameLine();
            if(ImGui::Button("Help (H)")) gShowHelp = !gShowHelp;
            // Row 2: Start, Train To Target, Save Config
            if(ImGui::Button("Start")) StartTraining(graph, false);
            ImGui::SameLine();
            if(ImGui::Button("Train To Target (T)")) StartTraining(graph, true);

            ImGui::SameLine();
            if(ImGui::Button("Save Config") && SaveEngineConf("engine.conf")) Logf("Config saved: engine.conf");
            ImGui::EndDisabled();
        }
        else
        {
            ImGui::TextDisabled("Training in progress...");
            if(ImGui::Button("Stop"))
            {
                gTrain.active = false;
                gTrain.done = true;

                // Ensure any dimming overlay and HUD are cleared when stopping
                gShowTrainingWindow = false;
                gHUDHasRect = false;

                // Show result window until confirmed
                extern bool gCelebrationVisible; // already static above in same TU
                extern bool gTargetReachedVisual;
                extern bool gTrainInterrupted;
                gCelebrationVisible = true;
                gTargetReachedVisual = false;
                gTrainInterrupted = true;
                SaveEngineConf("engine.conf");
                Logf("Training stopped at epoch %d (MSE=%.6f).", gTrain.epoch, gTrain.lastMSE);
            }
        }

        ImGui::Separator();
        ImGui::BeginDisabled(gTrain.active || gPendingTopology.has_value());
        ImGui::Text("Dataset: %d lessons", gLessonsCount);
        ImGui::TextWrapped("Lesson outputs use the applied Lesson target values; unset targets are zero. Predictions are shown separately.");
        if(ImGui::Button("Clear lesson targets")) ClearLessonTargets();

        // Row 3: Add Lesson, Randomize Hidden, Delete All Lessons
        if(ImGui::Button("Add Lesson (M)"))
        {
            std::vector<float> in,out; CollectIO(graph,in,out, true);
            TrainSample sample;
            sample.in = std::move(in);
            sample.out = std::move(out);
            gDataset.push_back(std::move(sample));
            gLessonsCount=(int)gDataset.size();
            SaveEngineConf("engine.conf");
            Logf("Lesson added. Total=%d", gLessonsCount);
        }

        ImGui::SameLine();
        if(ImGui::Button("Add sequence frame"))
        {
            std::vector<float> in, out;
            CollectIO(graph, in, out);
            gSequenceDraft.push_back(std::move(in));
            Logf("Sequence frame added (%zu).", gSequenceDraft.size());
        }
        ImGui::SameLine();
        if(ImGui::Button("Save sequence lesson"))
        {
            if(gSequenceDraft.empty())
            {
                Logf("Add at least one sequence frame first.");
            }
            else
            {
                std::vector<float> in, out;
                CollectIO(graph, in, out, true);
                TrainSample sample;
                sample.sequence = gSequenceDraft;
                sample.in = gSequenceDraft.back();
                sample.out = std::move(out);
                sample.desc = "Custom sequence";
                gDataset.push_back(std::move(sample));
                gSequenceDraft.clear();
                gLessonsCount = (int)gDataset.size();
                Logf("Sequence lesson saved. Total=%d", gLessonsCount);
            }
        }
        ImGui::SameLine();
        if(ImGui::Button("Clear sequence")) gSequenceDraft.clear();
        ImGui::TextDisabled("Sequence draft: %zu frames (set inputs, add frame, repeat, then apply lesson targets and save)",
                            gSequenceDraft.size());

        ImGui::SameLine();
        if(ImGui::Button("Randomize Hidden (X)"))
        {
            NG_RandomizeHidden(graph, 1.0f, 0.5f);
            Logf("Hidden layers randomized.");
        }

        ImGui::SameLine();
        if(ImGui::Button("Delete All Lessons"))
        {
            gDataset.clear(); gLessonsCount = 0;
            SaveEngineConf("engine.conf");
            Logf("All lessons deleted. forever. it was shit anyway..");
        }

        // Row 4: Save/Load network
        if(ImGui::Button("Save Network (S)"))
        {
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            // timestamped default name
            {
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#if defined(_WIN32)
                localtime_s(&tm, &t);
#else
                localtime_r(&t, &tm);
#endif
                char buf[128];
                std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
                std::snprintf(gSaveName, sizeof(gSaveName), "%s", buf);
            }
            gShowSaveMenu = true;
        }

        ImGui::SameLine();
        if(ImGui::Button("Load Network (O)"))
        {
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            gShowLoadMenu = true;
        }

        // Lessons table with editable descriptions and actions
        ImGui::Separator();
        static std::vector<std::array<char,256>> lessonDescBuf;
        if(lessonDescBuf.size() != gDataset.size())
        {
            lessonDescBuf.resize(gDataset.size());
            for(size_t i=0;i<gDataset.size();++i)
            {
                std::snprintf(lessonDescBuf[i].data(), lessonDescBuf[i].size(), "%s", gDataset[i].desc.c_str());
            }
        }

        int pendingDelete = -1;
        if(ImGui::BeginTable("lessons_table", 6, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_Resizable|ImGuiTableFlags_SizingStretchSame))
        {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 40.0f);
            ImGui::TableSetupColumn("steps", ImGuiTableColumnFlags_WidthFixed, 55.0f);
            ImGui::TableSetupColumn("in");
            ImGui::TableSetupColumn("out");
            ImGui::TableSetupColumn("desc", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("actions", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableHeadersRow();
            int showN = (int)gDataset.size();
            for(int i=0;i<showN;i++)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); 
                ImGui::Text("%d", i);
                auto& s = gDataset[i];
                ImGui::TableSetColumnIndex(1);
                const bool hasSequence = !s.sequence.empty();
                bool sequenceExpanded = false;
                ImGui::PushID(i + 200000);
                if(hasSequence)
                {
                    // TreeNode keeps expansion state per lesson and clearly marks sequences.
                    sequenceExpanded = ImGui::TreeNodeEx("sequence", ImGuiTreeNodeFlags_SpanAvailWidth,
                                                         "SEQ (%zu)", s.sequence.size());
                    if(ImGui::IsItemHovered())
                        ImGui::SetTooltip("Click to show/hide saved sequence frames");
                }
                else
                    ImGui::TextDisabled("single");
                ImGui::PopID();
                auto vecToStr=[&](const std::vector<float>& v)
                { 
                    std::string r="["; 
                    for(size_t j=0;j<v.size();++j)
                    { 
                        char b[32]; 
                        std::snprintf(b,sizeof(b),"%.2f", v[j]); 
                        r+=b; 
                        if(j+1<v.size()) r+=","; 
                    } 
                    r+="]"; 
                    return r; 
                };

                ImGui::TableSetColumnIndex(2);
                { 
                    std::string ins = vecToStr(s.in); 
                    ImGui::TextUnformatted(ins.c_str()); 
                }

                ImGui::TableSetColumnIndex(3);
                {
                    std::string outs= vecToStr(s.out); 
                    ImGui::TextUnformatted(outs.c_str()); 
                }

                ImGui::TableSetColumnIndex(4);
                ImGui::PushID(i);
                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##desc", lessonDescBuf[i].data(), lessonDescBuf[i].size());
                ImGui::PopID();
                ImGui::TableSetColumnIndex(5);
                ImGui::PushID(i+100000);
                if(ImGui::Button("Save"))
                {
                    s.desc = std::string(lessonDescBuf[i].data());
                    Logf("Lesson %d description saved.", i);
                }
                ImGui::SameLine();
                if(ImGui::Button("Delete"))
                {
                    pendingDelete = i;
                }
                ImGui::PopID();
                if(sequenceExpanded)
                {
                    // Detail row directly beneath the owning lesson.
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextDisabled("  ");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextDisabled("frames");
                    ImGui::TableSetColumnIndex(2);
                    for(size_t frameIndex = 0; frameIndex < s.sequence.size(); ++frameIndex)
                    {
                        const std::string values = vecToStr(s.sequence[frameIndex]);
                        ImGui::Text("t%zu: %s", frameIndex + 1, values.c_str());
                    }
                    ImGui::TableSetColumnIndex(3);
                    ImGui::TextDisabled("Target:");
                    ImGui::TextWrapped("%s", vecToStr(s.out).c_str());
                    ImGui::TableSetColumnIndex(4);
                    ImGui::TextDisabled("Ordered input frames; target applies to final step.");
                    // TreeNodeEx is closed after rendering its details.
                    ImGui::TreePop();
                }
            }
            ImGui::EndTable();
        }
        if(!gTrain.active && pendingDelete >= 0 && pendingDelete < (int)gDataset.size())
        {
            gDataset.erase(gDataset.begin() + pendingDelete);
            gLessonsCount = (int)gDataset.size();
            if((int)lessonDescBuf.size() > pendingDelete) lessonDescBuf.erase(lessonDescBuf.begin() + pendingDelete);
            Logf("Lesson %d deleted.", pendingDelete);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        if(ImGui::Button("Save network (S)"))
        {
            std::error_code _ec; std::filesystem::create_directories("saved", _ec);
            // timestamped default name
            {
                std::time_t t = std::time(nullptr);
                std::tm tm{};
#if defined(_WIN32)
                localtime_s(&tm, &t);
#else
                localtime_r(&t, &tm);
#endif
                char buf[128];
                std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
                std::snprintf(gSaveName, sizeof(gSaveName), "%s", buf);
            }
            gShowSaveMenu = true;
        }

        ImGui::SameLine();
        if(ImGui::Button("Load network (O)"))
        {
            std::error_code _ec; 
            std::filesystem::create_directories("saved", _ec);
            gShowLoadMenu = true;
        }
            ImGui::End();
        }
    }

    // Load menu (choose a network from 'saved')
    if(gShowLoadMenu)
    {
        ImGui::SetNextWindowSize(ImVec2(520, 420), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Load Network", &gShowLoadMenu, ImGuiWindowFlags_NoCollapse))
        {
            static std::vector<std::string> files;
            static int selected = -1;

            // Refresh list on open: simple heuristic
            if(ImGui::IsWindowAppearing())
            {
                files.clear(); selected = -1;
                std::error_code ec; std::filesystem::create_directories("saved", ec);
                for(const auto& de : std::filesystem::directory_iterator("saved", ec))
                {
                    if(!de.is_regular_file()) continue;
                    auto p = de.path();
                    if(p.extension() == ".json") files.push_back(p.filename().string());
                }
                std::sort(files.begin(), files.end());
            }

            if(ImGui::Button("Refresh"))
            {
                files.clear(); selected = -1;
                std::error_code ec;
                for(const auto& de : std::filesystem::directory_iterator("saved", ec))
                {
                    if(!de.is_regular_file()) continue;
                    auto p = de.path();
                    if(p.extension() == ".json") files.push_back(p.filename().string());
                }
                std::sort(files.begin(), files.end());
            }

            ImGui::SameLine();
            if(ImGui::Button("Cancel"))
            {
                gShowLoadMenu = false;
            }
            ImGui::Separator();

            ImGui::Text("Directory: %s", "saved/");
            ImVec2 listSize = ImVec2(-1, ImGui::GetTextLineHeightWithSpacing()*14.0f);
            if(ImGui::BeginListBox("##saved_list", listSize))
            {
                for(int i=0;i<(int)files.size();++i)
                {
                    bool sel = (i==selected);
                    if(ImGui::Selectable(files[i].c_str(), sel)) selected = i;
                    if(sel) ImGui::SetItemDefaultFocus();
                    if(ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
                    {
                        // Double-click to load
                        std::string path = std::string("saved/") + files[i];
                        std::string msg;
                        if(NG_LoadJson(graph, path, &msg))
                        {
                            gToastMsg = std::string(u8"Network loaded: ") + files[i];
                            Logf("Network loaded: %s", path.c_str());
                            gToastT = glfwGetTime();
                            gShowLoadMenu = false;
                        } 
                        else 
                        {
                            gToastMsg = std::string(u8"Load failed: ") + msg;
                            Logf("Load failed: %s", msg.c_str());
                            gToastT = glfwGetTime();
                        }
                    }
                }
                ImGui::EndListBox();
            }

            bool canLoad = selected >= 0 && selected < (int)files.size();
            if(!files.empty()) ImGui::Text("Selected: %s", canLoad ? files[selected].c_str() : "");
            if(ImGui::Button("Load") && canLoad)
            {
                std::string path = std::string("saved/") + files[selected];
                std::string msg;
                if(NG_LoadJson(graph, path, &msg))
                {
                    gToastMsg = std::string(u8"Network loaded: ") + files[selected];
                    Logf("Network loaded: %s", path.c_str());
                    gToastT = glfwGetTime();
                    gShowLoadMenu = false;
                } 
                else 
                {
                    gToastMsg = std::string(u8"Load failed: ") + msg;
                    Logf("Fuck! Load failed: %s", msg.c_str());
                    gToastT = glfwGetTime();
                }
            }
        }
        ImGui::End();
    }

    // Save menu (enter file name to save under 'saved/')
    if(gShowSaveMenu)
    {
        ImGui::SetNextWindowSize(ImVec2(480, 140), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Save Network", &gShowSaveMenu, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::Text("Directory: saved/");
            ImGui::InputText("File name", gSaveName, IM_ARRAYSIZE(gSaveName));
            
            // Check for Enter key press to confirm save - only when input field is focused
            bool enterPressed = (ImGui::IsItemFocused() || ImGui::IsItemActive()) && 
                               (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter));
            bool shouldSave = false;
            
            ImGui::Spacing();
            if(ImGui::Button("Save"))
            {
                shouldSave = true;
            }

            ImGui::SameLine();
            if(ImGui::Button("Cancel")) gShowSaveMenu = false;
            
            // Handle Enter key or Save button
            if(shouldSave || enterPressed)
            {
                std::string path = gSaveName;
                // ensure .json
                if(path.size()<5 || path.substr(path.size()-5) != ".json") path += ".json";
                std::error_code ec; std::filesystem::create_directories("saved", ec);
                if(NG_SaveJson(graph, path))
                {
                    Logf("Network saved: %s", path.c_str());
                    gShowSaveMenu = false;
                    if(SaveEngineConf("engine.conf"))
                    {
                        gToastMsg = "Network saved: " + path;
                        gToastT = glfwGetTime();
                    }
                    else Logf("Network was saved, but engine.conf could not be saved.");
                }
            }
        }
        ImGui::End();
    }

    // Help overlay (keybinds + quick start)
    if(gShowHelp)
    {
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImGui::SetNextWindowPos(ImVec2(18, 18), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImVec2 helpCenter = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(helpCenter, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        
        ImGui::SetNextWindowBgAlpha(0.92f);

        ImGui::SetNextWindowPos(helpCenter, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::Begin("Help & Shortcuts", &gShowHelp,
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings);
        char help_title[128];
        std::snprintf(help_title, sizeof(help_title), "NeuralT - Help & Shortcuts (%s)", kAppVersion);
        ImGui::TextColored(ImVec4(1,0.95f,0.6f,1), help_title);
        ImGui::Separator();
        ImGui::BulletText("H                    - toggle this help");
        ImGui::BulletText("F1                   - toggle Neural Editor");
        ImGui::BulletText("F2                   - NeuralT panel (visual settings)");
        ImGui::BulletText("F3                   - Log window");
        ImGui::BulletText("F4                   - Training parameters");
        ImGui::BulletText("F5                   - Help");
        ImGui::BulletText("F6                   - Training HUD");
        ImGui::BulletText("[ / ]                - previous / next layer");
        ImGui::BulletText("1..9                 - select layer by index");
        ImGui::BulletText("Ctrl+N               - new network (keep Input and Output only)");
        ImGui::BulletText("N                    - add neuron to current layer");
        ImGui::BulletText("A                    - add a hidden layer after Input, Hidden or GRU");
        ImGui::BulletText("D                    - remove last neuron from current layer");
        ImGui::BulletText("Delete               - remove current hidden/GRU layer and reconnect neighbours");
        ImGui::BulletText("Topology changes require confirmation to clear existing lessons and sequence drafts");
        ImGui::BulletText("X                    - randomize hidden-layer weights");
        ImGui::BulletText("S                    - save network (open Save dialog)");
        ImGui::BulletText("O                    - load network (open Load dialog)");
        ImGui::BulletText("M                    - add a lesson from inputs and applied lesson targets");
        ImGui::BulletText("B                    - start training on collected lessons");
        ImGui::BulletText("T                    - train to target (fine-tune until Target MSE)");
        ImGui::BulletText("W                    - toggle live feed wave effect");
        ImGui::BulletText("Q                    - random live feed for all input neurons");
        ImGui::BulletText("TAB                  - toggle live feed on all inputs");
        ImGui::BulletText("Space                - toggle live feed for selected input");
        ImGui::BulletText("Home                 - show and re-layout all primary windows");
        ImGui::BulletText("Esc                  - stop training, or close the application");
        ImGui::BulletText("V / R                - VSync / smooth turbine start-stop");
        ImGui::BulletText("Shift+T              - change MLP theme");
        ImGui::BulletText("Shift+G              - change GRU theme (16 palettes)");
        ImGui::BulletText("C                    - change paired MLP + GRU theme (16 pairs)");
        ImGui::Separator();
        ImGui::Text("Creating example networks");
        ImGui::BulletText("Use buttons like 'Rabbit Example' in Training parameters for presets");
        ImGui::BulletText("Build manually: add layers (A), neurons (N), edit in Neural Editor");
        ImGui::BulletText("Set labels and descriptions in neuron editor (select neuron)");
        ImGui::BulletText("Set Lesson target for each output and Apply; unspecified targets are zero");
        ImGui::BulletText("Add lessons (M) with current inputs and these targets");
        ImGui::BulletText("GRU: Reset memory, then set inputs, Apply and Next sample for each step");
        ImGui::BulletText("Train (B or T) after setting parameters");
        ImGui::BulletText("Mouse                - LMB orbit, MMB or Shift+LMB pan, wheel zoom");
        ImGui::Separator();
        ImGui::Text("Training parameters");
        ImGui::BulletText("Epochs - number of full passes over the dataset");
        ImGui::BulletText("Learning rate - step size (typical start ~0.001)");
        ImGui::BulletText("L1 - L1 regularization (|w|), encourages sparsity");
        ImGui::BulletText("L2 - L2 regularization (w^2), stabilizes training");
        ImGui::BulletText("Target MSE - early stop threshold; <= 0 disables");
        ImGui::BulletText("MSE is evaluated on all training lessons after each epoch, with fixed weights");
        ImGui::BulletText("Sequence targets apply to the final step; GRU memory resets for each lesson");
        ImGui::Separator();
        ImGui::Text("Dataset tips");
        ImGui::BulletText("Normalize inputs/outputs to the 0..1 range");
        ImGui::BulletText("Collect diverse examples covering edge cases");
        ImGui::BulletText("If loss plateaus, adjust epochs or learning rate; consider lowering L2");
        ImGui::Separator();
        ImGui::Text("Quick start");
        ImGui::BulletText("1) Add lessons: press M or use the panel (saved to lessons.json)");
        ImGui::BulletText("2) Set epochs, learning rate, L1/L2, optional Target MSE in 'Training parameters'");
        ImGui::BulletText("3) Start: click 'Start' or press B (HUD shows in the center)");
        ImGui::BulletText("4) Watch MSE and epoch progress (The lower the better)"); 
        ImGui::BulletText("ESC is emergency 'Stop' and interrupts training just in case");
        ImGui::BulletText("5) On finish: Skynet will be launched.");
        ImGui::Separator();
        ImGui::Text("Metrics");
        ImGui::BulletText("IQ gain - heuristic metric: percentage decrease in MSE vs. start");
        ImGui::BulletText("Formula: IQ_gain = round(max(0, (MSE_start - MSE_final)/max(MSE_start,1e-6)) * 100)");
        ImGui::BulletText("Interpretation: baseline IQ 0 like you");
        ImGui::End();
        }

    if(gOpenTopologyPopup)
    {
        ImGui::OpenPopup("Topology change###TopologyChange");
        gOpenTopologyPopup=false;
    }
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(),ImGuiCond_Appearing,ImVec2(0.5f,0.5f));
    if(ImGui::BeginPopupModal("Topology change###TopologyChange",nullptr,ImGuiWindowFlags_AlwaysAutoResize))
    {
        if(gPendingTopology)
        {
            ImGui::TextUnformatted(u8"Changing the topology will delete all lessons and the sequence draft.");
            ImGui::Text(u8"Lessons: %zu. Draft frames: %zu.",gDataset.size(),gSequenceDraft.size());
            if(!gPendingTopology->replacementLessons.empty())
                ImGui::Text(u8"The selected example will then add its own lessons: %zu.",gPendingTopology->replacementLessons.size());
            ImGui::Separator();
            if(ImGui::Button(u8"Change topology and delete lessons"))
            {
                auto change=std::move(*gPendingTopology);
                gPendingTopology.reset();
                NG_CommitTopology(graph,std::move(change));
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if(ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
            {
                gPendingTopology.reset();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SetItemDefaultFocus();
        }
        else ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // Toast overlay
    if(!gToastMsg.empty())
    {
        double now = glfwGetTime();
        if(now - gToastT < 4.0)
        {
            ImGui::SetNextWindowBgAlpha(0.92f);
            ImGui::SetNextWindowPos(ImVec2(18, 18), ImGuiCond_Always);
            ImGui::Begin("Message", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextUnformatted(gToastMsg.c_str());
            ImGui::End();
        }
    }

    // Training step & result banner (state is file-scope globals)
    if(gTrain.active || gShowTrainingWindow)
    {
        // run one epoch per frame while active
        if(gTrain.active)
        {
            double t0 = nowSeconds();
            gInferenceNeedsReset = true;
            float mse = TrainOneEpoch(graph, gTrain.lr, gTrain.l1, gTrain.l2);
            if(!std::isfinite(mse))
            {
                gTrain.active = false;
                gTrain.done = false;
                gShowTrainingWindow = false;
                gHUDHasRect = false;
                gCelebrationVisible = false;
                gTargetReachedVisual = false;
                gToastMsg = "Training stopped: " + gTrainingError;
                gToastT = nowSeconds();
                Logf("Training stopped: %s; model was not saved.", gTrainingError.c_str());
            }
            else
            {
                gTrain.lastMSE = mse;
                gTrain.mseHistory.push_back(mse);
                gTrain.epoch++;
                double t1 = nowSeconds();
                double dt = (t1 - t0);
                gTrain.avgEpochMs = (gTrain.avgEpochMs*0.9 + dt*1000.0*0.1);

                // Early stop on target MSE
                bool targetReached = (gTrain.targetMSE>0.0f && mse <= gTrain.targetMSE);

                if(gTrain.epoch >= gTrain.totalEpochs || targetReached)
                {
                    gTrain.active=false; gTrain.done=true;
                    gShowTrainingWindow = false;
                    gHUDHasRect = false;
                    SaveEngineConf("engine.conf");
                    const bool saved=NG_SaveJson(graph, "network_trained.json");
                    Logf("Trained model %s", saved ? "saved: network_trained.json" : "could not be saved; it remains in memory.");
                    Logf("Training finished: epoch=%d, MSE=%.6f%s.",
                         gTrain.epoch, mse, targetReached?" (target eliminated.)": "");
                    gCelebrationVisible = true;
                    gTrainDoneTime = nowSeconds();
                    gTargetReachedVisual = targetReached;
                    gTrainInterrupted = false;
                }
            }
        }

        // draw center HUD (MSE plot etc.)
        DrawTrainingHUD();

        // Darken the whole screen except HUD window
        if(gHUDHasRect)
        {
            ImDrawList* fg = ImGui::GetForegroundDrawList();
            ImVec2 sz = ImGui::GetIO().DisplaySize;
            ImU32 shade = ImGui::GetColorU32(ImVec4(0,0,0,0.6f));
            // top
            fg->AddRectFilled(ImVec2(0,0), ImVec2(sz.x, gHUDMin.y), shade);
            // left
            fg->AddRectFilled(ImVec2(0,gHUDMin.y), ImVec2(gHUDMin.x, gHUDMax.y), shade);
            // right
            fg->AddRectFilled(ImVec2(gHUDMax.x, gHUDMin.y), ImVec2(sz.x, gHUDMax.y), shade);
            // bottom
            fg->AddRectFilled(ImVec2(0, gHUDMax.y), ImVec2(sz.x, sz.y), shade);
        }

        if(gTrain.done && gCelebrationVisible)
        {
                // Centered result window with very large font
                ImVec2 winSz(720, 460);
                ImVec2 sz = ImGui::GetIO().DisplaySize;
                ImGui::SetNextWindowPos(ImVec2((sz.x-winSz.x)*0.5f, (sz.y-winSz.y)*0.5f), ImGuiCond_Always);
                ImGui::SetNextWindowSize(winSz, ImGuiCond_Always);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18,16));
                ImVec4 bg = gTargetReachedVisual ? ImVec4(0.12f,0.55f,0.22f,0.95f) : ImVec4(0.90f,0.55f,0.15f,0.95f);
                ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
                ImGui::Begin("TrainingResult", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoInputs);
                // Title (always "Training Finished") and status line
                ImGui::SetWindowFontScale(3.0f);
                const char* titleTop = "Training Finished";
                ImVec2 ts = ImGui::CalcTextSize(titleTop);
                ImGui::SetCursorPosX((winSz.x - ts.x)*0.5f);
                ImGui::TextUnformatted(titleTop);
                ImGui::SetWindowFontScale(1.4f);
                const char* status = gTrainInterrupted ? "Training Interrupted" : (gTargetReachedVisual ? "Target Reached" : "Target Not Reached");
                ImVec2 st = ImGui::CalcTextSize(status);
                ImGui::SetCursorPosX((winSz.x - st.x)*0.5f);
                ImGui::TextUnformatted(status);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::Spacing(); ImGui::Separator();
                // IQ gain (heuristic)
                float mse0 = (!gTrain.mseHistory.empty()? gTrain.mseHistory.front() : gTrain.lastMSE);
                float improve = std::max(0.0f, mse0 - gTrain.lastMSE);
                float ratio = (mse0>1e-6f) ? (improve / mse0) : 0.0f;
                int iqGain = (int)std::round(std::max(0.0f, ratio) * 100.0f);
                ImGui::Spacing();
                ImGui::SetWindowFontScale(2.0f);
                int currentIQ = 100 + iqGain;
                char iqbuf[64]; std::snprintf(iqbuf, sizeof(iqbuf), "Current IQ: %d", currentIQ);
                ImVec2 iqts = ImGui::CalcTextSize(iqbuf);
                ImGui::SetCursorPosX((winSz.x - iqts.x)*0.5f);
                ImGui::TextUnformatted(iqbuf);
                ImGui::SetWindowFontScale(1.0f);
                ImGui::Spacing(); ImGui::Separator();
                // Summary
                ImGui::Text("Epoch: %d / %d", gTrain.epoch, gTrain.totalEpochs);
                ImGui::Text("Final MSE: %.6f  Target: %s", gTrain.lastMSE, gTrain.targetMSE>0?std::to_string(gTrain.targetMSE).c_str():"-");
                ImGui::Spacing();
                // Final MSE plot with axes and legend
                ImVec2 plotSize(std::max(400.0f, winSz.x - 80.0f), 160.0f);
                ImGui::PlotLines("##final_mse", gTrain.mseHistory.data(), (int)gTrain.mseHistory.size(), 0, nullptr, FLT_MAX, FLT_MAX, plotSize);
                {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    ImVec2 p0 = ImGui::GetItemRectMin();
                    ImVec2 p1 = ImGui::GetItemRectMax();
                    // Axes
                    dl->AddLine(ImVec2(p0.x, p0.y), ImVec2(p0.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
                    dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p1.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
                    // Ticks + labels
                    const int xt = 8, yt = 4;
                    for(int i=0;i<=xt;i++)
                    {
                        float t=(float)i/(float)xt; float x=p0.x + t*(p1.x-p0.x);
                        dl->AddLine(ImVec2(x,p1.y), ImVec2(x,p1.y-4.0f), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
                    }
                    // y-range estimation
                    float ymin=1e9f, ymax=-1e9f; int n=(int)gTrain.mseHistory.size();
                    for(int i=0;i<n;i++){ float v=gTrain.mseHistory[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
                    if(!(ymax>ymin)){ ymin=0.0f; ymax=1.0f; }
                    for(int i=0;i<=yt;i++)
                    {
                        float t=(float)i/(float)yt; float y=p1.y - t*(p1.y-p0.y);
                        dl->AddLine(ImVec2(p0.x,y), ImVec2(p0.x+5.0f,y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
                        float v=ymin + t*(ymax-ymin); char buf[32]; std::snprintf(buf,sizeof(buf),"%.3f", v);
                        dl->AddText(ImVec2(p0.x+6.0f, y-8.0f), ImGui::GetColorU32(ImVec4(0.85f,0.9f,1.0f,0.9f)), buf);
                    }

                    // Target line
                    if(gTrain.targetMSE>0.0f)
                    {
                        float ty = p1.y - (gTrain.targetMSE - ymin) / std::max(1e-6f, (ymax-ymin)) * (p1.y-p0.y);
                        dl->AddLine(ImVec2(p0.x,ty), ImVec2(p1.x,ty), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,0.9f)), 1.5f);
                    }

                    // Legend (inside plot, top-left)
                    ImVec2 lp = ImVec2(p0.x + 6.0f, p0.y + 6.0f);
                    dl->AddRectFilled(ImVec2(lp.x, lp.y), ImVec2(lp.x+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.2f,0.85f,0.7f,1.0f)));
                    dl->AddText(ImVec2(lp.x+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "MSE");
                    if(gTrain.targetMSE>0.0f)
                    {
                        float ox = lp.x + 60.0f;
                        dl->AddRectFilled(ImVec2(ox, lp.y), ImVec2(ox+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,1.0f)));
                        dl->AddText(ImVec2(ox+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "Target");
                    }
                }
                ImGui::End();
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
        }
    }

    /* Stirlitz arrived at a secret meeting at the agreed address, hour, and minute.
       He knocked exactly 286 times, but nobody answered. Stirlitz leaned out and looked at the windowsill.
       There were 7 irons there — a sign that the operation was blown. */

    // Draw result window even when HUD is hidden or training inactive
#ifdef USE_IMGUI
    if(gCelebrationVisible && !(gTrain.active || gShowTrainingWindow))
    {
        ImVec2 winSz(720, 460);
        ImVec2 sz = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2((sz.x-winSz.x)*0.5f, (sz.y-winSz.y)*0.5f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(winSz, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18,16));
        ImVec4 bg = gTargetReachedVisual ? ImVec4(0.12f,0.55f,0.22f,0.95f) : ImVec4(0.90f,0.55f,0.15f,0.95f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
        ImGui::Begin("TrainingResult", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoInputs);
        
        // Title (always "Training Finished") and status line
        ImGui::SetWindowFontScale(3.0f);
        const char* titleTop = "Training Finished";
        ImVec2 ts = ImGui::CalcTextSize(titleTop);
        ImGui::SetCursorPosX((winSz.x - ts.x)*0.5f);
        ImGui::TextUnformatted(titleTop);
        ImGui::SetWindowFontScale(1.4f);
        const char* status = gTrainInterrupted ? "EMERGENCY PUSSY-BUTTON PRESSED!!!" : (gTargetReachedVisual ? "Target Reached" : "Target Not Reached");
        ImVec2 st = ImGui::CalcTextSize(status);
        ImGui::SetCursorPosX((winSz.x - st.x)*0.5f);
        ImGui::TextUnformatted(status);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Spacing(); ImGui::Separator();

        // IQ gain (heuristic)
        float mse0 = (!gTrain.mseHistory.empty()? gTrain.mseHistory.front() : gTrain.lastMSE);
        float improve = std::max(0.0f, mse0 - gTrain.lastMSE);
        float ratio = (mse0>1e-6f) ? (improve / mse0) : 0.0f;
        int iqGain = (int)std::round(std::max(0.0f, ratio) * 100.0f);
        ImGui::Spacing();
        ImGui::SetWindowFontScale(2.0f);
        int currentIQ = 100 + iqGain;
        char iqbuf[64]; std::snprintf(iqbuf, sizeof(iqbuf), "Current IQ: %d", currentIQ);
        ImVec2 iqts = ImGui::CalcTextSize(iqbuf);
        ImGui::SetCursorPosX((winSz.x - iqts.x)*0.5f);
        ImGui::TextUnformatted(iqbuf);
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Spacing(); ImGui::Separator();

        // Summary
        ImGui::Text("Epoch: %d / %d", gTrain.epoch, gTrain.totalEpochs);
        ImGui::Text("Final MSE: %.6f  Target: %s", gTrain.lastMSE, gTrain.targetMSE>0?std::to_string(gTrain.targetMSE).c_str():"-");
        ImGui::Spacing();

        // Final MSE plot with axes and legend
        ImVec2 plotSize(std::max(400.0f, winSz.x - 80.0f), 160.0f);
        ImGui::PlotLines("##final_mse2", gTrain.mseHistory.data(), (int)gTrain.mseHistory.size(), 0, nullptr, FLT_MAX, FLT_MAX, plotSize);
        {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImVec2 p0 = ImGui::GetItemRectMin();
            ImVec2 p1 = ImGui::GetItemRectMax();

            // Axes
            dl->AddLine(ImVec2(p0.x, p0.y), ImVec2(p0.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);
            dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p1.x, p1.y), ImGui::GetColorU32(ImVec4(0.85f,0.85f,0.9f,0.8f)), 1.0f);

            // Ticks + labels
            const int xt = 8, yt = 4;
            for(int i=0;i<=xt;i++)
            {
                float t=(float)i/(float)xt; float x=p0.x + t*(p1.x-p0.x);
                dl->AddLine(ImVec2(x,p1.y), ImVec2(x,p1.y-4.0f), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
            }

            // y-range estimation
            float ymin=1e9f, ymax=-1e9f; int n=(int)gTrain.mseHistory.size();
            for(int i=0;i<n;i++){ float v=gTrain.mseHistory[i]; if(v<ymin) ymin=v; if(v>ymax) ymax=v; }
            if(!(ymax>ymin)){ ymin=0.0f; ymax=1.0f; }
            for(int i=0;i<=yt;i++)
            {
                float t=(float)i/(float)yt; float y=p1.y - t*(p1.y-p0.y);
                dl->AddLine(ImVec2(p0.x,y), ImVec2(p0.x+5.0f,y), ImGui::GetColorU32(ImVec4(0.8f,0.8f,0.9f,0.7f)), 1.0f);
                float v=ymin + t*(ymax-ymin); char buf[32]; std::snprintf(buf,sizeof(buf),"%.3f", v);
                dl->AddText(ImVec2(p0.x+6.0f, y-8.0f), ImGui::GetColorU32(ImVec4(0.85f,0.9f,1.0f,0.9f)), buf);
            }

            // Target line
            if(gTrain.targetMSE>0.0f)
            {
                float ty = p1.y - (gTrain.targetMSE - ymin) / std::max(1e-6f, (ymax-ymin)) * (p1.y-p0.y);
                dl->AddLine(ImVec2(p0.x,ty), ImVec2(p1.x,ty), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,0.9f)), 1.5f);
            }

            // Legend (inside plot, top-left)
            ImVec2 lp = ImVec2(p0.x + 6.0f, p0.y + 6.0f);
            dl->AddRectFilled(ImVec2(lp.x, lp.y), ImVec2(lp.x+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.2f,0.85f,0.7f,1.0f)));
            dl->AddText(ImVec2(lp.x+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "MSE");
            if(gTrain.targetMSE>0.0f)
            {
                float ox = lp.x + 60.0f;
                dl->AddRectFilled(ImVec2(ox, lp.y), ImVec2(ox+10, lp.y+10), ImGui::GetColorU32(ImVec4(0.95f,0.4f,0.2f,1.0f)));
                dl->AddText(ImVec2(ox+14, lp.y-2), ImGui::GetColorU32(ImVec4(0.9f,0.95f,1,1)), "Target");
            }
        }
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }
#endif

    // After layout pass, clear the relayout flag
    if(gForceRelayoutWindows) gForceRelayoutWindows = false;
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif

    if(cfg.vsync) glfwSwapInterval(1); else glfwSwapInterval(0);
    glfwSwapBuffers((GLFWwindow*)window);
    glfwPollEvents();
}

void Engine::shutdown()
{
#ifdef USE_IMGUI
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
#endif
    renderer.destroy();
    turbine.destroy();
    if(window){ glfwDestroyWindow((GLFWwindow*)window); window=nullptr; }
    glfwTerminate();
}

void Engine::onResize(int /*w*/, int /*h*/){}

static InputState gInput;

void Engine::onMouseButton(int button, int action, int mods)
{
    if(button==0) gInput.mouseDownL = (action==1);
    if(button==1) gInput.mouseDownR = (action==1);
    if(button==2) gInput.mouseDownM = (action==1);
    gInput.alt = (mods & 0x0004) != 0;
    gInput.ctrl = (mods & 0x0002) != 0;
    gInput.shift= (mods & 0x0001) != 0;

#ifdef USE_IMGUI
    // Close celebration banner on any click
    if(action==1)
    {
        // celebration handled elsewhere (no-op)
        // We intentionally rely on frame logic to hide it after input via onKey below.
    }
#endif
}
void Engine::onCursorPos(double x, double y)
{
    double dx = x - gInput.mouseX;
    double dy = y - gInput.mouseY;
    gInput.mouseX = x; 
    gInput.mouseY = y;

#ifdef USE_IMGUI
    if(ImGui::GetIO().WantCaptureMouse) return;
#endif
    if(gInput.mouseDownL && !gInput.shift) cam.orbit((float)dx, (float)dy);
    if(gInput.mouseDownM || (gInput.mouseDownL && gInput.shift)) cam.pan((float)dx,(float)dy);
}
void Engine::onScroll(double /*dx*/, double dy)
{
#ifdef USE_IMGUI
    if(ImGui::GetIO().WantCaptureMouse) return;
#endif
    cam.dolly((float)dy);
}

void Engine::onKey(int key, int /*scancode*/, int action, int mods)
{
    bool down = action != 0; if(!down) return;
#ifdef USE_IMGUI
    // The modal handles Escape; do not let it close the application or run editor shortcuts.
    if(gPendingTopology) return;
#endif
    // If result window is visible, allow closing it with Enter/Space/Esc
#ifdef USE_IMGUI
    if(gCelebrationVisible)
    {
        if(key==GLFW_KEY_ESCAPE || key==GLFW_KEY_ENTER || key==GLFW_KEY_KP_ENTER || key==GLFW_KEY_SPACE){
            gCelebrationVisible = false; gTrain.done = false; return;
        }
    }
#endif
    // ESC should interrupt training at any moment
    if(key==GLFW_KEY_ESCAPE)
    {
        if(gTrain.active)
        {
            gTrain.active = false;
            gShowTrainingWindow = false;
            gHUDHasRect = false;
            gTrain.done = true;
            gCelebrationVisible = true;
            gTargetReachedVisual = false;
            gTrainInterrupted = true;
            SaveEngineConf("engine.conf");
            Logf("Training interrupted (ESC) at epoch %d, MSE=%.6f", gTrain.epoch, gTrain.lastMSE);
            return;
        } 
        else 
        {
            // If no result overlay, allow closing app
            if(!gCelebrationVisible){ glfwSetWindowShouldClose((GLFWwindow*)window, 1); }
            return;
        }
    }

#ifdef USE_IMGUI
    // When editing text (e.g., labels) - don't run shortcuts
    ImGuiIO& _io = ImGui::GetIO();
    if(gEditingLabel || _io.WantTextInput || _io.WantCaptureKeyboard) return;
#endif
#ifdef USE_IMGUI
    // Function-key visibility toggles (blocked while editing text)
    if(key==GLFW_KEY_F1){ gShowEditorWindow = !gShowEditorWindow; return; }
    if(key==GLFW_KEY_F2){ gShowNeuralTPanel = !gShowNeuralTPanel; return; }
    if(key==GLFW_KEY_F3){ gShowLogWindow    = !gShowLogWindow; return; }
    if(key==GLFW_KEY_F4){ gShowParamsWindow = !gShowParamsWindow; return; }
    if(key==GLFW_KEY_F5){ gShowHelp = !gShowHelp; return; }
    if(key==GLFW_KEY_F6){ gShowTrainingWindow = !gShowTrainingWindow; return; }
    if(key==GLFW_KEY_H){ gShowHelp = !gShowHelp; return; }

    if(key==GLFW_KEY_LEFT_BRACKET)
    {
        graph.currentLayer = std::max(0, graph.currentLayer-1);
        return;
    }

    if(key==GLFW_KEY_RIGHT_BRACKET)
    {
        graph.currentLayer = std::min((int)graph.layers.size()-1, graph.currentLayer+1);
        return;
    }

    if(key>=GLFW_KEY_1 && key<=GLFW_KEY_9)
    {
        int idx = key - GLFW_KEY_1; // 0-based
        if(idx < (int)graph.layers.size()) graph.currentLayer = idx;
        return;
    }

    if(key==GLFW_KEY_N && (mods & GLFW_MOD_CONTROL))
    {
        if(gTrain.active){ Logf("Stop training before creating a new network."); return; }
        // Preserve the number of input/output neurons, but discard all hidden/GRU
        // layers, their weights and state, and rebuild direct Input -> Output edges.
        const int inputCount = !graph.layers.empty() ? (int)graph.layers.front().nodeIdx.size() : 3;
        const int outputCount = graph.layers.size() >= 2 ? (int)graph.layers.back().nodeIdx.size() : 3;
        NeuralGraph candidate;
        NG_BuildTopology(candidate, std::max(1,inputCount), {}, std::max(1,outputCount));
        NG_RequestTopology(graph,std::move(candidate));
        return;
    }
    if(key==GLFW_KEY_N){ NG_AddNeuron(graph, graph.currentLayer); return; }
    if(key==GLFW_KEY_D){ NG_RemoveLastNeuron(graph, graph.currentLayer); return; }
    if(key==GLFW_KEY_A){ NG_InsertHiddenAfterCurrent(graph); return; }
    if(key==GLFW_KEY_DELETE){ NG_RemoveCurrentHiddenLayer(graph); return; }
    if(key==GLFW_KEY_TAB)
    {
        graph.liveInputs = !graph.liveInputs;
        // Apply to all input neurons
        for(const auto& L : graph.layers)
        {
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx)
            {
                if(idx<0 || idx>=(int)graph.nodes.size()) continue;
                auto& n = graph.nodes[idx];
                n.live = graph.liveInputs;
                if(graph.liveInputs && n.speed<=0.0f){ n.speed = 0.6f; n.phase = 0.0f; }
            }
        }
        return;
    }

    // 'S' handled below with save dialog (opens Save window)
    if(key==GLFW_KEY_X)
    {
        if(gTrain.active) { Logf("Stop training before randomizing weights."); return; }
        NG_RandomizeHidden(graph, 1.0f, 0.5f);
        Logf("Hidden layers randomized.");
        return;
    }
    if(key==GLFW_KEY_M)
    {
        if(gTrain.active) { Logf("Stop training before adding lessons."); return; }
        std::vector<float> in,out;
        CollectIO(graph,in,out, true);
        TrainSample sample;
        sample.in = std::move(in);
        sample.out = std::move(out);
        gDataset.push_back(std::move(sample));
        gLessonsCount=(int)gDataset.size();
        SaveEngineConf("engine.conf");
        Logf("Lesson added. Total=%d", gLessonsCount);
        return;
    }
    if(key==GLFW_KEY_B)
    {
        StartTraining(graph, false);
        return;
    }
#endif
    if(key==GLFW_KEY_R){ cfg.rotate = !cfg.rotate; }
    if(key==GLFW_KEY_C)
    {
        const size_t nextPair = (size_t)((gPairedThemeIndex + 1) % 16);
        ApplyPairedTheme(renderer, nextPair);
        Logf("Paired theme selected: MLP=%s, GRU=%s",
             renderer.theme.name.c_str(), gGRUTheme.name.c_str());
        return;
    }
    if(key==GLFW_KEY_G && (mods & GLFW_MOD_SHIFT) != 0)
    {
        gGRUThemeIndex = (gGRUThemeIndex + 1) % GRUThemes().size();
        gGRUTheme = GRUThemes()[gGRUThemeIndex];
        Logf("GRU theme selected: %s", gGRUTheme.name.c_str());
        return;
    }
    if(key==GLFW_KEY_V){ cfg.vsync = !cfg.vsync; }
    if(key==GLFW_KEY_T){
        if((mods & GLFW_MOD_SHIFT) != 0)
        {
            if(renderer.theme.name=="NeonNoir") renderer.theme = UITheme::DeepSpace();
            else 
            if(renderer.theme.name=="DeepSpace") renderer.theme = UITheme::SolarFlare();
            else 
            if(renderer.theme.name=="SolarFlare") renderer.theme = UITheme::Sunset();
            else 
            if(renderer.theme.name=="Sunset") renderer.theme = UITheme::EmberForge();
            else 
            if(renderer.theme.name=="EmberForge") renderer.theme = UITheme::Cyberpunk();
            else 
            if(renderer.theme.name=="Cyberpunk") renderer.theme = UITheme::Crimson();
            else 
            if(renderer.theme.name=="Crimson") renderer.theme = UITheme::Ruby();
            else 
            /* Ruby or other */ 
            renderer.theme = UITheme::NeonNoir();
            Logf("Theme selected: %s", renderer.theme.name.c_str());
            return;
        } 
        else 
        {
            StartTraining(graph, true);
            return;
        }
    }

    if(key==GLFW_KEY_SPACE)
    {
        // Toggle live feed on selected input neuron
        int selId = graph.selected;
        if(selId != -1)
        {
            for(auto& n : graph.nodes)
            {
                if(n.id==selId && n.isInput)
                {
                    bool was = n.live; n.live = !n.live;
                    if(n.live && !was && n.speed<=0.0f){ n.speed=0.6f; n.phase=0.0f; }
                    break;
                }
            }
        }
        return;
    }

    if(key==GLFW_KEY_S)
    {
#ifdef USE_IMGUI
        std::error_code _ec; std::filesystem::create_directories("saved", _ec);
        // timestamped default name for quick save
        {
            std::time_t t = std::time(nullptr);
            std::tm tm{};
#if defined(_WIN32)
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            char buf[128];
            std::strftime(buf, sizeof(buf), "saved/network_%Y%m%d_%H%M%S.json", &tm);
            std::snprintf(gSaveName, sizeof(gSaveName), "%s", buf);
        }
        gShowSaveMenu = true;
        return;
#else
        if(NG_SaveJson(graph, "network.json"))
        {
            Logf("Network saved: network.json");
            SaveEngineConf("engine.conf");
        }
        return;
#endif
    }

    if(key==GLFW_KEY_HOME)
    {
        // Show all primary windows and re-layout to fit screen
        cfg.showUI = true;
        gShowEditorWindow = true;
        gShowNeuralTPanel = true;
        gShowLogWindow = true;
        gShowParamsWindow = true;
        gForceRelayoutWindows = true;
        return;
    }

    if(key==GLFW_KEY_O)
    {
#ifdef USE_IMGUI
        std::error_code _ec; 
        std::filesystem::create_directories("saved", _ec);
        gShowLoadMenu = true;
        return;
#else
        std::string msg; 
        if(NG_LoadJson(graph, "network.json", &msg))
        {
             Logf("Network loaded: network.json"); 
        } 
        else 
        { 
            Logf("Load failed: %s", msg.c_str()); 
        }
        return;
#endif
    }

    if(key==GLFW_KEY_W)
    {
        gLiveFeedWaveActive = !gLiveFeedWaveActive;
        if(gLiveFeedWaveActive)
        {
            gLiveFeedCurrentInput = 0;
            gLiveFeedInputTimer = 0.0f;
            gLiveFeedWavePhase = 0.0f;
            Logf("Sequential input sinusoids enabled (W)");
        } 
        else 
        {
            Logf("Sequential input sinusoids disabled (W)");
        }
        return;
    }

    if(key==GLFW_KEY_Q)
    {
        gRandomLiveFeedActive = !gRandomLiveFeedActive;
        int inputCount = 0;
        for(const auto& L : graph.layers)
        {
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx)
            {
                if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                auto& n = graph.nodes[idx];
                if(!n.isInput) continue;
                
                if(gRandomLiveFeedActive)
                {
                    n.amp = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
                    n.speed = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;
                    n.live = true;
                    n.phase = 0.0f;
                }
                else
                {
                    n.live = false;
                    n.speed = 0.0f;
                }
                inputCount++;
            }
        }
        Logf("Random live feed %s for %d input neurons",
             gRandomLiveFeedActive ? "applied" : "disabled", inputCount);
        return;
    }

    // Close celebration banner on any key press by resetting the 'done' flag visual
#ifdef USE_IMGUI
    if(gTrain.done)
    {
        // Toggle done off/on to hide yellow bar; log remains.
        gTrain.done = false;
    }
#endif
}

// orbit handled in callbacks
void Engine::processCamera(){}

// main
// ─────────────────────────────────────────────────────────────────────────────
#include <locale.h>
int main()
{
    // Use C locale for consistent parsing and formatting
    setlocale(LC_ALL, "C");
    AppConfig cfg;
    char main_title[128];
    std::snprintf(main_title, sizeof(main_title), "NeuralT %s ( tiktok: @justdieblondie )", kAppVersion);
    cfg.title = main_title;
    Engine e;
    if(!e.init(cfg)) return 1;

    while(!glfwWindowShouldClose((GLFWwindow*)e.window))
    {
        e.update();
        e.render();
    }
    
    e.shutdown();
    return 0;
}
