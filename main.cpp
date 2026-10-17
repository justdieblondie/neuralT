
// main.cpp  - NeuralT - Neural Editor v3.0
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
#include <regex>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────
class Engine;
extern Engine* gEngine;

// ─────────────────────────────────────────────────────────────────────────────
// Small globals for UI
// ─────────────────────────────────────────────────────────────────────────────
#ifdef USE_IMGUI
static bool gShowHelp = false;
static std::string gToastMsg; 
static double gToastT=0.0; 
static bool gShowTrainingWindow=false;
static bool gEditingLabel = false; // block shortcuts while editing labels
// Visibility toggles for windows/panels (start hidden)
static bool gShowEditorWindow = true;      // F1
static bool gShowNeuralXPanel = false;      // F2 (mirrors cfg.showUI)
static bool gShowLogWindow    = false;      // F3
static bool gShowParamsWindow = false;      // F4
static bool gShowLoadMenu     = false;      // 'O' key or button
static bool gForceRelayoutWindows = false;  // 'Home' key re-arrangement
static bool gShowSaveMenu     = false;      // 'S' key or button
static char gSaveName[256]    = "saved/network.json";

// Graph view: zoom (=/-/0, Ctrl/Cmd+wheel) and cinematic camera drift (C)
static constexpr float kDefaultGraphZoom = 1.0f;   // whole network fits the screen
static float gGraphZoom  = kDefaultGraphZoom;
static bool  gAutoPan    = false;
// Presentation mode (P): full-screen borderless graph with only the title on top
static bool  gPresentation = true;

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
    Mesh m; float halfT=T*0.5f;
    for(int i=0;i<=slices;i++){
        float t = (float)i/slices;
        float w = rootW*(1.0f-t) + tipW*t;
        float xL = -w*0.5f, xR =  w*0.5f;
        float y  = 0.0f; float z  = t*L;
        float twist = t * 0.6f; float c=std::cos(twist), s=std::sin(twist);
        auto rot = [&](float x, float yv){ return Vec3{ x*c + 0*yv, yv, x*(-s) }; };
        std::array<Vec3,4> corners = { rot(xL,-halfT), rot(xR,-halfT), rot(xR,halfT), rot(xL,halfT) };
        Vec3 n{0,1,0};
        m.vertices.push_back({{corners[0].x, y+corners[0].y, z+corners[0].z}, n, {0,0}});
        m.vertices.push_back({{corners[1].x, y+corners[1].y, z+corners[1].z}, n, {1,0}});
        m.vertices.push_back({{corners[2].x, y+corners[2].y, z+corners[2].z}, n, {1,1}});
        m.vertices.push_back({{corners[3].x, y+corners[3].y, z+corners[3].z}, n, {0,1}});
        uint32_t base = i*4;
        if(i>0){
            uint32_t b0 = base-4, b1 = base-3, b2 = base-2, b3 = base-1;
            uint32_t c0 = base,   c1 = base+1, c2 = base+2, c3 = base+3;
            m.indices.insert(m.indices.end(), { b0,b1,c1, b0,c1,c0, b1,b2,c2, b1,c2,c1, b2,b3,c3, b2,c3,c2, b3,b0,c0, b3,c0,c3 });
        }
    }
    m.upload(); return m;
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
    (void)innerR; // inner hole not modeled in this simple gear; silence unused warning
    // Simple radial gear: alternating outer radius to form teeth, centered around origin, axis = Y
    Mesh m;
    int segPerTooth = 4; // base tooth shape resolution
    int seg = std::max(8, teeth * segPerTooth);
    float halfW = width*0.5f;
    // Build two rings (bottom/top), then side quads
    std::vector<Vec3> ringBottom, ringTop;
    ringBottom.reserve(seg); ringTop.reserve(seg);

    for(int i=0;i<seg;i++)
    {
        float t = (float)i/(float)seg;
        float ang = t * 2.f*3.1415926535f;
        // tooth shaping: widen outer radius on tooth peaks
        float toothPhase = t * teeth;
        float frac = toothPhase - std::floor(toothPhase);
        float tooth = (frac<0.5f)? 1.0f : 0.0f; // crude square tooth
        float r = outerR + tooth * toothDepth;
        float c=std::cos(ang), s=std::sin(ang);
        ringBottom.push_back({r*c, -halfW, r*s});
        ringTop.push_back({r*c,  halfW, r*s});
    }

    // inner cylinder (hole) not modeled for simplicity; approximate normals
    auto addTri = [&](const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& n)
    {
        uint32_t base = (uint32_t)m.vertices.size();
        m.vertices.push_back({a,n,{0,0}});
        m.vertices.push_back({b,n,{0,0}});
        m.vertices.push_back({c,n,{0,0}});
        m.indices.insert(m.indices.end(), {base, base+1, base+2});
    };

    // side quads between bottom/top rings
    for(int i=0;i<seg;i++)
    {
        int j = (i+1)%seg;
        Vec3 a = ringBottom[i], b = ringBottom[j];
        Vec3 c = ringTop[j],    d = ringTop[i];
        // normal approx outward
        Vec3 edge{b.x-a.x, b.y-a.y, b.z-a.z};
        Vec3 up{0,1,0};
        Vec3 n{ edge.z*up.y - edge.y*up.z, edge.x*up.z - edge.z*up.x, edge.y*up.x - edge.x*up.y };
        float len = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z); if(len>1e-6f){ n.x/=len; n.y/=len; n.z/=len; }
        addTri(a,b,c,n); addTri(a,c,d,n);
    }
    m.upload(); return m;
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

UITheme UITheme::Ignition()
{
    UITheme t{}; t.name="Ignition";
    // Navy -> violet canvas, molten orange feed-forward edges, pale lime endpoints
    t.bg={0.04f,0.05f,0.13f}; t.panel={0.07f,0.07f,0.17f}; t.accent={1.0f,0.62f,0.22f}; t.grid={0.25f,0.30f,0.55f}; t.text={0.95f,0.95f,1.0f}; t.mesh={1.0f,0.55f,0.20f};
    t.nInput={0.93f,1.0f,0.70f}; t.nHidden={1.0f,0.70f,0.40f}; t.nOutput={1.0f,0.74f,0.32f}; t.nRing={1.0f,1.0f,0.60f}; t.nHighlight={0.90f,1.0f,0.50f};
    t.edgeStart={1.0f,0.42f,0.12f}; t.edgeEnd={1.0f,0.86f,0.48f}; t.edgeAlpha=0.85f; t.edgeBase=0.45f; t.edgeScale=0.55f;
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Turbine
// ─────────────────────────────────────────────────────────────────────────────
void TurbineModel::rebuild()
{
    destroy();
    // Scale up for a larger steampunk assembly
    float bodyR = params.hubRadius * 2.4f;
    float bodyH = params.hubRadius * 2.0f;
    float shaftR = params.shaftRadius * 1.4f;
    float shaftH = params.shaftLength * 1.2f;

    // Static vertical shaft
    shaft = Procedural::makeCylinder(shaftR, shaftH, 64, true);

    // Main turbine body (bigger)
    Mesh body = Procedural::makeCylinder(bodyR, bodyH, 72, true);

    // Gears: central ring gear (co-rotates)
    rotorTeeth = 40;
    // Solid-center gear disks (inner radius = 0)
    float gInner = 0.0f;
    float gOuter = bodyR * 1.15f;
    Mesh ringGear = Procedural::makeGear(gInner, gOuter, (gOuter-gInner)*0.35f, rotorTeeth, bodyH*0.35f);

    // Top rotor (propeller) at the top of the vertical shaft
    Mesh topRotor; 
    topRotor.vertices.clear(); 
    topRotor.indices.clear(); 
    topRotor.vao=topRotor.vbo=topRotor.ebo=0;

    Mesh blade = Procedural::makeBlade(params.bladeRootWidth*1.2f, params.bladeTipWidth*1.2f,
                                       std::max(params.bladeLength, bodyR*1.0f), params.bladeThickness*1.2f, 16);

    int topBlades = std::clamp(params.blades, 1, 10);
    // Center rotor exactly on the vertical shaft axis (no lateral offset)
    float rOffset = 0.0f;
    float yTop = shaftH*0.5f + bodyR*0.10f;

    for(int i=0;i<topBlades;i++)
    {
        float ang = (2.f*3.1415926535f * i)/std::max(1, topBlades);
        float c=std::cos(ang), s=std::sin(ang);
        Mesh b = blade;
        for(auto& v : b.vertices)
        {
            float x=v.pos.x, y=v.pos.y, z=v.pos.z;
            float nx =  c*x + s*z;
            float nz = -s*x + c*z;
            if(rOffset != 0.0f) nx += rOffset;
            v.pos = {nx, y + yTop, nz};
        }
        appendMesh(topRotor, b);
    }
    topRotor.upload();

    // Compose rotating assembly: body + ring + side gears (horizontal & vertical appear coupled)
    rotor.destroy(); 
    rotor.vertices.clear(); 
    rotor.indices.clear(); 
    rotor.vao=rotor.vbo=rotor.ebo=0;
    appendMesh(rotor, body);
    appendMesh(rotor, ringGear);
    appendMesh(rotor, topRotor);
    rotor.upload();

    // Outer static gear train on the left to visually mesh with ring
    outerTeeth = 52; 
    outerTeethR = 52;
    float oInner = 0.0f; // solid center
    float oOuter = oInner*1.12f;

    // Ensure reasonable outer radius when inner is 0
    oOuter = std::max(oOuter, gOuter*1.25f);
    outerGear = Procedural::makeGear(oInner, oOuter, (oOuter-oInner)*0.35f, outerTeeth, bodyH*0.30f);
    outerGearOffset = -(gOuter + oOuter*0.9f); // left

    // Right gear (mirror)
    outerGearR = Procedural::makeGear(oInner, oOuter, (oOuter-oInner)*0.35f, outerTeethR, bodyH*0.30f);
    outerGearOffsetR = +(gOuter + oOuter*0.9f); // right

    // Steampunk pipes: multiple verticals + elbows + valves
    pipes.destroy(); pipes.vertices.clear(); pipes.indices.clear(); pipes.vao=pipes.vbo=pipes.ebo=0;
    auto addPipe = [&](float R, float H, float ox, float oy, float oz)
    {
        Mesh p = Procedural::makeCylinder(R, H, 24, true);
        for(auto& v: p.vertices){ v.pos.x += ox; v.pos.y += oy; v.pos.z += oz; }
        appendMesh(pipes, p);
    };

    // Four main risers around body
    float pr = bodyR * 1.8f;
    for(int i=0;i<4;i++)
    {
        float ang = i * (3.1415926535f*0.5f);
        float c=std::cos(ang), s=std::sin(ang);
        addPipe(bodyR*0.08f, bodyH*2.0f, pr*c, -bodyH*0.5f, pr*s);
        // Elbow (horizontal segment)
        addPipe(bodyR*0.08f, bodyH*0.8f, pr*c + (c>0? bodyH*0.4f : -bodyH*0.4f), 0.0f, pr*s);
    }

    // Valves near the body
    for(int i=0;i<6;i++)
    {
        float t = (2.f*3.1415926535f*i)/6.f; float c=std::cos(t), s=std::sin(t);
        Mesh v = Procedural::makeCylinder(bodyR*0.10f, bodyR*0.15f, 20, true);
        for(auto& vx : v.vertices){ vx.pos.x += (bodyR*1.25f)*c; vx.pos.z += (bodyR*1.25f)*s; }
        appendMesh(pipes, v);
    }
    pipes.upload();

    // Steam lines (stylized) near pipe elbows and valves
    std::vector<Vec3> pts; 
    std::vector<uint32_t> inds; 
    pts.reserve(128); 
    inds.reserve(256);

    auto addSteam = [&](Vec3 p, Vec3 dir, float len, int seg)
    {
        uint32_t b = (uint32_t)pts.size();
        for(int i=0;i<=seg;i++)
        {
            float t=(float)i/seg; float w = std::sin(t*3.14159f);
            pts.push_back({ p.x + dir.x*len*t, p.y + dir.y*len*t + 0.05f*w, p.z + dir.z*len*t });
            if(i>0){ inds.push_back(b+i-1); inds.push_back(b+i); }
        }
    };

    addSteam({pr,0,0}, {0,1,0}, bodyH*0.8f, 12);
    addSteam({-pr,0,0}, {0,1,0}, bodyH*0.7f, 12);
    addSteam({0,0,pr}, {0,1,0}, bodyH*0.6f, 12);
    addSteam({0,0,-pr}, {0,1,0}, bodyH*0.9f, 12);
    schematicLines = Procedural::makeLineList(pts, inds);
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
static void NG_RebuildEdges(NeuralGraph& g)
{
    g.edges.clear();
    auto nid = [&](int idx){ return g.nodes[idx].id; };
    for(size_t L=0; L+1<g.layers.size(); ++L)
    {
        for(int a : g.layers[L].nodeIdx)
        {
            for(int b : g.layers[L+1].nodeIdx)
            {
                g.edges.push_back({ nid(a), nid(b), 0.5f });
            }
        }
    }
}

static int NG_NextId(const NeuralGraph& g)
{
    int mx = -1;
    for(const auto& n : g.nodes)
    {
        mx = std::max(mx, n.id);
    }
    return mx + 1;
}
static void NG_AddNeuron(NeuralGraph& g, int layerIdx)
{
    if(layerIdx < 0 || layerIdx >= (int)g.layers.size()) return;
    auto& L = g.layers[layerIdx];
    Neuron n;
    n.id = NG_NextId(g);
    n.isInput = (L.kind == LayerKind::Input);
    n.label = (L.kind==LayerKind::Input? "In" :
               L.kind==LayerKind::Output? "Out" : "H") + std::to_string((int)L.nodeIdx.size()+1);
    g.nodes.push_back(n);
    g.layers[layerIdx].nodeIdx.push_back((int)g.nodes.size()-1);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f, g.lastCanvasH>0?g.lastCanvasH:500.0f);
    NG_RebuildEdges(g);
}

static void NG_RemoveLastNeuron(NeuralGraph& g, int layerIdx)
{
    if(layerIdx < 0 || layerIdx >= (int)g.layers.size()) return;
    auto& L = g.layers[layerIdx];
    if(L.nodeIdx.empty()) return;
    int absIdx = L.nodeIdx.back();
    int removedId = g.nodes[absIdx].id;
    L.nodeIdx.pop_back();

    // Erase neuron from nodes[] and fix indices in all layers
    g.nodes.erase(g.nodes.begin() + absIdx);
    for(auto& Lay : g.layers)
    {
        for(int& idx : Lay.nodeIdx)
        {
            if(idx > absIdx) idx -= 1;
        }
    }

    // Clear selection if we removed selected
    if(g.selected == removedId) g.selected = -1;

    // Rebuild full connections
    NG_RebuildEdges(g);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f, g.lastCanvasH>0?g.lastCanvasH:500.0f);
}

// Insert a new hidden layer to the right of the current hidden layer
static void NG_InsertHiddenAfterCurrent(NeuralGraph& g)
{
    int li = g.currentLayer;
    if(li < 0 || li >= (int)g.layers.size()) return;
    if(g.layers[li].kind != LayerKind::Hidden) return;
    int insertPos = li + 1;
    Layer L; L.kind = LayerKind::Hidden; L.act = Act::ReLU;
    int hiddenCount = 0; for(const auto& Ly : g.layers) if(Ly.kind==LayerKind::Hidden) hiddenCount++;
    L.name = std::string("HL ") + std::to_string(hiddenCount + 1);
    L.nodeIdx.clear(); L.x0=L.y0=L.x1=L.y1=0.0f;

    if(insertPos < 0) 
    {
        insertPos = 0;
    }

    if(insertPos > (int)g.layers.size()) 
    {
        insertPos = (int)g.layers.size();
    }

    g.layers.insert(g.layers.begin() + insertPos, L);
    g.currentLayer = insertPos;
    // Add a single neuron to keep connectivity meaningful and rebuild edges/layout
    NG_AddNeuron(g, g.currentLayer);
}

static bool NG_RemoveCurrentHiddenLayer(NeuralGraph& g)
{
    const int layerIdx=g.currentLayer;
    if(layerIdx<0 || layerIdx>=(int)g.layers.size()) return false;
    if(g.layers[layerIdx].kind!=LayerKind::Hidden)
    {
        Logf("Only hidden layers can be deleted.");
        return false;
    }

    std::vector<int> removedIndices=g.layers[layerIdx].nodeIdx;
    std::vector<int> removedIds;
    removedIds.reserve(removedIndices.size());
    for(int idx:removedIndices)
        if(idx>=0 && idx<(int)g.nodes.size()) removedIds.push_back(g.nodes[idx].id);

    g.layers.erase(g.layers.begin()+layerIdx);
    std::sort(removedIndices.begin(),removedIndices.end(),std::greater<int>());
    for(int removedIdx:removedIndices)
    {
        if(removedIdx<0 || removedIdx>=(int)g.nodes.size()) continue;
        g.nodes.erase(g.nodes.begin()+removedIdx);
        for(Layer& layer:g.layers)
            for(int& idx:layer.nodeIdx) if(idx>removedIdx) --idx;
    }

    if(std::find(removedIds.begin(),removedIds.end(),g.selected)!=removedIds.end()) g.selected=-1;
    g.currentLayer=std::clamp(layerIdx-1,0,std::max(0,(int)g.layers.size()-1));
    NG_RebuildEdges(g);
    g.layoutLayers(g.lastCanvasW>0?g.lastCanvasW:800.0f,
                   g.lastCanvasH>0?g.lastCanvasH:500.0f);
    Logf("Hidden layer deleted; neighbouring layers reconnected.");
    return true;
}

// Build a specific multi-layer topology
static void NG_BuildTopology(NeuralGraph& g, int inN, const std::vector<int>& hidden, int outN)
{
    g.nodes.clear(); g.edges.clear(); g.layers.clear();
    g.selected = -1; std::memset(g.editBuf,0,sizeof(g.editBuf));
    g.currentLayer = 0; g.liveInputs = false;
    int id = 0;

    // Inputs
    {
        Layer L; L.kind = LayerKind::Input; L.name = "Inputs"; L.act = Act::Linear;
        for(int i=0;i<inN;i++){ Neuron n; n.id=id++; n.isInput=true; n.label="In"+std::to_string(i+1); n.speed=0.0f; n.amp=1.0f; g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }

    // Hidden layers
    for(size_t li=0; li<hidden.size(); ++li)
    {
        Layer L; L.kind = LayerKind::Hidden; L.name = "HL "+std::to_string((int)li+1); L.act = Act::ReLU;
        for(int i=0;i<hidden[li]; ++i){ Neuron n; n.id=id++; n.isInput=false; n.label="H"+std::to_string((int)li+1)+"."+std::to_string(i+1); g.nodes.push_back(n); L.nodeIdx.push_back((int)g.nodes.size()-1); }
        g.layers.push_back(std::move(L));
    }

    // Outputs
    {
        Layer L; L.kind = LayerKind::Output; L.name = "Outputs"; L.act = Act::Sigmoid;
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
        case Act::GRU:     return "GRU";
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
        case Act::Tanh:    return Act::GRU;
        case Act::GRU:     return Act::Linear;
    }
    return Act::Linear;
}

// Training params accessors (declared here, defined later)
// Also define dataset model earlier for save/load embedding
struct TrainSample{ std::vector<float> in; std::vector<float> out; std::string desc = "Default"; };
static std::vector<TrainSample> gDataset;
static int gLessonsCount = 0;
static float TrainGetLR();
static float TrainGetL1();
static float TrainGetL2();
static int   TrainGetEpochs();
static float TrainGetTargetMSE();
static void  TrainApplyParams(float lr, float l1, float l2, int epochs, float targetMSE);
static void  GRU_EnsureShape(Layer& L, int nIn);

static void NG_SaveJson(const NeuralGraph& g, const std::string& path)
{
    std::ofstream out(path);
    if(!out) return;
    out << "{\n";
    out << "  \"version\": \"turbinos-graph-1\",\n";

    // Training parameters snapshot
    out << "  \"learning_rate\": " << TrainGetLR() << ",\n";
    out << "  \"l1\": " << TrainGetL1() << ",\n";
    out << "  \"l2\": " << TrainGetL2() << ",\n";
    out << "  \"epochs\": " << TrainGetEpochs() << ",\n";
    out << "  \"target_mse\": " << TrainGetTargetMSE() << ",\n";

    // Theme information
    const UITheme& theme = (gEngine && gEngine->window ? gEngine->renderer.theme : UITheme::Cyberpunk());
    out << "  \"theme\": {\n";
    out << "    \"name\": \"" << theme.name << "\",\n";
    out << "    \"bg\": [" << theme.bg.x << "," << theme.bg.y << "," << theme.bg.z << "],\n";
    out << "    \"panel\": [" << theme.panel.x << "," << theme.panel.y << "," << theme.panel.z << "],\n";
    out << "    \"accent\": [" << theme.accent.x << "," << theme.accent.y << "," << theme.accent.z << "],\n";
    out << "    \"grid\": [" << theme.grid.x << "," << theme.grid.y << "," << theme.grid.z << "],\n";
    out << "    \"text\": [" << theme.text.x << "," << theme.text.y << "," << theme.text.z << "],\n";
    out << "    \"mesh\": [" << theme.mesh.x << "," << theme.mesh.y << "," << theme.mesh.z << "],\n";
    out << "    \"nInput\": [" << theme.nInput.x << "," << theme.nInput.y << "," << theme.nInput.z << "],\n";
    out << "    \"nHidden\": [" << theme.nHidden.x << "," << theme.nHidden.y << "," << theme.nHidden.z << "],\n";
    out << "    \"nOutput\": [" << theme.nOutput.x << "," << theme.nOutput.y << "," << theme.nOutput.z << "],\n";
    out << "    \"nRing\": [" << theme.nRing.x << "," << theme.nRing.y << "," << theme.nRing.z << "],\n";
    out << "    \"nHighlight\": [" << theme.nHighlight.x << "," << theme.nHighlight.y << "," << theme.nHighlight.z << "],\n";
    out << "    \"edgeStart\": [" << theme.edgeStart.x << "," << theme.edgeStart.y << "," << theme.edgeStart.z << "],\n";
    out << "    \"edgeEnd\": [" << theme.edgeEnd.x << "," << theme.edgeEnd.y << "," << theme.edgeEnd.z << "],\n";
    out << "    \"edgeAlpha\": " << theme.edgeAlpha << ",\n";
    out << "    \"edgeBase\": " << theme.edgeBase << ",\n";
    out << "    \"edgeScale\": " << theme.edgeScale << "\n";
    out << "  },\n";
    out << "  \"nodes\": [\n";
    for(size_t i=0;i<g.nodes.size();++i)
    {
        const auto& n = g.nodes[i];
        out << "    {\n";
        out << "      \"id\": " << n.id << ",\n";
        out << "      \"label\": \"" << n.label << "\",\n";
        out << "      \"desc\": \"" << n.desc << "\",\n";
        out << "      \"isInput\": " << (n.isInput? "true":"false") << ",\n";
        out << "      \"value\": " << n.value << ",\n";
        out << "      \"speed\": " << n.speed << ",\n";
        out << "      \"bias\": " << ng_get_bias(n) << ",\n";
        out << "      \"amp\": " << n.amp << ",\n";
        out << "      \"x\": " << n.x << ",\n";
        out << "      \"y\": " << n.y << "\n";
        out << "    }";

        if(i+1<g.nodes.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"layers\": [\n";
    for(size_t li=0; li<g.layers.size(); ++li)
    {
        const auto& L = g.layers[li];
        const char* kind = (L.kind==LayerKind::Input? "Input" : (L.kind==LayerKind::Output? "Output" : "Hidden"));
        out << "    {\n";
        out << "      \"kind\": \"" << kind << "\",\n";
        out << "      \"name\": \"" << L.name << "\",\n";
        out << "      \"activation\": \"" << ActName(L.act) << "\",\n";
        out << "      \"node_ids\": [";

        for(size_t j=0;j<L.nodeIdx.size();++j)
        {
            int idxN = L.nodeIdx[j];
            int id = (idxN>=0 && idxN < (int)g.nodes.size()) ? g.nodes[idxN].id : -1;
            out << id; if(j+1<L.nodeIdx.size()) out << ", ";
        }
        out << "]\n";
        out << "    }";
        if(li+1<g.layers.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"edges\": [\n";
    for(size_t ei=0; ei<g.edges.size(); ++ei)
    {
        const auto& e = g.edges[ei];
        out << "    {\n";
        out << "      \"a\": " << e.a << ",\n";
        out << "      \"b\": " << e.b << ",\n";
        out << "      \"w\": " << e.w << "\n";
        out << "    }";
        if(ei+1<g.edges.size()) out << ",";
        out << "\n";
    }

    out << "  ],\n";
    // Embed lessons dataset
    out << "  \"lessons\": [\n";
    for(size_t i=0;i<gDataset.size();++i)
    {
        const auto& s = gDataset[i];
        out << "    {\n";
        out << "      \"in\": [";
        for(size_t j=0;j<s.in.size();++j)
        { 
            out << s.in[j]; 
            if(j+1<s.in.size()) out << ", "; 
        }

        out << "],\n";
        out << "      \"out\": [";
        for(size_t j=0;j<s.out.size();++j)
        { 
            out << s.out[j]; 
            if(j+1<s.out.size()) out << ", "; 
        }
        out << "],\n";
        out << "      \"desc\": \"" << s.desc << "\"\n";
        out << "    }";
        if(i+1<gDataset.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    // GRU gate parameters (candidate weights are stored as ordinary edges)
    out << "  \"recurrent\": [\n";
    bool firstRec = true;
    for(size_t li=0; li<g.layers.size(); ++li)
    {
        const Layer& L = g.layers[li];
        if(L.act != Act::GRU || L.bz.empty()) continue;
        auto arr = [&](const char* key, const std::vector<float>& v, bool last)
        {
            out << "      \"" << key << "\": [";
            for(size_t j=0;j<v.size();++j){ out << v[j]; if(j+1<v.size()) out << ","; }
            out << "]" << (last ? "\n" : ",\n");
        };
        if(!firstRec) out << ",\n";
        firstRec = false;
        out << "    {\n";
        out << "      \"layer\": " << li << ",\n";
        arr("wz", L.Wz, false); arr("wr", L.Wr, false);
        arr("uz", L.Uz, false); arr("ur", L.Ur, false); arr("uh", L.Uh, false);
        arr("bz", L.bz, false); arr("br", L.br, true);
        out << "    }";
    }
    out << "\n  ]\n";
    out << "}\n";
}

// Very simple loader matching NG_SaveJson format
static bool NG_LoadJson(NeuralGraph& g, const std::string& path, std::string* msg)
{
    std::ifstream in(path);
    if(!in){ if(msg) *msg = "Missing file: " + path; return false; }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Parse training parameters (if present)
    try{
        auto parseFloatC = [&](const char* key, float defv)
        {
            try{
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+eE0-9\\.]+)");
                std::smatch m; if(std::regex_search(s,m,rx))
                {
                    std::stringstream ss(m[1].str());
                    ss.imbue(std::locale::classic());
                    float v=defv; ss >> v; return v;
                }
            }catch(...){}
            return defv;
        };

        auto parseIntC = [&](const char* key, int defv)
        {
            try{
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+0-9]+)");
                std::smatch m; if(std::regex_search(s,m,rx))
                {
                    std::stringstream ss(m[1].str());
                    ss.imbue(std::locale::classic());
                    int v=defv; ss >> v; return v;
                }
            }catch(...){}
            return defv;
        };

        float _lr = parseFloatC("learning_rate", TrainGetLR());
        float _l1 = parseFloatC("l1", TrainGetL1());
        float _l2 = parseFloatC("l2", TrainGetL2());
        float _tm = parseFloatC("target_mse", TrainGetTargetMSE());
        int   _ep = parseIntC("epochs", TrainGetEpochs());
        TrainApplyParams(_lr, _l1, _l2, _ep, _tm);
    }catch(...){}

    auto find_section = [&](const char* key)->std::string
    {
        size_t ks = s.find(std::string("\"")+key+"\"");
        if(ks==std::string::npos) return {};
        size_t lb = s.find('[', ks);
        if(lb==std::string::npos) return {};
        int depth=0; size_t i=lb;
        for(; i<s.size(); ++i)
        {
            char c=s[i];
            if(c=='[') depth++;
            else if(c==']'){ depth--; if(depth==0){ ++i; break; } }
        }
        return s.substr(lb, i-lb);
    };

    std::string nodesSec = find_section("nodes");
    std::string layersSec = find_section("layers");
    std::string edgesSec = find_section("edges");
    std::string lessonsSec = find_section("lessons");
    if(nodesSec.empty() || layersSec.empty()){ if(msg) *msg="Bad file (sections missing)."; return false; }

    g.nodes.clear(); g.layers.clear(); g.edges.clear();

    // Try regex with desc, fall back to legacy regex without desc
    std::regex rxNodeDesc(R"(\{[^{}]*\"id\"\s*:\s*(-?\d+)[^{}]*\"label\"\s*:\s*\"([^\"]*)\"[^{}]*\"desc\"\s*:\s*\"([^\"]*)\"[^{}]*\"isInput\"\s*:\s*(true|false)[^{}]*\"value\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"speed\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"bias\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"x\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"y\"\s*:\s*([-+eE0-9\.]+)[^{}]*\})");
    std::regex rxNodeLegacy(R"(\{[^{}]*\"id\"\s*:\s*(-?\d+)[^{}]*\"label\"\s*:\s*\"([^\"]*)\"[^{}]*\"isInput\"\s*:\s*(true|false)[^{}]*\"value\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"speed\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"bias\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"x\"\s*:\s*([-+eE0-9\.]+)[^{}]*\"y\"\s*:\s*([-+eE0-9\.]+)[^{}]*\})");
    auto it = std::sregex_iterator(nodesSec.begin(), nodesSec.end(), rxNodeDesc);
    std::unordered_map<int, int> id2idx;
    int idx=0;
    if(it != std::sregex_iterator())
    {
        for(; it!=std::sregex_iterator(); ++it)
        {
            auto m=*it; Neuron n;
            n.id = std::atoi(m[1].str().c_str());
            n.label = m[2].str();
            n.desc = m[3].str();
            n.isInput = (m[4].str()=="true");
            n.value = std::atof(m[5].str().c_str());
            n.speed = std::atof(m[6].str().c_str());
            n.bias = std::atof(m[7].str().c_str());
            n.x = std::atof(m[8].str().c_str());
            n.y = std::atof(m[9].str().c_str());
            {
                const std::string blk = m.str();
                const size_t ka = blk.find("\"amp\"");
                if(ka != std::string::npos){ size_t c = blk.find(':', ka); if(c != std::string::npos) n.amp = (float)std::atof(blk.c_str()+c+1); }
            }
            g.nodes.push_back(n);
            id2idx[n.id] = idx++;
        }
    } 
    else 
    {
        auto it2 = std::sregex_iterator(nodesSec.begin(), nodesSec.end(), rxNodeLegacy);
        for(; it2!=std::sregex_iterator(); ++it2)
        {
            auto m=*it2; Neuron n;
            n.id = std::atoi(m[1].str().c_str());
            n.label = m[2].str();
            n.isInput = (m[3].str()=="true");
            n.value = std::atof(m[4].str().c_str());
            n.speed = std::atof(m[5].str().c_str());
            n.bias = std::atof(m[6].str().c_str());
            n.x = std::atof(m[7].str().c_str());
            n.y = std::atof(m[8].str().c_str());
            g.nodes.push_back(n);
            id2idx[n.id] = idx++;
        }
    }
    if(g.nodes.empty()){ if(msg) *msg="No nodes found."; return false; }

    auto strToAct = [](const std::string& s)->Act
    {
        if(s=="ReLU") return Act::ReLU;
        if(s=="Tanh") return Act::Tanh;
        if(s=="Sigmoid") return Act::Sigmoid;
        if(s=="GRU") return Act::GRU;
        return Act::Linear;
    };

    std::regex rxLayer(R"(\{\s*\"kind\"\s*:\s*\"(Input|Hidden|Output)\"\s*,\s*\"name\"\s*:\s*\"([^\"]*)\"\s*,\s*\"activation\"\s*:\s*\"([^\"]*)\"\s*,\s*\"node_ids\"\s*:\s*\[([^\]]*)\]\s*\})");
    auto itL = std::sregex_iterator(layersSec.begin(), layersSec.end(), rxLayer);
    for(; itL!=std::sregex_iterator(); ++itL)
    {
        auto m=*itL;
        Layer L;
        std::string k = m[1].str();
        L.kind = (k=="Input")? LayerKind::Input : (k=="Output")? LayerKind::Output : LayerKind::Hidden;
        L.name = m[2].str();
        L.act = strToAct(m[3].str());
        L.nodeIdx.clear();
        std::string ids = m[4].str();
        std::stringstream ss(ids);
        std::string tok;
        while(std::getline(ss, tok, ','))
        {
            int id = std::atoi(tok.c_str());
            auto itid = id2idx.find(id);
            if(itid!=id2idx.end()) L.nodeIdx.push_back(itid->second);
        }
        g.layers.push_back(std::move(L));
    }
    if(g.layers.empty()){ if(msg) *msg="No layers found."; return false; }

    // Load edges if present; otherwise, rebuild full connections
    if(!edgesSec.empty())
    {
        std::regex rxEdge(R"(\{\s*\"a\"\s*:\s*(-?\d+)\s*,\s*\"b\"\s*:\s*(-?\d+)\s*,\s*\"w\"\s*:\s*([-+eE0-9\.]+)\s*\})");
        auto itE = std::sregex_iterator(edgesSec.begin(), edgesSec.end(), rxEdge);
        for(; itE!=std::sregex_iterator(); ++itE)
        {
            auto m=*itE;
            Edge e; e.a = std::atoi(m[1].str().c_str()); e.b = std::atoi(m[2].str().c_str()); e.w = std::atof(m[3].str().c_str());
            g.edges.push_back(e);
        }
        if(g.edges.empty()) NG_RebuildEdges(g);
    } else {
        NG_RebuildEdges(g);
    }
    // Load lessons if present
    if(!lessonsSec.empty())
    {
        gDataset.clear();
        std::regex rxLesson(R"(\{[^{}]*\"in\"\s*:\s*\[([^\]]*)\][^{}]*\"out\"\s*:\s*\[([^\]]*)\][^{}]*\})");
        auto itL = std::sregex_iterator(lessonsSec.begin(), lessonsSec.end(), rxLesson);
        auto parseVec = [&](const std::string& arr){
            std::vector<float> v; std::stringstream ss(arr); std::string tok; while(std::getline(ss,tok,',')){ try{ v.push_back(std::stof(tok)); }catch(...){} } return v; };
        for(; itL!=std::sregex_iterator(); ++itL){
            auto m=*itL; TrainSample s; s.in = parseVec(m[1].str()); s.out = parseVec(m[2].str());
            std::regex rxDesc(R"(\"desc\"\s*:\s*\"([^\"]*)\")"); std::smatch md; std::string block = m.str();
            if(std::regex_search(block, md, rxDesc)) s.desc = md[1].str(); else s.desc = "Default";
            gDataset.push_back(std::move(s));
        }
        gLessonsCount = (int)gDataset.size();
    }
    
    // Load theme if present
    try{
        std::regex rxTheme(R"(\"theme\"\s*:\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\})");
        std::smatch themeMatch;
        if(std::regex_search(s, themeMatch, rxTheme))
        {
            std::string themeSection = themeMatch[1].str();
            
            // Parse theme name
            std::regex rxName(R"(\"name\"\s*:\s*\"([^\"]*)\")");
            std::smatch nameMatch;
            std::string themeName = "Cyberpunk"; // default
            if(std::regex_search(themeSection, nameMatch, rxName))
            {
                themeName = nameMatch[1].str();
            }
            
            // Helper function to parse Vec3 arrays
            auto parseVec3 = [&](const std::string& key) -> Vec3 
            {
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*\\[\\s*([-+eE0-9\\.]+)\\s*,\\s*([-+eE0-9\\.]+)\\s*,\\s*([-+eE0-9\\.]+)\\s*\\]");
                std::smatch m;
                if(std::regex_search(themeSection, m, rx))
                {
                    try
                    {
                        return {std::stof(m[1].str()), std::stof(m[2].str()), std::stof(m[3].str())};
                    }
                    catch(...){}
                }
                return {0.0f, 0.0f, 0.0f};
            };
            
            // Helper function to parse float values
            auto parseFloat = [&](const std::string& key, float defVal) -> float 
            {
                std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+eE0-9\\.]+)");
                std::smatch m;
                if(std::regex_search(themeSection, m, rx))
                {
                    try
                    {
                        return std::stof(m[1].str());
                    }
                    catch(...){}
                }
                return defVal;
            };
            
            // Create theme from loaded data
            UITheme loadedTheme;
            loadedTheme.name = themeName;
            loadedTheme.bg = parseVec3("bg");
            loadedTheme.panel = parseVec3("panel");
            loadedTheme.accent = parseVec3("accent");
            loadedTheme.grid = parseVec3("grid");
            loadedTheme.text = parseVec3("text");
            loadedTheme.mesh = parseVec3("mesh");
            loadedTheme.nInput = parseVec3("nInput");
            loadedTheme.nHidden = parseVec3("nHidden");
            loadedTheme.nOutput = parseVec3("nOutput");
            loadedTheme.nRing = parseVec3("nRing");
            loadedTheme.nHighlight = parseVec3("nHighlight");
            loadedTheme.edgeStart = parseVec3("edgeStart");
            loadedTheme.edgeEnd = parseVec3("edgeEnd");
            loadedTheme.edgeAlpha = parseFloat("edgeAlpha", 0.85f);
            loadedTheme.edgeBase = parseFloat("edgeBase", 0.6f);
            loadedTheme.edgeScale = parseFloat("edgeScale", 0.8f);
            
            // Apply theme to renderer if engine exists
            if(gEngine && gEngine->window)
            {
                gEngine->renderer.theme = loadedTheme;
                Logf("Theme loaded: %s", themeName.c_str());
            }
        }
    }
    catch(...)
    {
        // If theme loading fails, continue with default theme
        Logf("Theme loading failed, using default theme");
    }

    // GRU gate parameters. Parsed by hand: the arrays are long enough to make
    // std::regex recurse very deeply.
    {
        const size_t rs = s.find("\"recurrent\"");
        size_t pos = rs;
        while(pos != std::string::npos)
        {
            pos = s.find("\"layer\"", pos);
            if(pos == std::string::npos) break;
            const size_t colon = s.find(':', pos);
            if(colon == std::string::npos) break;
            const int li = std::atoi(s.c_str() + colon + 1);
            const size_t blockEnd = s.find('}', colon);
            auto readArr = [&](const char* key)->std::vector<float>
            {
                std::vector<float> v;
                const size_t k = s.find(std::string("\"")+key+"\"", colon);
                if(k == std::string::npos || k > blockEnd) return v;
                size_t lb = s.find('[', k), rb = s.find(']', k);
                if(lb == std::string::npos || rb == std::string::npos) return v;
                const char* c = s.c_str() + lb + 1;
                const char* end = s.c_str() + rb;
                while(c < end)
                {
                    char* next = nullptr;
                    float f = std::strtof(c, &next);
                    if(next == c){ ++c; continue; }
                    v.push_back(f); c = next;
                }
                return v;
            };
            if(li > 0 && li < (int)g.layers.size() && g.layers[li].act == Act::GRU)
            {
                Layer& L = g.layers[li];
                L.Wz = readArr("wz"); L.Wr = readArr("wr");
                L.Uz = readArr("uz"); L.Ur = readArr("ur"); L.Uh = readArr("uh");
                L.bz = readArr("bz"); L.br = readArr("br");
                GRU_EnsureShape(L, (int)g.layers[li-1].nodeIdx.size()); // re-init if inconsistent
            }
            pos = (blockEnd == std::string::npos) ? blockEnd : blockEnd + 1;
        }
    }
    for(auto& n : g.nodes)
    {
        n.h = 0.0f;
        if(n.isInput) n.phase = ng_frand(0.0f, 6.2831853f); // de-synchronise live sensor waves
    }

    // After loading, compute a fresh layout to scale to current canvas
    float W = g.lastCanvasW>0? g.lastCanvasW : 800.0f;
    float H = g.lastCanvasH>0? g.lastCanvasH : 500.0f;
    g.layoutLayers(W, H);
    for(auto& n : g.nodes){ n.vx=0.0f; n.vy=0.0f; }
    return true;
}

// ===================== Training dataset & Engine config =====================
static std::unordered_map<int,float> gManualOutputs;
static bool gUseManualOverrides = true;

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

// Accessors to avoid forward-dep issues
static float TrainGetLR(){ return gTrain.lr; }
static float TrainGetL1(){ return gTrain.l1; }
static float TrainGetL2(){ return gTrain.l2; }
static int   TrainGetEpochs(){ return gTrain.totalEpochs; }
static float TrainGetTargetMSE(){ return gTrain.targetMSE; }
static void  TrainApplyParams(float lr, float l1, float l2, int epochs, float targetMSE)
{
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
static void SaveEngineConf(const char* path)
{
    std::ofstream f(path);
    if(!f) return;
    f << "{\n";
    f << "  \"engine_version\": \"" << kAppVersion << ",\n";

    f << "  \"learning_rate\": " << gTrain.lr << ",\n";
    f << "  \"l1\": " << gTrain.l1 << ",\n";
    f << "  \"l2\": " << gTrain.l2 << ",\n";
    f << "  \"epochs\": " << gTrain.totalEpochs << "\n";

    // Camera
    f << ",\n  \"cam_yaw\": " << gEngine->cam.yaw;
    f << ",\n  \"cam_pitch\": " << gEngine->cam.pitch;
    f << ",\n  \"cam_distance\": " << gEngine->cam.distance;
    f << ",\n  \"cam_tx\": " << gEngine->cam.target.x;
    f << ",\n  \"cam_ty\": " << gEngine->cam.target.y;
    f << ",\n  \"cam_tz\": " << gEngine->cam.target.z;
    f << "}\n";
}

static float parseNumber(const std::string& s, const char* key, float defv)
{
    try
    {
        std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+eE0-9\\.]+)");
        std::smatch m;
        if(std::regex_search(s,m,rx)){ return std::stof(m[1].str()); }
    }
    catch(...){}
    return defv;
}
static int parseInt(const std::string& s, const char* key, int defv)
{
    try
    {
        std::regex rx(std::string("\"")+key+"\"\\s*:\\s*([-+0-9]+)");
        std::smatch m;
        if(std::regex_search(s,m,rx)){ return std::stoi(m[1].str()); }
    }
    catch(...){}
    return defv;
}

static void LoadEngineConf(const char* path)
{
    std::ifstream f(path);
    if(!f)
    {
        SaveEngineConf(path);
        return;
    }

    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    gTrain.lr = parseNumber(all, "learning_rate", gTrain.lr);
    gTrain.l1 = parseNumber(all, "l1", gTrain.l1);
    gTrain.l2 = parseNumber(all, "l2", gTrain.l2);
    gTrain.targetMSE = parseNumber(all, "target_mse", gTrain.targetMSE);
    gTrain.totalEpochs = parseInt(all, "epochs", gTrain.totalEpochs);

    // lessons count is derived from dataset file; do not read from engine.conf anymore
    // Camera (apply immediately if engine exists)
    float cyaw = parseNumber(all, "cam_yaw", gConf.cam_yaw);
    float cpitch = parseNumber(all, "cam_pitch", gConf.cam_pitch);
    float cdist = parseNumber(all, "cam_distance", gConf.cam_distance);
    float ctx = parseNumber(all, "cam_tx", gConf.cam_tx);
    float cty = parseNumber(all, "cam_ty", gConf.cam_ty);
    float ctz = parseNumber(all, "cam_tz", gConf.cam_tz);
    gConf.cam_yaw=cyaw; gConf.cam_pitch=cpitch; gConf.cam_distance=cdist;
    gConf.cam_tx=ctx; gConf.cam_ty=cty; gConf.cam_tz=ctz;
    if(gEngine)
    {
        gEngine->cam.yaw = cyaw;
        gEngine->cam.pitch = cpitch;
        gEngine->cam.distance = cdist;
        gEngine->cam.target = {ctx, cty, ctz};
    }
}

// Deprecated: lessons are embedded in the saved network file; 

static void CollectIO(const NeuralGraph& g, std::vector<float>& in, std::vector<float>& out)
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
            for(int idx : L.nodeIdx) out.push_back(g.nodes[idx].value);
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
        case Act::Tanh:
        case Act::GRU:     return std::tanh(x); // GRU candidate
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
        case Act::Tanh:
        case Act::GRU:     return 1.0f - y*y;
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

static bool IsGRULayer(const NeuralGraph& g, size_t li)
{
    return li > 0 && li < g.layers.size() && g.layers[li].kind == LayerKind::Hidden && g.layers[li].act == Act::GRU;
}

// (Re)initialise the gate matrices whenever the layer or its predecessor changed size.
static void GRU_EnsureShape(Layer& L, int nIn)
{
    const int n = (int)L.nodeIdx.size();
    if((int)L.bz.size() == n && (int)L.br.size() == n &&
       (int)L.Wz.size() == n*nIn && (int)L.Wr.size() == n*nIn &&
       (int)L.Uz.size() == n*n && (int)L.Ur.size() == n*n && (int)L.Uh.size() == n*n) return;
    const float aIn  = std::sqrt(1.0f / (float)std::max(1, nIn));
    const float aRec = std::sqrt(1.0f / (float)std::max(1, n));
    auto fill = [](std::vector<float>& v, int count, float a){ v.resize(count); for(float& w : v) w = ng_frand(-a, a); };
    fill(L.Wz, n*nIn, aIn); fill(L.Wr, n*nIn, aIn);
    fill(L.Uz, n*n, aRec);  fill(L.Ur, n*n, aRec); fill(L.Uh, n*n, aRec);
    // Negative update-gate bias: the memory starts out "sticky" and smooths its input.
    L.bz.assign(n, -1.0f);
    L.br.assign(n, 0.0f);
}

static float Sigm(float x){ return 1.0f/(1.0f+std::exp(-x)); }

static float CandidateInput(const NeuralGraph& g, const GraphCache& cache, const Neuron& n)
{
    float sum = ng_get_bias(n);
    auto it = cache.inEdges.find(n.id);
    if(it != cache.inEdges.end())
    {
        for(int ei : it->second)
        {
            const auto& e = g.edges[ei];
            auto ita = cache.id2idx.find(e.a);
            if(ita!=cache.id2idx.end()) sum += e.w * g.nodes[ita->second].value;
        }
    }
    return sum;
}

// GRU step for one layer:
//   z = sigm(Wz x + Uz h + bz)          update gate
//   r = sigm(Wr x + Ur h + br)          reset gate
//   c = tanh(Wedges x + b + Uh (r*h))   candidate (input weights are the drawn edges)
//   h' = (1-z) h + z c
// The new state is written to n.value; it is committed to n.h only when advance=true,
// so repeated redraw passes do not run the memory forward.
static void GRU_Forward(NeuralGraph& g, const GraphCache& cache, size_t li, bool advance)
{
    Layer& L = g.layers[li];
    const Layer& P = g.layers[li-1];
    const int n = (int)L.nodeIdx.size(), nIn = (int)P.nodeIdx.size();
    GRU_EnsureShape(L, nIn);
    std::vector<float> x(nIn), hp(n), rh(n);
    for(int k=0;k<nIn;++k) x[k] = g.nodes[P.nodeIdx[k]].value;
    for(int i=0;i<n;++i) hp[i] = g.nodes[L.nodeIdx[i]].h;
    for(int i=0;i<n;++i)
    {
        float az = L.bz[i], ar = L.br[i];
        for(int k=0;k<nIn;++k){ az += L.Wz[i*nIn+k]*x[k]; ar += L.Wr[i*nIn+k]*x[k]; }
        for(int j=0;j<n;++j){ az += L.Uz[i*n+j]*hp[j]; ar += L.Ur[i*n+j]*hp[j]; }
        Neuron& nn = g.nodes[L.nodeIdx[i]];
        nn.gz = Sigm(az); nn.gr = Sigm(ar);
        rh[i] = nn.gr * hp[i];
    }
    for(int i=0;i<n;++i)
    {
        Neuron& nn = g.nodes[L.nodeIdx[i]];
        float a = CandidateInput(g, cache, nn);
        for(int j=0;j<n;++j) a += L.Uh[i*n+j]*rh[j];
        nn.gc = std::tanh(a);
        nn.hPrev = hp[i];
        nn.value = (1.0f-nn.gz)*hp[i] + nn.gz*nn.gc;
        if(advance) nn.h = nn.value;
    }
}

static void GRU_ResetState(NeuralGraph& g)
{
    for(size_t li=0; li<g.layers.size(); ++li)
        if(IsGRULayer(g, li))
            for(int idx : g.layers[li].nodeIdx){ g.nodes[idx].h = 0.0f; g.nodes[idx].value = 0.0f; }
}

static bool HasGRU(const NeuralGraph& g)
{
    for(size_t li=0; li<g.layers.size(); ++li) if(IsGRULayer(g, li)) return true;
    return false;
}

static void ForwardPassCached(NeuralGraph& g, const GraphCache& cache, bool advance)
{
    for(size_t li=0; li<g.layers.size(); ++li)
    {
        const Layer& L = g.layers[li];
        if(L.kind == LayerKind::Input) continue;
        if(IsGRULayer(g, li)){ GRU_Forward(g, cache, li, advance); continue; }
        for(int idx : L.nodeIdx)
        {
            Neuron& n = g.nodes[idx];
            float v_ = 0.0f;
            bool usedOverride = false;
            if(L.kind == LayerKind::Output && gUseManualOverrides && !gTrain.active && !g.liveInputs)
            {
                auto itov = gManualOutputs.find(n.id);
                if(itov != gManualOutputs.end()){
                    v_ = itov->second; usedOverride = true;
                }
            }

            if(!usedOverride) v_ = ActApply(L.act, CandidateInput(g, cache, n));
            if(L.kind == LayerKind::Output){ if(v_ < 0.0f) v_ = 0.0f; else if(v_ > 1.0f) v_ = 1.0f; }
            n.value = v_;
        }
    }
}

static void ForwardPass(NeuralGraph& g, bool advance=false)
{
    GraphCache cache = BuildCache(g);
    ForwardPassCached(g, cache, advance);
}

static float sgn(float x){ return (x>0)-(x<0); }

// Each lesson is a snapshot, so GRU memories start from zero for every lesson and
// are run for kGRUSettleSteps on the same input; the last step is backpropagated
// (one-step truncated BPTT, treating the previous state as constant).
static constexpr int kGRUSettleSteps = 3;

static float TrainOneEpoch(NeuralGraph& g, float lr, float l1, float l2)
{
    if(gDataset.empty()) return 0.0f;
    float mse = 0.0f;

    GraphCache cache = BuildCache(g);
    const bool hasGRU = HasGRU(g);

    for(const auto& s : gDataset)
    {
        size_t pi=0;
        for(const auto& L : g.layers)
        {
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx)
            {
                g.nodes[idx].value = (pi < s.in.size() ? s.in[pi++] : 0.0f);
            }
        }
        if(hasGRU)
        {
            GRU_ResetState(g);
            for(int step=0; step<kGRUSettleSteps-1; ++step) ForwardPassCached(g, cache, true);
        }
        ForwardPassCached(g, cache, false);

        std::vector<int> outNodeIdx;
        Act outAct = Act::Linear;
        for(const auto& L : g.layers){
            if(L.kind == LayerKind::Output)
            {
                outAct = L.act;
                for(int idx : L.nodeIdx) outNodeIdx.push_back(idx);
            }
        }

        std::unordered_map<int,float> delta; // by node id
        for(size_t i=0; i<outNodeIdx.size(); ++i)
        {
            int idx = outNodeIdx[i];
            Neuron& n = g.nodes[idx];
            float y = n.value;
            float t = (i < s.out.size() ? s.out[i] : 0.0f);
            float err = y - t;
            mse += err*err;
            float d = err * ActDeriv(outAct, y);
            delta[n.id] = d;
        }

        // dLoss/dValue reaching a node through GRU gate weights (not via drawn edges)
        std::unordered_map<int,float> gateGrad;

        // Hidden layers backprop (right-to-left; skip last layer already handled)
        for(int Li=(int)g.layers.size()-2; Li>=0; --Li)
        {
            auto& L = g.layers[Li];
            if(L.kind == LayerKind::Input) continue;

            auto downstream = [&](const Neuron& n)
            {
                float sum = 0.0f;
                auto it = cache.outEdges.find(n.id);
                if(it != cache.outEdges.end())
                {
                    for(int ei : it->second)
                    {
                        const auto& e = g.edges[ei];
                        auto it2 = delta.find(e.b);
                        if(it2 != delta.end()) sum += e.w * it2->second;
                    }
                }
                auto ig = gateGrad.find(n.id);
                if(ig != gateGrad.end()) sum += ig->second;
                return sum;
            };

            if(IsGRULayer(g, (size_t)Li))
            {
                const Layer& P = g.layers[Li-1];
                const int n = (int)L.nodeIdx.size(), nIn = (int)P.nodeIdx.size();
                std::vector<float> x(nIn), ah(n), az(n), ar(n);
                for(int k=0;k<nIn;++k) x[k] = g.nodes[P.nodeIdx[k]].value;
                for(int i=0;i<n;++i)
                {
                    const Neuron& nn = g.nodes[L.nodeIdx[i]];
                    const float dh = downstream(nn);
                    ah[i] = dh * nn.gz * (1.0f - nn.gc*nn.gc);
                    az[i] = dh * (nn.gc - nn.hPrev) * nn.gz * (1.0f - nn.gz);
                }
                for(int j=0;j<n;++j)
                {
                    const Neuron& nj = g.nodes[L.nodeIdx[j]];
                    float drh = 0.0f;
                    for(int i=0;i<n;++i) drh += L.Uh[i*n+j]*ah[i];
                    ar[j] = drh * nj.hPrev * nj.gr * (1.0f - nj.gr);
                }
                // gradient into the previous layer through the gates
                for(int k=0;k<nIn;++k)
                {
                    float gsum = 0.0f;
                    for(int i=0;i<n;++i) gsum += L.Wz[i*nIn+k]*az[i] + L.Wr[i*nIn+k]*ar[i];
                    gateGrad[g.nodes[P.nodeIdx[k]].id] += gsum;
                }
                // gate parameter updates
                for(int i=0;i<n;++i)
                {
                    for(int j=0;j<n;++j)
                    {
                        const Neuron& nj = g.nodes[L.nodeIdx[j]];
                        float& uh = L.Uh[i*n+j]; uh -= lr * (ah[i]*nj.gr*nj.hPrev + l2*uh + l1*sgn(uh));
                        float& uz = L.Uz[i*n+j]; uz -= lr * (az[i]*nj.hPrev + l2*uz + l1*sgn(uz));
                        float& ur = L.Ur[i*n+j]; ur -= lr * (ar[i]*nj.hPrev + l2*ur + l1*sgn(ur));
                    }
                    for(int k=0;k<nIn;++k)
                    {
                        float& wz = L.Wz[i*nIn+k]; wz -= lr * (az[i]*x[k] + l2*wz + l1*sgn(wz));
                        float& wr = L.Wr[i*nIn+k]; wr -= lr * (ar[i]*x[k] + l2*wr + l1*sgn(wr));
                    }
                    L.bz[i] -= lr * az[i];
                    L.br[i] -= lr * ar[i];
                    delta[g.nodes[L.nodeIdx[i]].id] = ah[i]; // drives edge + bias updates below
                }
                continue;
            }

            Act act = L.act;
            for(int idx : L.nodeIdx)
            {
                Neuron& n = g.nodes[idx];
                float d = ActDeriv(act, n.value) * downstream(n);
                delta[n.id] = d;
            }
        }

        // Weights update
        for(auto& e : g.edges)
        {
            float srcv = 0.0f;
            auto ita = cache.id2idx.find(e.a);
            if(ita!=cache.id2idx.end()) srcv = g.nodes[ita->second].value;
            float db = delta[e.b];
            float grad = srcv * db + l2*e.w + l1*sgn(e.w);
            e.w -= lr * grad;
        }

        // Bias update
        for(const auto& L : g.layers)
        {
            if(L.kind == LayerKind::Input) continue;
            for(int idx : L.nodeIdx)
            {
                Neuron& n = g.nodes[idx];
                float db = delta[n.id];
                float b = ng_get_bias(n);
                ng_set_bias(n, b - lr * db);
            }
        }
    }
    mse /= (float)std::max<size_t>(1, gDataset.size());
    if(hasGRU) GRU_ResetState(g); // training leaves per-lesson states behind
    return mse;
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
    (void)kind; // hidden labels sit to the left of their neuron, outputs to the right
    const float labelSpace=4.0f;
    const float safeGap=std::max(16.0f,radius*2.0f+labelSpace);
    const float blockSpan=std::min(y1-y0,safeGap*(float)(count-1));
    const float blockTop=(y0+y1-blockSpan)*0.5f;
    return blockTop+blockSpan*(float)slot/(float)(count-1);
}

void NeuralGraph::buildDemo()
{
    nodes.clear(); edges.clear(); layers.clear();
    selected = -1; std::memset(editBuf,0,sizeof(editBuf));
    currentLayer = 0;
    liveInputs = false;

    int nIn = 3;
    std::array<int,3> nHL = {5, 8, 5};
    int nOut = 3;

    int id = 0;

    // Inputs
    {
        Layer L; 
        L.kind = LayerKind::Input; 
        L.name = "Inputs"; 
        L.act = Act::Linear;
        layers.push_back(L);
        for(int i=0;i<nIn;i++)
        {
            Neuron n; n.id=id++; n.isInput=true; n.label = "In"+std::to_string(i+1); n.speed = 0.0f; n.amp = 1.0f;
            nodes.push_back(n); layers.back().nodeIdx.push_back((int)nodes.size()-1);
        }
    }

    // HL1..HL3
    for(int li=0; li<3; ++li)
    {
        Layer L; 
        L.kind = LayerKind::Hidden; 
        L.name = "HL " + std::to_string(li+1);
        L.act = Act::ReLU;
        for(int i=0;i<nHL[li]; ++i)
        {
            Neuron n; n.id=id++; n.isInput=false; n.label = "H"+std::to_string(li+1)+"."+std::to_string(i+1);
            nodes.push_back(n); L.nodeIdx.push_back((int)nodes.size()-1);
        }
        layers.push_back(std::move(L));
    }

    // Outputs
    {
        Layer L; L.kind = LayerKind::Output; 
        L.name = "Outputs"; 
        L.act = Act::Sigmoid;
        layers.push_back(L);
        for(int i=0;i<nOut;i++)
        {
            Neuron n; n.id=id++; n.isInput=false; n.label = "Out"+std::to_string(i+1);
            nodes.push_back(n); layers.back().nodeIdx.push_back((int)nodes.size()-1);
        }
    }

    NG_RebuildEdges(*this);
    layoutLayers(800.0f, 500.0f);
}

void NeuralGraph::layoutLayers(float W, float H)
{
    const float marginX = 40.f, marginY = 40.f;
    const float colGap  = (W - 2*marginX);
    const size_t visibleColumns = collapseHiddenLayers ? 3u : std::max<size_t>(1, layers.size());
    const float colW    = colGap / (float)visibleColumns;

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
    float rVert = std::max(6.0f, 0.5f*minSpacingY - 3.0f);

    // Horizontal limit: ensure circle fits within column; leave padding
    float rHorizGeneric = std::max(6.0f, 0.5f*colW - 12.0f);

    // Inputs have a mini plot on the left: ensure it stays within canvas
    const float kMiniPlotW = ImGui::GetFontSize()*3.6f;

    float rHorizInput = std::max(6.0f, 0.5f*colW - (10.0f + kMiniPlotW + 6.0f));
    float rHoriz = std::min(rHorizGeneric, rHorizInput);
    float globalR = std::clamp(std::min(rVert, rHoriz), 6.0f, 38.0f);

    // Inputs and outputs are the equally important endpoints of the graph.
    // Do not let dense hidden layers shrink either endpoint to tiny dots.
    const float endpointR=std::clamp(ImGui::GetFontSize()*0.95f,12.0f,26.0f);

    float x = marginX;
    size_t displayColumn = 0;
    for(size_t li=0; li<layers.size(); ++li)
    {
        Layer& L = layers[li];
        if(collapseHiddenLayers && L.kind == LayerKind::Hidden)
        {
            L.x0 = marginX + colW + 6.f; L.x1 = marginX + colW*2.f - 6.f;
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
        x = marginX + colW*(float)displayColumn;
        if(collapseHiddenLayers && L.kind == LayerKind::Input){ ++displayColumn; x = marginX + colW*(float)displayColumn; }
    }
}

void NeuralGraph::tickLiveInputs(float dt)
{
    // Do not clear manual output overrides here; they are used when liveInputs is off
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
    tickLiveInputs(dt);
    
    // Recompute forward pass whenever inputs may have changed; this is the one
    // place per frame where GRU memories advance in time.
    ForwardPass(*this, true);

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
    ImGuiWindowFlags winFlags = ImGuiWindowFlags_NoCollapse;
    if(gPresentation)
    {
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(disp, ImGuiCond_Always);
        winFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    }
    else
    {
        ImGui::SetNextWindowSize(ImVec2(width, height), cond);
        ImGui::SetNextWindowPos(ImVec2(10,10), cond);
    }

    char editorTitle[128];
    std::snprintf(editorTitle, sizeof(editorTitle), "Neural Network (Neurons: %zu)###Neural Editor", nodes.size());
    const bool editorVisible = ImGui::Begin(editorTitle, &open, winFlags);
    if(gPresentation) ImGui::PopStyleVar(3);
    if(editorVisible)
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

        if(gPresentation)
        {
            // Title strip, like the window caption in the original v4.0 build
            char ttl[128];
            std::snprintf(ttl, sizeof(ttl), "NeuralT %s ( tiktok: @justdieblondie )", kAppVersion);
            const float th = ImGui::GetFrameHeightWithSpacing() + 6.0f;
            const ImVec2 ts = ImGui::CalcTextSize(ttl);
            const ImVec2 wp = ImGui::GetCursorScreenPos();
            ImDrawList* tdl = ImGui::GetWindowDrawList();
            tdl->AddRectFilled(wp, ImVec2(wp.x + avail.x, wp.y + th), ImGui::GetColorU32(ImVec4(0.11f,0.12f,0.16f,1.0f)));
            tdl->AddText(ImVec2(wp.x + (avail.x - ts.x)*0.5f, wp.y + (th - ts.y)*0.5f),
                         ImGui::GetColorU32(ImVec4(0.92f,0.94f,1.0f,1.0f)), ttl);
            ImGui::Dummy(ImVec2(avail.x, th));
        }
        else
        {
        ImGui::Checkbox("Live Feed", &showLiveFeed);
        ImGui::SameLine();
        if(ImGui::Button(collapseHiddenLayers ? "Show hidden layers" : "Hide hidden layers"))
        {
            collapseHiddenLayers = !collapseHiddenLayers;
            layoutLayers(lastCanvasW > 0 ? lastCanvasW : 800.0f, lastCanvasH > 0 ? lastCanvasH : 500.0f);
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderFloat("Zoom (=/-)", &gGraphZoom, 1.0f, 4.0f, "%.1fx");
        ImGui::SameLine();
        ImGui::Checkbox("Camera drift (C)", &gAutoPan);
        }

        // The small control row is the only space retained when the feed is
        // collapsed, allowing the graph to fill all remaining editor space.
        const float controlsH = ImGui::GetFrameHeightWithSpacing();
        const float maxFeedH = std::max(80.0f, avail.y - 120.0f);
        const float feedH = showLiveFeed ? std::clamp(preferredLiveFeedH, 80.0f, maxFeedH) : 0.0f;
        // top control row + bottom actions row (separator + buttons)
        float canvasH = gPresentation
            ? avail.y - (ImGui::GetFrameHeightWithSpacing() + 6.0f)
            : avail.y - controlsH - (controlsH + 6.0f) - feedH - (showLiveFeed ? 8.0f : 0.0f);
        if(canvasH < 80.0f) canvasH = 80.0f;
        const ImVec2 viewSize = ImVec2(avail.x, canvasH);

        // The graph lives in a larger virtual canvas inside a scrolling view.
        // Width grows with zoom; height grows more gently so tall columns stay readable.
        gGraphZoom = std::clamp(gGraphZoom, 1.0f, 4.0f);
        const float zoomY = 1.0f + (gGraphZoom - 1.0f) * 0.35f;
        ImVec2 canvasSize = ImVec2(viewSize.x * gGraphZoom, viewSize.y * zoomY);
        ImGuiWindowFlags viewFlags = ImGuiWindowFlags_NoScrollWithMouse;
        if(gGraphZoom > 1.001f) viewFlags |= ImGuiWindowFlags_HorizontalScrollbar;
        else viewFlags |= ImGuiWindowFlags_NoScrollbar;
        ImGui::BeginChild("graph_view", viewSize, ImGuiChildFlags_None, viewFlags);
        ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * (1.0f + (gGraphZoom - 1.0f) * 0.30f));
        {
            // Navigation: wheel scrolls, Shift+wheel pans sideways, Ctrl/Cmd+wheel zooms
            // around the cursor; right/middle drag pans. Any of these stops the drift.
            ImGuiIO& io = ImGui::GetIO();
            const bool viewHovered = ImGui::IsWindowHovered();
            const float sx = ImGui::GetScrollX(), sy = ImGui::GetScrollY();
            if(viewHovered && io.MouseWheel != 0.0f)
            {
                if(io.KeyCtrl || io.KeySuper)
                {
                    const float oldZ = gGraphZoom;
                    gGraphZoom = std::clamp(gGraphZoom * (io.MouseWheel > 0 ? 1.1f : 1.0f/1.1f), 1.0f, 4.0f);
                    const ImVec2 local(io.MousePos.x - ImGui::GetWindowPos().x, io.MousePos.y - ImGui::GetWindowPos().y);
                    const float rx = gGraphZoom / oldZ;
                    const float ry = (1.0f + (gGraphZoom-1.0f)*0.35f) / zoomY;
                    ImGui::SetScrollX((sx + local.x) * rx - local.x);
                    ImGui::SetScrollY((sy + local.y) * ry - local.y);
                }
                else if(io.KeyShift) ImGui::SetScrollX(sx - io.MouseWheel * 60.0f);
                else ImGui::SetScrollY(sy - io.MouseWheel * 60.0f);
                gAutoPan = false;
            }
            if(viewHovered && (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 2.0f) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 2.0f)))
            {
                ImGui::SetScrollX(sx - io.MouseDelta.x);
                ImGui::SetScrollY(sy - io.MouseDelta.y);
                gAutoPan = false;
            }
            if(gAutoPan)
            {
                // Slow handheld-style glide across the layers, with a gentle vertical sway
                const float t = (float)ImGui::GetTime();
                const float maxX = std::max(0.0f, canvasSize.x - viewSize.x);
                const float maxY = std::max(0.0f, canvasSize.y - viewSize.y + ImGui::GetStyle().ScrollbarSize);
                ImGui::SetScrollX(maxX * (0.5f - 0.5f*std::cos(t * 0.11f)));
                ImGui::SetScrollY(maxY * (0.5f - 0.5f*std::cos(t * 0.23f + 1.3f)));
            }
        }

        // Relayout every frame: the zoomed font changes header and label metrics.
        layoutLayers(canvasSize.x, canvasSize.y);
        lastCanvasW = canvasSize.x; lastCanvasH = canvasSize.y;

        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 canvasMax = ImVec2(origin.x + canvasSize.x, origin.y + canvasSize.y);

        const UITheme& _theme = (gEngine ? gEngine->renderer.theme : UITheme::Cyberpunk());

        // CANVAS background: theme bg drifting toward violet in the lower right
        {
            const Vec3& bg = _theme.bg;
            ImU32 cTL = ImGui::GetColorU32(ImVec4(bg.x, bg.y, bg.z, 1.0f));
            ImU32 cTR = ImGui::GetColorU32(ImVec4(bg.x+0.04f, bg.y+0.02f, bg.z+0.10f, 1.0f));
            ImU32 cBR = ImGui::GetColorU32(ImVec4(bg.x+0.14f, bg.y+0.03f, bg.z+0.16f, 1.0f));
            ImU32 cBL = ImGui::GetColorU32(ImVec4(bg.x+0.03f, bg.y+0.04f, bg.z+0.12f, 1.0f));
            dl->AddRectFilledMultiColor(origin, canvasMax, cTL, cTR, cBR, cBL);
            dl->AddRect(origin, canvasMax, ImGui::GetColorU32(ImVec4(0.25f,0.25f,0.3f,1.0f)), 6.0f);
        }

        // layer index of every node (for GRU styling)
        std::vector<int> nodeLayer(nodes.size(), -1);
        for(size_t li=0; li<layers.size(); ++li)
            for(int idx : layers[li].nodeIdx) if(idx>=0 && idx<(int)nodes.size()) nodeLayer[idx]=(int)li;
        auto isGRUNode = [&](int idx){ return idx>=0 && idx<(int)nodes.size() && nodeLayer[idx]>=0 && IsGRULayer(*this, (size_t)nodeLayer[idx]); };

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
            if(collapseHiddenLayers && L.kind == LayerKind::Hidden) continue;
            ImVec2 a(origin.x + L.x0, origin.y + L.y0);
            ImVec2 b(origin.x + L.x1, origin.y + L.y1);
            const bool gru = IsGRULayer(*this, li);
            const Vec3& pc = _theme.panel;
            ImU32 colBg = (int(li)==currentLayer) ?
                ImGui::GetColorU32(ImVec4(pc.x+0.08f,pc.y+0.10f,pc.z+0.14f,0.80f)) :
                ImGui::GetColorU32(ImVec4(pc.x,pc.y,pc.z,0.55f));
            dl->AddRectFilled(a, b, colBg, 8.0f);
            if(gru)
            {
                // Recurrent memory: glowing blue frame + a small "state flows down" marker
                dl->AddRect(a, b, ImGui::GetColorU32(ImVec4(0.25f,0.55f,1.0f,0.25f)), 8.0f, 0, 6.0f);
                dl->AddRect(a, b, ImGui::GetColorU32(ImVec4(0.35f,0.68f,1.0f,1.0f)), 8.0f, 0, 2.5f);
                ImVec2 m(a.x + 12.0f, b.y - 12.0f);
                dl->AddTriangleFilled(ImVec2(m.x-5,m.y-4), ImVec2(m.x+5,m.y-4), ImVec2(m.x,m.y+4),
                                      ImGui::GetColorU32(ImVec4(0.35f,0.68f,1.0f,1.0f)));
            }
            else
                dl->AddRect(a, b, ImGui::GetColorU32(ImVec4(0.35f,0.37f,0.48f,0.85f)), 8.0f, 0, 1.5f);

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
            if(gru) std::snprintf(la, sizeof(la), "Gates: update / reset / candidate");
            else std::snprintf(la, sizeof(la), "Activation: %s  (click)", actName);
            ImVec2 actPos(hp.x, hp.y + lh*2.0f);
            dl->AddText(actPos, ImGui::GetColorU32(ImVec4(0.9f,0.8f,1,1)), la);

            ImVec2 actMin = actPos;
            ImVec2 actMax = ImVec2(b.x - 8.0f, actPos.y + lh);
            if(ImGui::IsMouseHoveringRect(actMin, actMax) &&
               ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            {
                L.act = NextAct(L.act);
                if(L.act == Act::GRU && L.kind != LayerKind::Hidden) L.act = NextAct(L.act);
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
        ImVec2 hiddenHub(origin.x + canvasSize.x*0.5f, origin.y + canvasSize.y*0.54f);
        if(collapseHiddenLayers)
        {
            int hiddenLayerCount=0, hiddenNeuronCount=0;
            for(const Layer& layer : layers) if(layer.kind == LayerKind::Hidden)
            {
                ++hiddenLayerCount; hiddenNeuronCount += (int)layer.nodeIdx.size();
            }

            const float pulse = 0.5f + 0.5f*std::sin((float)ImGui::GetTime()*2.0f);
            dl->AddCircleFilled(hiddenHub, 37.0f + pulse*3.0f, ImGui::GetColorU32(ImVec4(.18f,.75f,.72f,.12f)), 40);
            dl->AddCircleFilled(hiddenHub, 29.0f, ImGui::GetColorU32(ImVec4(.10f,.16f,.22f,1.0f)), 40);
            dl->AddCircle(hiddenHub, 29.0f, ImGui::GetColorU32(ImVec4(.25f,.95f,.85f,1.0f)), 40, 3.0f);
            dl->AddCircle(hiddenHub, 18.0f, ImGui::GetColorU32(ImVec4(.25f,.95f,.85f,.8f)), 6, 2.0f);
            char summary[96]; std::snprintf(summary, sizeof(summary), "%d hidden layers / %d neurons", hiddenLayerCount, hiddenNeuronCount);
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

        const float timeNow = (float)ImGui::GetTime();
        const bool denseGraph = edges.size() > 2500;

        // Signal wave: one batch of particles crosses every layer gap at the same
        // time; when a batch reaches the next layer it continues into the next gap
        // in that gap's colour, so the wave visibly hands off layer to layer.
        constexpr float kWaveGapsPerSec = 0.45f;
        const float wavePos = timeNow * kWaveGapsPerSec;
        const float waveT = wavePos - std::floor(wavePos);   // 0..1 within each gap
        static const ImVec4 kWaveColors[] = {
            ImVec4(1.00f,0.96f,0.86f,1), ImVec4(0.55f,0.95f,1.00f,1), ImVec4(1.00f,0.62f,0.95f,1),
            ImVec4(1.00f,0.88f,0.40f,1), ImVec4(0.60f,1.00f,0.70f,1), ImVec4(0.80f,0.72f,1.00f,1) };
        constexpr int kWaveColorCount = (int)(sizeof(kWaveColors)/sizeof(kWaveColors[0]));
        struct Spark { ImVec2 p0, c1, c2, p3; float act; int gap; };
        std::vector<Spark> sparks;
        sparks.reserve(edges.size()/2);

        // Stable per-edge random numbers in [-1,1] (so each strand keeps its own shape)
        auto edgeHash = [](int a, int b, unsigned salt)->float
        {
            unsigned h = (unsigned)a*73856093u ^ (unsigned)b*19349663u ^ salt*83492791u;
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            return (float)(h & 0xFFFF) / 32767.5f - 1.0f;
        };

        float currentStrength = gStrengthSineEnabled ?
            (150.0f + 150.0f * std::sin(gWindStrengthPhase)) : gWind.strength;
        float currentSpeed = gSpeedSineEnabled ?
            (2.5f + 2.5f * std::sin(gWindSpeedPhase)) : gWind.speed;
        const float windGain = (currentStrength / 150.0f) * (0.25f + 0.75f*gWindInterpolant);

        // position of each node inside its layer (drives the strand families)
        std::vector<int> nodeSlot(nodes.size(), 0);
        for(const Layer& L : layers)
            for(size_t j=0;j<L.nodeIdx.size();++j) if(L.nodeIdx[j]>=0 && L.nodeIdx[j]<(int)nodes.size()) nodeSlot[L.nodeIdx[j]]=(int)j;

        for(size_t ei=0; ei<edges.size(); ++ei)
        {
            auto& e = edges[ei];
            Neuron* A = findById(e.a);
            Neuron* B = findById(e.b);
            if(!A||!B) continue;
            const int ai = int(A - nodes.data()), bi = int(B - nodes.data());
            const bool recurrentEdge = isGRUNode(ai) || isGRUNode(bi);
            if(!IsVisibleInput(*this, ai) || !IsVisibleInput(*this, bi)) continue;
            if(collapseHiddenLayers) continue;
            ImVec2 p1(origin.x+A->x, origin.y+A->y), p2(origin.x+B->x, origin.y+B->y);

            // Colour: theme gradient by weight; memory edges lavender -> magenta
            float t = (e.w + 1.0f) * 0.5f; t = t<0?0: (t>1?1:t);
            auto mixf = [](float a, float b, float tt){ return a + (b-a)*tt; };
            float r = mixf(_theme.edgeStart.x, _theme.edgeEnd.x, t);
            float g = mixf(_theme.edgeStart.y, _theme.edgeEnd.y, t);
            float b = mixf(_theme.edgeStart.z, _theme.edgeEnd.z, t);
            float a = _theme.edgeAlpha;
            if(recurrentEdge)
            {
                float tr = std::clamp(t*0.6f + (isGRUNode(ai) ? 0.4f : 0.0f), 0.0f, 1.0f);
                r = mixf(0.72f, 1.0f, tr); g = mixf(0.48f, 0.42f, tr); b = mixf(1.0f, 0.90f, tr);
                a *= 0.40f;
            }
            else a *= 0.70f;
            // Thin, crisp strands so every line can be followed end to end
            float thick = (_theme.edgeBase + _theme.edgeScale * std::fabs(e.w)) * zoomY;
            if(denseGraph) thick = std::clamp(thick * 0.7f, 0.8f, 1.15f);

            // Shape: every strand has its own bow (direction + depth) and a little
            // S-twist, which weaves the bundle into a lattice; on top of that each
            // strand sways slowly with its own phase.
            ImVec2 d = ImVec2(p2.x - p1.x, p2.y - p1.y);
            const float len = std::sqrt(d.x*d.x + d.y*d.y) + 1e-3f;
            const ImVec2 nrm(-d.y/len, d.x/len);
            const float h1 = edgeHash(e.a, e.b, 1u), h2 = edgeHash(e.a, e.b, 2u), h3 = edgeHash(e.a, e.b, 3u);
            // Strands from neighbouring neurons sweep together in families (smooth in
            // source/target slot), leaving the woven diamond gaps; a little per-strand
            // noise keeps them from looking combed.
            const bool endpointFan = A->isInput || (nodeLayer[bi] >= 0 && layers[nodeLayer[bi]].kind == LayerKind::Output);
            const float family = std::sin(0.55f*(float)nodeSlot[ai] + 0.23f*(float)nodeSlot[bi] + 0.9f*(float)nodeLayer[ai]);
            const float bowMax = endpointFan ? 0.09f : 0.22f;
            const float bow   = len * bowMax * (0.8f*family + 0.2f*h1);
            const float twist = len * (endpointFan ? 0.03f : 0.06f) * h2;
            const float sway  = len * 0.045f * windGain * std::sin(timeNow*(0.6f + 0.5f*std::fabs(h3))*currentSpeed + 6.2831f*h3);
            const float k1 = bow + twist + sway;
            const float k2 = bow - twist + sway*0.6f;
            ImVec2 c1 = ImVec2(p1.x + d.x*0.33f + nrm.x*k1, p1.y + d.y*0.33f + nrm.y*k1);
            ImVec2 c2 = ImVec2(p1.x + d.x*0.66f + nrm.x*k2, p1.y + d.y*0.66f + nrm.y*k2);

            const int segs = denseGraph ? 16 : 0;
            if(!denseGraph)
            {
                // small graphs can afford a soft glow under each strand
                const float pulse = 0.5f + 0.5f*std::sin(timeNow*2.0f + 0.15f*(float)(A->id + B->id));
                dl->AddBezierCubic(p1, c1, c2, p2, ImGui::GetColorU32(ImVec4(r,g,b,0.15f*pulse)), thick*3.0f);
            }
            dl->AddBezierCubic(p1, c1, c2, p2, ImGui::GetColorU32(ImVec4(r,g,b,a)), thick, segs);

            // Wave particles ride on a subset of strands (denser through the memories)
            const int stride = recurrentEdge ? 2 : 4;
            if(((unsigned)(e.a*31 + e.b) % (unsigned)stride) == 0 && nodeLayer[ai] >= 0)
                sparks.push_back({p1, c1, c2, p2, std::fabs(A->value), nodeLayer[ai]});
        }

        // Draw the wave: each dot with a short tail; strength follows the activation
        // of the neuron that sent it (silent neurons send faint signals).
        const int waveIndex = (int)std::floor(wavePos);
        for(const Spark& sp : sparks)
        {
            const ImVec4 wc = kWaveColors[((sp.gap + waveIndex) % kWaveColorCount + kWaveColorCount) % kWaveColorCount];
            const float strength = 0.30f + 0.70f*std::min(1.0f, sp.act);
            for(int k=0;k<4;++k)
            {
                const float tt = waveT - 0.022f*(float)k;
                if(tt < 0.0f) continue;
                const float u = 1.0f - tt;
                ImVec2 q(u*u*u*sp.p0.x + 3*u*u*tt*sp.c1.x + 3*u*tt*tt*sp.c2.x + tt*tt*tt*sp.p3.x,
                         u*u*u*sp.p0.y + 3*u*u*tt*sp.c1.y + 3*u*tt*tt*sp.c2.y + tt*tt*tt*sp.p3.y);
                const float alpha = (0.95f - 0.22f*(float)k) * strength;
                dl->AddCircleFilled(q, (2.2f - 0.4f*(float)k) * zoomY, ImGui::GetColorU32(ImVec4(wc.x,wc.y,wc.z,alpha)), 8);
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
                for(const Layer& layer:layers) if(layer.kind==LayerKind::Hidden &&
                    std::find(layer.nodeIdx.begin(), layer.nodeIdx.end(), visibleIdx)!=layer.nodeIdx.end()){ hiddenNode=true; break; }
                if(hiddenNode) continue;
            }
            ImVec2 p(origin.x+n.x, origin.y+n.y);
            bool isSel = (selected==n.id);

            // compute if this neuron is in the output layer
            bool isOutput = false;
            {
                int idxN = int(&n - &nodes[0]);
                for(const auto& L: layers)
                {
                    if(L.kind != LayerKind::Output) continue;
                    for(int j : L.nodeIdx){ if(j==idxN){ isOutput = true; break; } }
                    if(isOutput) break;
                }
            }

            auto colU = [&](const Vec3& v){ return ImGui::GetColorU32(ImVec4(v.x,v.y,v.z,1.0f)); };
            ImU32 baseFill = n.isInput ? colU(_theme.nInput) : (isOutput ? colU(_theme.nOutput) : colU(_theme.nHidden));
            ImU32 fill = isSel ? ImGui::GetColorU32(ImVec4(0.2f,0.5f,1.0f,1.0f)) : baseFill;
            ImU32 ring = isSel ? colU(_theme.nRing)
                               : ImGui::GetColorU32(ImVec4(0.9f,0.9f,0.95f,0.9f));

            // highlight top output: theme highlight
            if(isOutput && n.id == topOutId)
            {
                fill = colU(_theme.nHighlight);
            }

            if(isGRUNode(visibleIdx))
            {
                // GRU memory cell: diamond, brightness follows |h|
                const float r = n.radius + 1.5f;
                const float lv = std::min(1.0f, std::fabs(n.value));
                ImU32 dfill = isSel ? fill : ImGui::GetColorU32(ImVec4(0.55f+0.40f*lv, 0.62f+0.34f*lv, 0.85f+0.15f*lv, 1.0f));
                ImVec2 q0(p.x, p.y-r), q1(p.x+r, p.y), q2(p.x, p.y+r), q3(p.x-r, p.y);
                dl->AddQuadFilled(q0, q1, q2, q3, dfill);
                dl->AddQuad(q0, q1, q2, q3, ImGui::GetColorU32(ImVec4(0.92f,0.96f,1.0f,0.95f)), 1.5f);
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
                    if(L.kind != LayerKind::Hidden) continue;
                    for(int j : L.nodeIdx){ if(j==idxN){ isHidden = true; break; } }
                    if(isHidden) break;
                }
            }

            // label: hidden BELOW, input on LEFT (leave space for mini-plot), output on RIGHT
            ImVec2 ts = ImGui::CalcTextSize(n.label.c_str());
            ImVec2 tl;
            if(isHidden)
            {
                tl = ImVec2(p.x - n.radius - 4.0f - ts.x, p.y - ts.y*0.5f);
            } 
            else if(isOutput)
            {
                tl = ImVec2(p.x + n.radius + 8.0f, p.y - ts.y*0.5f);
            } 
            else 
            {
                tl = ImVec2(p.x - n.radius - 18.0f - ts.x - ImGui::GetFontSize()*3.6f, p.y - ts.y*0.5f); // input - space for mini-plot on the left
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

            // Manual override icon next to output labels when active
            if(isOutput)
            {
                bool manualActive = false;
                if(gUseManualOverrides && !gTrain.active && !liveInputs)
                {
                    manualActive = (gManualOutputs.find(n.id) != gManualOutputs.end());
                }

                if(manualActive)
                {
                    ImVec2 iconC = ImVec2(tl.x - 10.0f, p.y);
                    ImU32 icoCol = ImGui::GetColorU32(ImVec4(1.0f,0.95f,0.3f,1.0f)); // yellow icon
                    dl->AddCircleFilled(iconC, 4.0f, icoCol, 16);

                    // Tooltip area
                    ImGui::SetCursorScreenPos(ImVec2(iconC.x - 6.0f, iconC.y - 6.0f));
                    ImGui::InvisibleButton((std::string("ovr##")+std::to_string(n.id)).c_str(), ImVec2(12,12));
                    if(ImGui::IsItemHovered()) ImGui::SetTooltip("Manual override active");
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
                const float PW=ImGui::GetFontSize()*3.6f, PH=ImGui::GetFontSize()*1.6f;
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
                    dl->AddLine(prev, cur, plotFg, std::max(1.5f, ImGui::GetFontSize()*0.12f));
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
                ForwardPass(*this);
            } else 
            {
                selected = -1;
                std::memset(editBuf,0,sizeof(editBuf));
            }
        }

        ImGui::PopFont();
        ImGui::EndChild();

        // ====== LIVE FEED (under the graph) ======
        if(showLiveFeed && !gPresentation)
        {
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
                    if(lastSelForVal != sel->id){
                        pendingVal = sel->value;
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
                        ImGui::SliderFloat(u8"Value", &pendingVal, 0.0f, 1.0f, "%.4f");
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
                    if(doApply)
                    {
                        // Commit label
                        sel->label = editBuf;
                        sel->desc = descBuf;
                        // Commit value behavior
                        if(isOutput)
                        {
                            if(pendingVal < 0.0f) pendingVal = 0.0f; else if(pendingVal > 1.0f) pendingVal = 1.0f;
                            sel->value = pendingVal;
                            gManualOutputs[sel->id] = pendingVal;
                            gUseManualOverrides = true;
                        } 
                        else if(sel->isInput)
                        {
                            if(pendingVal < 0.0f) pendingVal = 0.0f; else if(pendingVal > 1.0f) pendingVal = 1.0f;
                            sel->value = pendingVal;
                            sel->speed = std::max(0.0f, pendingSpeed);
                            liveInputs = false;
                        }
                        ForwardPass(*this);
                    }

                    ImGui::PopItemWidth();
                }
            }
            }
            ImGui::EndChild();
        }

        // Local actions panel
        if(!gPresentation)
        {
        ImGui::Separator();
        ImGui::Checkbox("Live inputs (sin)", &liveInputs);
        ImGui::SameLine();
        if(ImGui::Button("Relayout")){ layoutLayers(canvasSize.x, canvasSize.y); }
        ImGui::SameLine();
        if(ImGui::Button("Add neuron (N)")){ NG_AddNeuron(*this, currentLayer); }
        ImGui::SameLine();
        if(ImGui::Button("Remove last (D)")){ NG_RemoveLastNeuron(*this, currentLayer); }
        }
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
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // required for a 3.3 core context on macOS
#endif

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
    turbine.params.current_rpm = cfg.rotate ? turbine.params.rpm : 0.0f;
    turbine.rebuild();

    srand((unsigned int)time(NULL));
    gWindStrengthPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindSpeedPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    gWindJitterPhase = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
    if (cfg.rotate) 
    {
        gWindInterpolant = 1.0f;
    }

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
    {
      bool exists = (std::ifstream("engine.conf").good());
      if(!exists)
      {
          // First run: write defaults
          SaveEngineConf("engine.conf");
      }
    }
#ifdef USE_IMGUI
    {
        std::string msg;
        if(std::ifstream(kStartupNetworkFile).good() && NG_LoadJson(graph, kStartupNetworkFile, &msg))
        {
            gLiveFeedWaveActive = false; // all sensors stream at once
            graph.liveInputs = true;
            Logf("Loaded %s", kStartupNetworkFile);
        }
    }
#endif
    Logf("Engine %s initialized.", kAppVersion);
    return true;
}

void Engine::update()
{
    double t = glfwGetTime(); timer.update(t);

    // Smooth wind transition
    const float kWIND_TRANSITION_SPEED = 1.5f; // seconds to go from 0 to 1
    if (cfg.rotate) 
    {
        gWindInterpolant += (float)timer.dt / kWIND_TRANSITION_SPEED;
        if (gWindInterpolant > 1.0f) gWindInterpolant = 1.0f;
    } 
    else 
    {
        gWindInterpolant -= (float)timer.dt / kWIND_TRANSITION_SPEED;
        if (gWindInterpolant < 0.0f) gWindInterpolant = 0.0f;
    }

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

    if(cfg.rotate)
    {
        float rps = turbine.params.rpm / 60.0f;
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

    Mat4 Mshaft = Mat4::translate({0,0,0});
    renderer.drawMesh(turbine.shaft, VP*Mshaft, Mshaft, renderer.theme.mesh, 0.05f);

    // Draw full rotor as one rotating assembly around Y axis at origin
    Mat4 Mrotor = Mat4::rotateY(rotorAngle);
    renderer.drawMesh(turbine.rotor, VP*Mrotor, Mrotor, renderer.theme.accent, 0.4f);

    // Draw static pipes
    Mat4 Mpipes = Mat4::translate({0,0,0});
    renderer.drawMesh(turbine.pipes, VP*Mpipes, Mpipes, renderer.theme.mesh, 0.2f);

    // Steam lines (stylized) - always visible
    renderer.drawLines(turbine.schematicLines, VP, renderer.theme.accent, 2.0f);

    // Draw meshing outer gears (counter-rotating), translated on X (left and right)
    float ratioL = (turbine.outerTeeth>0)? (turbine.rotorTeeth / (float)turbine.outerTeeth) : 1.0f;
    Mat4 MouterL = Mat4::translate({turbine.outerGearOffset, 0, 0}) * Mat4::rotateY(-rotorAngle*ratioL);
    renderer.drawMesh(turbine.outerGear, VP*MouterL, MouterL, renderer.theme.mesh, 0.6f);
    float ratioR = (turbine.outerTeethR>0)? (turbine.rotorTeeth / (float)turbine.outerTeethR) : 1.0f;
    Mat4 MouterR = Mat4::translate({turbine.outerGearOffsetR, 0, 0}) * Mat4::rotateY(rotorAngle*ratioR);
    renderer.drawMesh(turbine.outerGearR, VP*MouterR, MouterR, renderer.theme.mesh, 0.6f);

    if(cfg.showSchematic)
    {
        renderer.drawLines(turbine.schematicLines, VP, renderer.theme.grid, 1.0f);
    }

#ifdef USE_IMGUI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

    // Auto-layout windows each frame (deterministic positions)
    {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 disp = io.DisplaySize;

        // Left top: Neural Editor (big)
        float neW = std::max(400.0f, disp.x);
        float neH = std::max(300.0f, disp.y);
        if(gShowEditorWindow)
        {
            bool openNe = true;
            graph.drawCanvas(openNe, neW - 20.0f, neH - 20.0f);
        }

        // Top-right: NeuralT control panel
        ImGuiCond layCond = gForceRelayoutWindows ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
        ImGui::SetNextWindowPos(ImVec2(disp.x - 360.0f, 10.0f), layCond);
        ImGui::SetNextWindowSize(ImVec2(350.0f, 270.0f), layCond);
        if(cfg.showUI && gShowNeuralXPanel)
        {
            ImGui::Begin("Steam Punk Uber Crap Turbine", nullptr, ImGuiWindowFlags_NoCollapse);
            char title[128];
            std::snprintf(title, sizeof(title), "NeuralT %s | dt: %.3f ms", kAppVersion, timer.dt*1000.0);
            ImGui::TextUnformatted(title);
            ImGui::Text("Total Neurons: %zu", graph.nodes.size());
            ImGui::Checkbox("VSync (V)", &cfg.vsync);
            ImGui::SameLine(); ImGui::Checkbox("Rotate (R)", &cfg.rotate);
            ImGui::Checkbox("Schematic", &cfg.showSchematic);
            ImGui::SliderFloat("RPM", &turbine.params.rpm, 0.0f, 15000.0f);
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
            static bool randomLiveFeedActive = false;
            if(ImGui::Button(randomLiveFeedActive ? "Disable Random Feed (Q)" : "Enable Random Feed (Q)"))
            {
                randomLiveFeedActive = !randomLiveFeedActive;
                int inputCount = 0;
                for(const auto& L : graph.layers)
                {
                    if(L.kind != LayerKind::Input) continue;
                    for(int idx : L.nodeIdx)
                    {
                        if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                        auto& n = graph.nodes[idx];
                        if(!n.isInput) continue;
                        
                        if(randomLiveFeedActive)
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
                if(randomLiveFeedActive)
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
                randomLiveFeedActive = false;
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
            ImGui::SliderInt("Epochs", &gTrain.totalEpochs, 10, 5000);
            ImGui::SliderFloat("Learning rate", &gTrain.lr, 1e-5f, 1.0f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("L1", &gTrain.l1, 0.0f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("L2", &gTrain.l2, 0.0f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic);
            ImGui::InputFloat("Target MSE (<=0 off)", &gTrain.targetMSE, 0.0f, 0.0f, "%.6f");
            ImGui::Separator();
            // Row 1: Rabbit Example, Help
            if(ImGui::Button("Rabbit Example"))
            {
                // Build 6-10-8-5 network for rabbit scenario
                NG_BuildTopology(graph, 6, {10,8}, 5);
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

                    if(!graph.layers.empty() && graph.layers.front().kind==LayerKind::Input)
                    {
                        auto& L = graph.layers.front();
                        for(size_t i=0;i<L.nodeIdx.size() && i<6;i++)
                        { 
                            graph.nodes[L.nodeIdx[i]].label = std::string(kInputs[i]);
                            graph.nodes[L.nodeIdx[i]].desc = std::string(kInputDescs[i]);
                        }
                    }
                    if(!graph.layers.empty())
                    {
                        auto& L = graph.layers.back();
                        if(L.kind==LayerKind::Output)
                        {
                            for(size_t i=0;i<L.nodeIdx.size() && i<5;i++)
                            { 
                                graph.nodes[L.nodeIdx[i]].label = std::string(kActions[i]);
                                graph.nodes[L.nodeIdx[i]].desc = std::string(kActionDescs[i]);
                            }
                        }
                    }
                }

                // Randomize hidden layers for a fresh start
                NG_RandomizeHidden(graph, 0.8f, 0.3f);

                // Training params
                gTrain.lr = 0.01f;
                gTrain.l1 = 0.0f;
                gTrain.l2 = 0.0001f;
                gTrain.totalEpochs = 3000;
                gTrain.targetMSE = 0.02f;

                // Dataset from specification (6 inputs, 5 outputs one-hot) with descriptions
                gDataset.clear();
                auto make_sample = [&](const std::array<float,6>& in, const std::array<float,5>& out, const std::string& desc)
                {
                    TrainSample s; s.in.assign(in.begin(), in.end()); s.out.assign(out.begin(), out.end()); s.desc = desc; return s;
                };

                gDataset.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.8f, 0.3f, 0.6f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Fox nearby, no food → Flee"));
                gDataset.push_back(make_sample({0.0f, 0.8f, 0.0f, 0.7f, 0.9f, 0.5f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Fox nearby, bush close → Hide"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.9f, 0.0f, 0.2f, 0.1f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Hungry, sees carrot, no threat → Move to food"));
                gDataset.push_back(make_sample({0.8f, 0.0f, 0.9f, 0.0f, 0.1f, 0.2f}, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, "At carrot, hungry, safe → Eat"));
                gDataset.push_back(make_sample({0.1f, 0.0f, 0.0f, 0.0f, 0.2f, 0.9f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Full, no threat → Rest"));
                gDataset.push_back(make_sample({0.7f, 0.8f, 0.6f, 0.7f, 0.2f, 0.3f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Fox and food nearby, low hunger → Flee"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Tired, full, no signals → Rest"));
                gDataset.push_back(make_sample({0.0f, 0.2f, 0.9f, 0.1f, 0.1f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Very hungry, sees carrot, fox far → Move to food"));
                gDataset.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.7f, 0.0f, 0.2f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, "Low energy, smells fox, no shelter → Flee"));
                gDataset.push_back(make_sample({0.6f, 0.0f, 0.9f, 0.0f, 0.1f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Full, sees carrot, no fox → Rest"));
                gDataset.push_back(make_sample({0.0f, 0.9f, 0.0f, 0.6f, 0.5f, 0.4f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Strong fox smell, no shelter → Hide"));
                gDataset.push_back(make_sample({0.0f, 0.2f, 0.0f, 0.4f, 0.9f, 0.6f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}, "Fox far, bush close → Hide"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.6f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "Slightly hungry, safe → Rest"));
                gDataset.push_back(make_sample({0.7f, 0.1f, 0.0f, 0.2f, 0.0f, 0.3f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Fox very far, smells carrot → Move to food"));
                gDataset.push_back(make_sample({0.5f, 0.2f, 0.4f, 0.1f, 0.0f, 0.9f}, {0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, "Full, slight fox smell → Eat (safe, enough energy)"));
                gDataset.push_back(make_sample({0.6f, 0.0f, 0.7f, 0.0f, 0.1f, 0.7f}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, "High energy, senses carrot and safety → Rest"));
                gDataset.push_back(make_sample({0.0f, 0.0f, 0.7f, 0.0f, 0.8f, 0.3f}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, "Carrot visible behind bush, no fox → Move to food"));
                gLessonsCount = (int)gDataset.size();
                SaveEngineConf("engine.conf");
                Logf("Rabbit Example initialized: topology 6-10-8-5, lessons=%d", gLessonsCount);
            }

            ImGui::SameLine();
            if(ImGui::Button("Help (H)")) gShowHelp = !gShowHelp;
            // Row 2: Start, Train To Target, Save Config
            if(ImGui::Button("Start"))
            {
                if(!gDataset.empty())
                {
                    // Break symmetry if weights look uniform (e.g., freshly rebuilt)
                    bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
                    for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
                    if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
                    gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime = nowSeconds(); gTrain.avgEpochMs=0.0;
                    Logf("Training started: epochs=%d, lr=%.6f, L1=%.6f, L2=%.6f, targetMSE=%.6f",
                         gTrain.totalEpochs, gTrain.lr, gTrain.l1, gTrain.l2, gTrain.targetMSE);
                }
                else
                {
                    Logf("Cannot start training: dataset is empty.");
                }
            }

            ImGui::SameLine();
            if(ImGui::Button("Train To Target (T)"))
            {
                if(gDataset.empty())
                {
                    Logf("Cannot train to target: dataset is empty. idiot.");
                } 
                else if(gTrain.targetMSE <= 0.0f)
                {
                    Logf("Cannot train to target: set Target MSE > 0 first. dushebag.");
                } 
                else 
                {
                    bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
                    for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
                    if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
                    gTrain.totalEpochs = std::max(gTrain.totalEpochs, 200000); // give plenty of room
                    gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime = nowSeconds(); gTrain.avgEpochMs=0.0;
                    Logf("Train-to-target started: targetMSE=%.6f, epochs=%d, lr=%.6f", gTrain.targetMSE, gTrain.totalEpochs, gTrain.lr);
                }
            }

            ImGui::SameLine();
            if(ImGui::Button("Save Config")){ SaveEngineConf("engine.conf"); Logf("Config saved: engine.conf"); }
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
        ImGui::Text("Dataset: %d lessons", gLessonsCount);

        // Row 3: Add Lesson, Randomize Hidden, Delete All Lessons
        if(ImGui::Button("Add Lesson (M)"))
        {
            std::vector<float> in,out; CollectIO(graph,in,out);
            gDataset.push_back({in,out});
            gLessonsCount=(int)gDataset.size();
            SaveEngineConf("engine.conf");
            Logf("Lesson added. Total=%d", gLessonsCount);
        }

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
        if(ImGui::BeginTable("lessons_table", 5, ImGuiTableFlags_Borders|ImGuiTableFlags_RowBg|ImGuiTableFlags_Resizable|ImGuiTableFlags_SizingStretchSame))
        {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 40.0f);
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

                ImGui::TableSetColumnIndex(1); 
                { 
                    std::string ins = vecToStr(s.in); 
                    ImGui::TextUnformatted(ins.c_str()); 
                }

                ImGui::TableSetColumnIndex(2); 
                {
                    std::string outs= vecToStr(s.out); 
                    ImGui::TextUnformatted(outs.c_str()); 
                }

                ImGui::TableSetColumnIndex(3);
                ImGui::PushID(i);
                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##desc", lessonDescBuf[i].data(), lessonDescBuf[i].size());
                ImGui::PopID();
                ImGui::TableSetColumnIndex(4);
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
            }
            ImGui::EndTable();
        }
        if(pendingDelete >= 0 && pendingDelete < (int)gDataset.size())
        {
            gDataset.erase(gDataset.begin() + pendingDelete);
            gLessonsCount = (int)gDataset.size();
            if((int)lessonDescBuf.size() > pendingDelete) lessonDescBuf.erase(lessonDescBuf.begin() + pendingDelete);
            Logf("Lesson %d deleted.", pendingDelete);
        }

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
                NG_SaveJson(graph, path);
                SaveEngineConf("engine.conf");
                gToastMsg = std::string(u8"Shitty Network saved: ") + path;
                gToastT = glfwGetTime();
                Logf("Total-shit Network saved: %s", path.c_str());
                gShowSaveMenu = false;
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
        ImGui::BulletText("N                    - add neuron to current layer");
        ImGui::BulletText("A                    - add a hidden layer to the right of current");
        ImGui::BulletText("D                    - remove last neuron from current layer");
        ImGui::BulletText("Delete               - remove current hidden layer and reconnect neighbours");
        ImGui::BulletText("S                    - save network (open Save dialog)");
        ImGui::BulletText("O                    - load network (open Load dialog)");
        ImGui::BulletText("M                    - add a lesson from current IN/OUT values");
        ImGui::BulletText("B                    - start training on collected lessons");
        ImGui::BulletText("T                    - train to target (fine-tune until Target MSE)");
        ImGui::BulletText("G                    - toggle GRU memory on the current hidden layer");
        ImGui::BulletText("P                    - presentation mode (full-screen graph) / editor mode");
        ImGui::BulletText("= / - / 0            - zoom in / out / fit whole network");
        ImGui::BulletText("C                    - cinematic camera drift on/off");
        ImGui::BulletText("Ctrl/Cmd+wheel       - zoom at cursor; right-drag or wheel to pan");
        ImGui::BulletText("W                    - toggle live feed wave effect");
        ImGui::BulletText("Q                    - random live feed for all input neurons");
        ImGui::BulletText("TAB                  - toggle live feed on all inputs");
        ImGui::BulletText("V / R / G / Shift+T  - VSync / rotate / schematic / theme");
ImGui::Separator();
ImGui::Text("Creating example networks");
ImGui::BulletText("Use buttons like 'Rabbit Example' in Training parameters for presets");
ImGui::BulletText("Build manually: add layers (A), neurons (N), edit in Neural Editor");
ImGui::BulletText("Set labels and descriptions in neuron editor (select neuron)");
ImGui::BulletText("Add lessons (M) with current IN/OUT values");
ImGui::BulletText("Train (B or T) after setting parameters");
        ImGui::BulletText("Mouse                - LMB orbit, MMB or Shift+LMB pan, wheel zoom");
        ImGui::Separator();
        ImGui::Text("Training parameters");
        ImGui::BulletText("Epochs - number of full passes over the dataset");
        ImGui::BulletText("Learning rate - step size (typical start ~0.001)");
        ImGui::BulletText("L1 - L1 regularization (|w|), encourages sparsity");
        ImGui::BulletText("L2 - L2 regularization (w^2), stabilizes training");
        ImGui::BulletText("Target MSE - early stop threshold; <= 0 disables");
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
            gUseManualOverrides = false;
            float mse = TrainOneEpoch(graph, gTrain.lr, gTrain.l1, gTrain.l2);
            gUseManualOverrides = true;
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
                NG_SaveJson(graph, "network_trained.json");
                Logf("Training finished: epoch=%d, MSE=%.6f%s.",
                     gTrain.epoch, mse, targetReached?" (target eliminated.)": "");
                gCelebrationVisible = true;
                gTrainDoneTime = nowSeconds();
                gTargetReachedVisual = targetReached;
                gTrainInterrupted = false;
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

    // Dev aid: NEURALT_SNAPSHOT=out.ppm writes one frame (after ~3 s) and quits.
    if(const char* snap = std::getenv("NEURALT_SNAPSHOT"))
    {
        static int frame = 0;
        const char* atEnv = std::getenv("NEURALT_SNAPSHOT_FRAME");
        if(++frame == (atEnv ? std::atoi(atEnv) : 180))
        {
            std::vector<unsigned char> px((size_t)w*h*3);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            if(FILE* f = std::fopen(snap, "wb"))
            {
                std::fprintf(f, "P6\n%d %d\n255\n", w, h);
                for(int y=h-1; y>=0; --y) std::fwrite(px.data() + (size_t)y*w*3, 1, (size_t)w*3, f);
                std::fclose(f);
            }
            glfwSetWindowShouldClose((GLFWwindow*)window, GLFW_TRUE);
        }
    }

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
    if(key==GLFW_KEY_F2){ gShowNeuralXPanel = !gShowNeuralXPanel; return; }
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

    if(key==GLFW_KEY_N){ NG_AddNeuron(graph, graph.currentLayer); return; }
    if(key==GLFW_KEY_D){ NG_RemoveLastNeuron(graph, graph.currentLayer); return; }
    if(key==GLFW_KEY_A){ NG_InsertHiddenAfterCurrent(graph); return; }
    if(key==GLFW_KEY_DELETE){ NG_RemoveCurrentHiddenLayer(graph); return; }
    if(key==GLFW_KEY_TAB)
    {
        static bool s_allInputsLive = true; // toggled by TAB
        s_allInputsLive = !s_allInputsLive;
        graph.liveInputs = s_allInputsLive;
        // Apply to all input neurons
        for(const auto& L : graph.layers)
        {
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx)
            {
                if(idx<0 || idx>=(int)graph.nodes.size()) continue;
                auto& n = graph.nodes[idx];
                n.live = s_allInputsLive;
                if(s_allInputsLive && n.speed<=0.0f){ n.speed = 0.6f; n.phase = 0.0f; }
            }
        }
        return;
    }

    // 'S' handled below with save dialog (opens Save window)
    if(key==GLFW_KEY_X){ NG_RandomizeHidden(graph, 1.0f, 0.5f); Logf("Hidden layers randomized."); return; }
    if(key==GLFW_KEY_M){ std::vector<float> in,out; CollectIO(graph,in,out); gDataset.push_back({in,out}); gLessonsCount=(int)gDataset.size(); SaveEngineConf("engine.conf"); Logf("Lesson added. Total=%d", gLessonsCount); return; }
    if(key==GLFW_KEY_B)
    {
        gShowTrainingWindow = true;
        if(!gDataset.empty())
        {
            // Break symmetry if all weights are equal (freshly built graphs)
            bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
            for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
            if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
            gTrain.active=true; gTrain.done=false; gTrain.epoch=0; gTrain.mseHistory.clear(); gTrain.startTime=nowSeconds(); gTrain.avgEpochMs=0.0;
            Logf("Training started (hotkey B): epochs=%d, lr=%.6f, L1=%.6f, L2=%.6f, targetMSE=%.6f",
                 gTrain.totalEpochs, gTrain.lr, gTrain.l1, gTrain.l2, gTrain.targetMSE);
        } 
        else 
        {
            Logf("Cannot start training: dataset is empty.");
        }
        return;
    }
#endif
    if(key==GLFW_KEY_R){ cfg.rotate = !cfg.rotate; }
    if(key==GLFW_KEY_C){ gAutoPan = !gAutoPan; Logf("Camera drift %s", gAutoPan ? "on" : "off"); }
    if(key==GLFW_KEY_EQUAL || key==GLFW_KEY_KP_ADD){ gGraphZoom = std::min(4.0f, gGraphZoom*1.15f); }
    if(key==GLFW_KEY_MINUS || key==GLFW_KEY_KP_SUBTRACT){ gGraphZoom = std::max(1.0f, gGraphZoom/1.15f); }
    if(key==GLFW_KEY_0){ gGraphZoom = 1.0f; }
    if(key==GLFW_KEY_P){ gPresentation = !gPresentation; gForceRelayoutWindows = true; }
    if(key==GLFW_KEY_G)
    {
        if(graph.currentLayer >= 0 && graph.currentLayer < (int)graph.layers.size() &&
           graph.layers[graph.currentLayer].kind == LayerKind::Hidden)
        {
            Layer& L = graph.layers[graph.currentLayer];
            L.act = (L.act == Act::GRU) ? Act::ReLU : Act::GRU;
            for(int idx : L.nodeIdx) graph.nodes[idx].h = 0.0f;
            Logf("Layer '%s' -> %s", L.name.c_str(), ActName(L.act));
        }
        else Logf("Select a hidden layer to toggle GRU memory (G).");
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
            if(renderer.theme.name=="Ruby") renderer.theme = UITheme::Ignition();
            else 
            /* Ignition or other */ 
            renderer.theme = UITheme::NeonNoir();
            Logf("Theme selected: %s", renderer.theme.name.c_str());
            return;
        } 
        else 
        {
            // Train to target (fine-tune) via keyboard 't'
            if(!gDataset.empty() && gTrain.targetMSE > 0.0f)
            {
                bool uniform = true; float ref = (!graph.edges.empty()? graph.edges.front().w : 0.0f);
                for(const auto& e_ : graph.edges){ if(std::fabs(e_.w - ref) > 1e-6f){ uniform=false; break; } }
                if(uniform) NG_RandomizeHidden(graph, 1.0f, 0.5f);
                gTrain.totalEpochs = std::max(gTrain.totalEpochs, 200000);
                gTrain.active=true; 
                gTrain.done=false; 
                gTrain.epoch=0; 
                gTrain.mseHistory.clear(); 
                gTrain.startTime = nowSeconds(); 
                gTrain.avgEpochMs=0.0;
                Logf("Train-to-kill-target started (hotkey T): targetMSE=%.6f, epochs=%d, lr=%.6f", gTrain.targetMSE, gTrain.totalEpochs, gTrain.lr);
            } 
            else 
            {
                if(gDataset.empty()) Logf("Cannot train to target: dataset is empty you idiot.");
                else Logf("Cannot kill the target: set Target MSE > 0 first. Pussy.");
            }
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
        NG_SaveJson(graph, "network.json"); 
        SaveEngineConf("engine.conf"); 
        Logf("Network saved: network.json"); 
        return;
#endif
    }

    if(key==GLFW_KEY_HOME)
    {
        // Show all primary windows and re-layout to fit screen
        cfg.showUI = true;
        gShowEditorWindow = true;
        gShowNeuralXPanel = true;
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
        // Random live feed for all input neurons
        int inputCount = 0;
        for(const auto& L : graph.layers)
        {
            if(L.kind != LayerKind::Input) continue;
            for(int idx : L.nodeIdx)
            {
                if(idx < 0 || idx >= (int)graph.nodes.size()) continue;
                auto& n = graph.nodes[idx];
                if(!n.isInput) continue;
                
                // Set random amplitude (0.1 to 1.0)
                n.amp = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;

                // Set random speed (0.1 to 1.0 Hz)
                n.speed = 0.1f + (rand() / (float)RAND_MAX) * 0.9f;

                // Enable live feed
                n.live = true;

                // Reset phase for immediate effect
                n.phase = 0.0f;
                inputCount++;
            }
        }
        Logf("Random live feed applied to %d input neurons", inputCount);
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
int main(int argc, char** argv)
{
    // Use C locale for consistent parsing and formatting
    setlocale(LC_ALL, "C");
    AppConfig cfg;
    char main_title[128];
    std::snprintf(main_title, sizeof(main_title), "NeuralT %s ( tiktok: @justdieblondie )", kAppVersion);
    cfg.title = main_title;
    Engine e;
    if(!e.init(cfg)) return 1;

#ifdef USE_IMGUI
    // neuralT --pretrain <network.json> [maxEpochs]
    // Trains the file's lessons until its target MSE (or maxEpochs) and saves it back.
    if(argc >= 3 && std::string(argv[1]) == "--pretrain")
    {
        std::string path = argv[2], msg;
        if(!NG_LoadJson(e.graph, path, &msg)){ std::fprintf(stderr, "load failed: %s\n", msg.c_str()); return 1; }
        const int maxEpochs = (argc >= 4) ? std::atoi(argv[3]) : gTrain.totalEpochs;
        gUseManualOverrides = false;
        gTrain.active = true;
        float mse = 0.0f;
        int ep = 0;
        for(; ep < maxEpochs; ++ep)
        {
            mse = TrainOneEpoch(e.graph, gTrain.lr, gTrain.l1, gTrain.l2);
            if(ep % 50 == 0){ std::printf("epoch %d  mse %.6f\n", ep, mse); std::fflush(stdout); glfwPollEvents(); }
            if(gTrain.targetMSE > 0.0f && mse <= gTrain.targetMSE) break;
        }
        gTrain.active = false;
        std::printf("done: epoch %d  mse %.6f\n", ep, mse);
        NG_SaveJson(e.graph, path);
        e.shutdown();
        return 0;
    }
#else
    (void)argc; (void)argv;
#endif

    while(!glfwWindowShouldClose((GLFWwindow*)e.window))
    {
        e.update();
        e.render();
    }
    
    e.shutdown();
    return 0;
}
