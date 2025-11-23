#include <GLFW/glfw3.h>
#include <GL/gl.h>             // Opcjonalnie, nie zawsze wymagane
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

#define uchar unsigned char

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <cuda_gl_interop.h>

#include "cuda_buffer.cuh"
#include "random_fill.cuh"
#include "particles_t.h"
#include "constants.h"
#include "gpu_particles_handle_t.h"

#define TITLE "ElectronsAndProtons_CUDA"

// Makro wypisujące informację o błędzie i zwracające go
#define CUDA_CHECK_RET(call)                                      \
    ([&]() -> cudaError_t {                                       \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n",           \
                cudaGetErrorString(err), __FILE__, __LINE__);     \
        }                                                         \
        return err;                                               \
    })()

// Makro wypisujące informację o błędzie i skaczące do etykiety w przypadku błędu
#define CUDA_CHECK_GOTO(call, label)                            \
    do {                                                        \
    cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                cudaGetErrorString(err), __FILE__, __LINE__);   \
            goto label;                                         \
        }                                                       \
    } while (0)                                                 \

void value_to_color(const float value, const float min, const float max, unsigned char* r, unsigned char* g, unsigned char* b);
int setupWindowAndOpenGLContext(GLFWwindow * *window, GLuint * texture, const int width, const int height);
void parseInputArgs(const int argc, char* const* argv, int* width, int* height, int* count);
void calculateElectricField(float* field, const int width, const int height, const particles_t * particles);
void mapFieldValuesToPixels(const float* field, const int width, const int height, uint8_t * pixels);
void drawParticles(const particles_t * particles, uint8_t * pixels, const int width, const int height);
void updateAccelerations(particles_t * particles);
void moveParticles(particles_t * particles, const int width, const int height, const float dt, const float drag);
void displayTextureFromPixels(const GLuint * texture, const uint8_t * pixels, const int width, const int height);

cudaError_t allocateAndCopyParticlesToDevice(const particles_t * h_particles, gpu_particles_handle_t * handle);
void freeParticlesOnGPU(gpu_particles_handle_t * handle);

// Konwersja wartości pola do koloru
__device__ uchar4 value_to_color(float value, float min, float max)
{
    float t = (value - min) / (max - min);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    uchar r, g, b;

    if (t < 0.5f) {
        float f = 2.0f * t;
        float ratio = 1.0f - sqrtf(1.0f - f);
        uchar byte = (uchar)(ratio * 255.0f);
        r = byte;
        g = byte;
        b = 255;
    }
    else {
        float f = 2.0f * (t - 0.5f);
        float ratio = 1.0f - sqrtf(f);
        uchar byte = (uchar)(ratio * 255.0f);
        r = 255;
        g = byte;
        b = byte;
    }

    return make_uchar4(r, g, b, 255);
}

// Kernel zapisujący pole bezpośrednio do tekstury (surface)
__global__ void calculateFieldToTextureKernel(const particles_t* particles, int width, int height, cudaSurfaceObject_t surface)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int x = i % width;
    int y = i / width;

    float value = 0.0;
    for (int j = 0; j < particles->count; j++) {
        float dx = x - particles->x[j];
        float dy = y - particles->y[j];
        float dist2 = dx * dx + dy * dy + 1e-3;
        value += particles->q[j] / dist2;
    }

    uchar4 color = value_to_color((float)value, -1.0f, 1.0f);
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

// Funkcje OpenGL
int setupWindowAndTexture(GLFWwindow * *window, GLuint * texture, int width, int height)
{
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    *window = glfwCreateWindow(width, height, TITLE, nullptr, nullptr);
    if (!*window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(*window);
    glOrtho(0, 1, 0, 1, -1, 1);

    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    return 0;
}

void drawTexture(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(0, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 1);
    glTexCoord2f(0, 0); glVertex2f(0, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

__global__ void calculateElectricField_Kernel(const particles_t* particles, const int width, const int height, float* field_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    float value = 0;
    for (int j = 0; j < particles->count; j++) {
        int x = i % width;
        int y = i / width;
        float dx = x - particles->x[j];
        float dy = y - particles->y[j];
        float dist2 = dx * dx + dy * dy + EPS;
        value += particles->q[j] / dist2;
    }
    field_out[i] = value;
}

int main(int argc, char** argv)
{
    // seed dla rand() wykorzystywanego do generacji losowych cząstek
    srand((unsigned int)time(nullptr));

    // Domyślne parametry symulacji
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int count = DEFAULT_PARTICLE_COUNT_CUDA;
    float drag = DEFAULT_DRAG;

    // Parsowanie argumentów wejściowych
    parseInputArgs(argc, argv, &width, &height, &count);
    printf("Ustawienia symulacji:\nRozdzielczość = %d x %d\nLiczba cząstek = %d\nOpór = %f\n", width, height, count, drag);

    // Otworzenie okna i ustawienie kontekstu OpenGL
    GLFWwindow* window;
    GLuint texture;
    if (setupWindowAndOpenGLContext(&window, &texture, width, height) < 0) {
        printf("Error during window setup\n");
        return -1;
    }

    // Inicjalizacja tablic reprezentujących pole i piksele
    float* field = new float[width * height];
    uint8_t* pixels = new uint8_t[width * height * 3];

    // Inicjalizacja losowych cząstek
    particles_t particles(count);
    generateRandomlyMovingParticles(&particles, 0, width, 0, height, -2, +2, -2, +2);

    // Dodanie dużych cząsteczek
    modifyParticleData(&particles, 0, width / 4.0, height / 4.0, 0, 0, 100 * CHARGE_PROTON, 1e9 * MASS_PROTON);
    modifyParticleData(&particles, 1, width / 4.0 * 3, height / 4.0 * 3, 0, 0, 100 * CHARGE_ELECTRON, 1e9 * MASS_PROTON);

    uint64_t frames = 0;
    float frameRefreshInterval = 0.2;
    float prevFrameTime = glfwGetTime();
    float prevTime = prevFrameTime;

    bool paused = false;
    bool spaceWasPressed = false;

    // CUDA-OpenGL interop
    cudaGraphicsResource* cudaTextureResource = nullptr;
    cudaError_t err = CUDA_CHECK_RET(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error during registering texture");
        goto cleanup;
    }

    gpu_particles_handle_t gpu_particles_handle;
    err = allocateAndCopyParticlesToDevice(&particles, &gpu_particles_handle);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error during allocation and copying of particles");
        goto cleanup;
    }
    particles.cleanup();

    while (!glfwWindowShouldClose(window))
    {
        // Wyznaczanie dt
        float currentFrameTime = glfwGetTime();
        float dt = currentFrameTime - prevFrameTime;
        prevFrameTime = currentFrameTime;
        dt = std::min(dt, 0.03f);

        // Zatrzymywanie / wznawianie symulacji
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            if (!spaceWasPressed) {
                paused = !paused;
                spaceWasPressed = true;
            }
        }
        else {
            spaceWasPressed = false;
        }

        if (!paused)
        {
            // Mapujemy teksturę do CUDA
            CUDA_CHECK_RET(cudaGraphicsMapResources(1, &cudaTextureResource, 0));
            cudaArray* textureArray;
            CUDA_CHECK_RET(cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaTextureResource, 0, 0));

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = textureArray;
            cudaSurfaceObject_t surface = 0;
            CUDA_CHECK_RET(cudaCreateSurfaceObject(&surface, &resDesc));

            // Kernel
            int threads = 256;
            int blocks = (width * height + threads - 1) / threads;
            calculateFieldToTextureKernel << <blocks, threads >> > (gpu_particles_handle.d_struct, width, height, surface);
            cudaDeviceSynchronize();

            CUDA_CHECK_RET(cudaDestroySurfaceObject(surface));
            CUDA_CHECK_RET(cudaGraphicsUnmapResources(1, &cudaTextureResource, 0));

            // Wyświetlamy teksturę
            drawTexture(texture);
            glfwSwapBuffers(window);
        }

        // Wyznaczanie FPS
        frames++;
        if (currentFrameTime - prevTime >= frameRefreshInterval) {
            float fps = frames / (currentFrameTime - prevTime);
            char title[128];
            snprintf(title, sizeof(title), "FPS: %.1f - %s%s", fps, TITLE, paused ? " [PAUSED]" : "");
            glfwSetWindowTitle(window, title);
            frames = 0;
            prevTime = currentFrameTime;
        }

        glfwPollEvents();
    }

cleanup:
    glfwTerminate();
    //particles.cleanup();
    delete[] field;
    delete[] pixels;
    cudaGraphicsUnregisterResource(cudaTextureResource);
    freeParticlesOnGPU(&gpu_particles_handle);
    return 0;
}

// Funkcja alokująca i kopiująca cząstki na GPU, zwraca uchwyt do cząstek na GPU zawierający wskaźniki do poszczególnych tablic i do całej struktury
cudaError_t allocateAndCopyParticlesToDevice(const particles_t* h_particles, gpu_particles_handle_t* handle)
{
    if (!h_particles || !handle) return cudaErrorInvalidValue;

    int count = h_particles->count;
    cudaError_t err = cudaSuccess;
    particles_t tmp;

    // Zerujemy handle
    handle->d_struct = nullptr;
    handle->d_x = handle->d_y = handle->d_vx = handle->d_vy = nullptr;
    handle->d_ax = handle->d_ay = handle->d_q = handle->d_m = nullptr;

    // Alokacja struktury GPU
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_struct, sizeof(particles_t)), cleanup);

    // Alokacja tablic GPU
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_x, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_y, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_vx, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_vy, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_ax, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_ay, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_q, count * sizeof(float)), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&handle->d_m, count * sizeof(float)), cleanup);

    // Kopiowanie danych CPU -> GPU
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_x, h_particles->x, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_y, h_particles->y, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_vx, h_particles->vx, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_vy, h_particles->vy, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_ax, h_particles->ax, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_ay, h_particles->ay, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_q, h_particles->q, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_m, h_particles->m, count * sizeof(float), cudaMemcpyHostToDevice), cleanup);

    // Tworzenie tymczasowej struktury z wskaźnikami GPU
    tmp.count = count;
    tmp.x = handle->d_x;
    tmp.y = handle->d_y;
    tmp.vx = handle->d_vx;
    tmp.vy = handle->d_vy;
    tmp.ax = handle->d_ax;
    tmp.ay = handle->d_ay;
    tmp.q = handle->d_q;
    tmp.m = handle->d_m;

    // Kopiowanie struktury na GPU
    CUDA_CHECK_GOTO(cudaMemcpy(handle->d_struct, &tmp, sizeof(particles_t), cudaMemcpyHostToDevice), cleanup);

    return cudaSuccess;

cleanup:
    if (handle->d_x) cudaFree(handle->d_x);
    if (handle->d_y) cudaFree(handle->d_y);
    if (handle->d_vx) cudaFree(handle->d_vx);
    if (handle->d_vy) cudaFree(handle->d_vy);
    if (handle->d_ax) cudaFree(handle->d_ax);
    if (handle->d_ay) cudaFree(handle->d_ay);
    if (handle->d_q) cudaFree(handle->d_q);
    if (handle->d_m) cudaFree(handle->d_m);
    if (handle->d_struct) cudaFree(handle->d_struct);
    return err;
}

// Funkcja zwalniająca pamięć GPU zajmowaną przez cząstki
void freeParticlesOnGPU(gpu_particles_handle_t* handle)
{
    printf("cleanup CUDA memory\n");
    cudaFree(handle->d_x);
    cudaFree(handle->d_y);
    cudaFree(handle->d_vx);
    cudaFree(handle->d_vy);
    cudaFree(handle->d_ax);
    cudaFree(handle->d_ay);
    cudaFree(handle->d_q);
    cudaFree(handle->d_m);
    cudaFree(handle->d_struct);
    printf("done\n");
}

// Parsowanie argumentów
void parseInputArgs(const int argc, char* const* argv, int* width, int* height, int* count) {
    int iarg1, iarg2, iarg3;
    //float darg;
    switch (argc)
    {
    case 2:
        iarg1 = atoi(argv[1]);
        if (iarg1 > 0) *count = iarg1;
        break;
    case 3:
        iarg1 = atoi(argv[1]);
        iarg2 = atoi(argv[2]);
        if (iarg1 > 0 && iarg2 > 0) {
            *width = iarg1;
            *height = iarg2;
        }
        break;
    case 4:
        iarg1 = atoi(argv[1]);
        iarg2 = atoi(argv[2]);
        iarg3 = atoi(argv[3]);
        if (iarg1 > 0 && iarg2 > 0 && iarg3 > 0) {
            *width = iarg1;
            *height = iarg2;
            *count = iarg3;
        }
        break;
    default:
        break;
    }
}

// Ustawienie okna i OpenGL
int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, const int width, const int height) {
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    *window = glfwCreateWindow(width, height, TITLE, nullptr, nullptr);
    if (!*window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(*window);
    glfwSwapInterval(0);
    glOrtho(0, 1, 0, 1, -1, 1);

    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    return 0;
}

void displayTextureFromPixels(const GLuint* texture, const uint8_t* pixels, const int width, const int height) {
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(0, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 1);
    glTexCoord2f(0, 0); glVertex2f(0, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

// Obliczanie natężenia pola elektrycznego
void calculateElectricField(float* field, const int width, const int height, const particles_t* particles) {
    // what do I do here actually?
}

// Mapowanie wartości natężenia pola elektrycznego na gradient
void mapFieldValuesToPixels(const float* field, const int width, const int height, uint8_t* pixels) {
    for (int i = 0; i < width * height; i++) {
        unsigned char r, g, b;
        value_to_color(field[i], CHARGE_ELECTRON, CHARGE_PROTON, &r, &g, &b);
        pixels[i * 3 + 0] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }
}

// Nanoszenie cząstek do rysowanej tablicy pixeli
void drawParticles(const particles_t* particles, uint8_t* pixels, const int width, const int height) {
    for (int i = 0; i < particles->count; i++)
    {
        int x = (int)particles->x[i];
        int y = (int)particles->y[i];
        if (x < 0 || x >= width) {
            fprintf(stderr, "ERR: x = %d\n", x);
            continue;
        }
        if (y < 0 || y >= height) {
            fprintf(stderr, "ERR: y = %d\n", y);
            continue;
        }
        int idx = (y * width + x) * 3;
        if (idx < 0 || idx >= width * height * 3) {
            fprintf(stderr, "ERR: idx = %d\n", idx);
            continue;
        }
        pixels[idx] = 0;     // R
        pixels[idx + 1] = 0; // G
        pixels[idx + 2] = 0; // B
    }
}

// Obliczanie przyspieszeń wynikających z pola elektrycznego
void updateAccelerations(particles_t* particles) {
    for (int i = 0; i < particles->count; i++) {
        float ax = 0.0;
        float ay = 0.0;
        for (int j = 0; j < particles->count; j++) {
            if (i == j) continue;

            float dx = particles->x[i] - particles->x[j];
            float dy = particles->y[i] - particles->y[j];
            float dist2 = dx * dx + dy * dy + EPS;
            float invDist = 1.0 / sqrt(dist2);
            float invDist3 = invDist * invDist * invDist;

            float f = K * particles->q[i] * particles->q[j] * invDist3 / particles->m[i];
            ax += f * dx;
            ay += f * dy;
        }

        particles->ax[i] = ax;
        particles->ay[i] = ay;
    }
}

// Zmiana prędkości i położenia cząstek na podstawie przyspieszeń
void moveParticles(particles_t* particles, const int width, const int height, const float dt, const float drag) {
    for (int i = 0; i < particles->count; i++) {
        // Opory ruchu
        particles->ax[i] -= drag * particles->vx[i];
        particles->ay[i] -= drag * particles->vy[i];

        // --- Forward Euler ---
        //particles->vx[i] += particles->ax[i] * dt;
        //particles->vy[i] += particles->ay[i] * dt;
        //particles->x[i] += particles->vx[i] * dt;
        //particles->y[i] += particles->vy[i] * dt;

        // --- Semi-Implicit Euler ---
        particles->vx[i] += particles->ax[i] * dt;
        particles->vy[i] += particles->ay[i] * dt;
        particles->x[i] += (particles->vx[i] + 0.5 * (particles->ax[i]) * dt) * dt;
        particles->y[i] += (particles->vy[i] + 0.5 * (particles->ay[i]) * dt) * dt;

        // Odbicia
        if (particles->x[i] < 0 && particles->vx[i] < 0) {
            particles->x[i] = 0;
            particles->vx[i] *= -1;
        }
        if (particles->x[i] > width - 1 && particles->vx[i] > 0) {
            particles->x[i] = width - 1;
            particles->vx[i] *= -1;
        }
        if (particles->y[i] < 0 && particles->vy[i] < 0) {
            particles->y[i] = 0;
            particles->vy[i] *= -1;
        }
        if (particles->y[i] > height - 1 && particles->vy[i] > 0) {
            particles->y[i] = height - 1;
            particles->vy[i] *= -1;
        }
    }
}

// Konwersja wartości z zakresu [min, max] na podwójny gradient RGB (blue -> white -> red)
static void value_to_color(const float value, const float min, const float max, unsigned char* r, unsigned char* g, unsigned char* b) {
    float t = (value - min) / (max - min);
    t = std::clamp(t, 0.0f, 1.0f);

    if (t < 0.5f) {
        float f = 2 * t;
        float ratio = 1.0f - sqrt(1.0f - f);
        unsigned char byte = ratio * 255;
        *r = byte;
        *g = byte;
        *b = 255;
    }
    else {
        float f = 2.0f * (t - 0.5f);
        float ratio = 1.0f - sqrt(f);
        unsigned char byte = ratio * 255;
        *r = 255;
        *g = byte;
        *b = byte;
    }
}