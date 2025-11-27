#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <algorithm>
#include <cstdio>
#include <ctime>

#include "particles_t.h"
#include "constants.h"

#define TITLE "ElectronsAndProtons_CUDA"

#define uchar unsigned char

// Makro wypisujące informację o błędzie i zwracające go
#define CUDA_CHECK_RET(call)                                    \
    ([&]() -> cudaError_t {                                     \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                cudaGetErrorString(err), __FILE__, __LINE__);   \
        }                                                       \
        return err;                                             \
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

// Uchwyt do cząstek na GPU
struct gpu_particles_handle_t {
    particles_t* d_struct;
    float* d_x, * d_y, * d_vx, * d_vy, * d_ax, * d_ay, * d_q, * d_m;
};

int setupWindowAndOpenGLContext(GLFWwindow * *window, GLuint * texture, const int width, const int height);
void parseInputArgs(const int argc, char* const* argv, int* width, int* height, int* count);

cudaError_t allocateAndCopyParticlesToDevice(const particles_t * h_particles, gpu_particles_handle_t * handle);
void freeParticlesOnGPU(gpu_particles_handle_t * handle);
void drawTexture(const GLuint & texture);

__device__ uchar4 value_to_color(float value, float min, float max);
__global__ void calculateFieldToTexture_KernelShared(const particles_t * particles, int width, int height, cudaSurfaceObject_t surface);
__global__ void drawParticlesToTexture_Kernel(const particles_t * particles, int width, int height, cudaSurfaceObject_t surface);
__global__ void updateAccelerations_KernelShared(const particles_t * particles);
__global__ void moveParticles_Kernel(particles_t * particles, int width, int height, float dt, float drag);

int main(int argc, char** argv)
{
    // seed dla rand() wykorzystywanego do generacji losowych cząstek
    srand((unsigned int)time(nullptr));

    // Domyślne parametry symulacji
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int particleCount = DEFAULT_PARTICLE_COUNT_CUDA;
    float drag = DEFAULT_DRAG;

    // Parsowanie argumentów wejściowych
    parseInputArgs(argc, argv, &width, &height, &particleCount);
    printf("Ustawienia symulacji:\nRozdzielczość = %d x %d\nLiczba cząstek = %d\nOpór = %f\n", width, height, particleCount, drag);

    // Otworzenie okna i ustawienie kontekstu OpenGL
    GLFWwindow* window;
    GLuint texture;
    if (setupWindowAndOpenGLContext(&window, &texture, width, height) < 0) {
        printf("Error during window setup\n");
        return -1;
    }

    // Inicjalizacja losowych cząstek
    particles_t particles(particleCount);
    generateRandomlyMovingParticles(&particles, 0, width, 0, height, -2, +2, -2, +2);

    // Dodanie dużych cząsteczek
    modifyParticleData(&particles, 0, width / 4.0, height / 4.0, 0, 0, 1e3 * CHARGE_ELECTRON, 1e9 * MASS_ELECTRON);
    modifyParticleData(&particles, 1, width / 4.0 * 3, height / 4.0 * 3, 0, 0, 1e3 * CHARGE_ELECTRON, 1e9 * MASS_ELECTRON);
    modifyParticleData(&particles, 2, width / 4.0, height / 4.0 * 3, 0, 0, 1000 * CHARGE_PROTON, 1e9 * MASS_PROTON);
    modifyParticleData(&particles, 3, width / 4.0 * 3, height / 4.0, 0, 0, 1000 * CHARGE_PROTON, 1e9 * MASS_PROTON);

    uint64_t frames = 0;
    float frameRefreshInterval = 0.2;
    float prevFrameTime = glfwGetTime();
    float prevTime = prevFrameTime;

    bool paused = false;
    bool spaceWasPressed = false;

    cudaGraphicsResource* cudaTextureResource = nullptr;
    
    int deviceCount = 0;
    cudaError_t err = CUDA_CHECK_RET(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0){
        fprintf(stderr, "No CUDA devices found\n");
        goto cleanup;
    }

    err = CUDA_CHECK_RET(cudaSetDevice(0));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error while setting device\n");
        goto cleanup;
    }

    // CUDA-OpenGL
    err = CUDA_CHECK_RET(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during registering texture\n");
        goto cleanup;
    }

    // Alokacja i kopiowanie cząstek na GPU
    gpu_particles_handle_t gpu_particles_handle;
    err = allocateAndCopyParticlesToDevice(&particles, &gpu_particles_handle);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during allocation and copying of particles\n");
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
        else if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }
        else {
            spaceWasPressed = false;
        }

        if (!paused)
        {
            // Mapowanie i tworzenie tekstury CUDA
            CUDA_CHECK_RET(cudaGraphicsMapResources(1, &cudaTextureResource));
            cudaArray* textureArray;
            CUDA_CHECK_RET(cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaTextureResource, 0, 0));

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = textureArray;
            cudaSurfaceObject_t surface = 0;
            CUDA_CHECK_RET(cudaCreateSurfaceObject(&surface, &resDesc));

            // Obliczanie pola elektrycznego i zapisywanie go do tekstury
            int threads = 512;
            int blocks = (width * height + threads - 1) / threads;
            calculateFieldToTexture_KernelShared <<<blocks, threads, 3 * threads * sizeof(float)>>> (gpu_particles_handle.d_struct, width, height, surface);
            CUDA_CHECK_RET(cudaGetLastError());
            CUDA_CHECK_RET(cudaDeviceSynchronize());

            // Kernel nanoszący cząstki na teksturę
            threads = 512;
            blocks = (particleCount + threads - 1) / threads;
            drawParticlesToTexture_Kernel <<<blocks, threads>>> (gpu_particles_handle.d_struct, width, height, surface);
            CUDA_CHECK_RET(cudaGetLastError());
            CUDA_CHECK_RET(cudaDeviceSynchronize());

            // Kernel obliczający przyspieszenia cząstek
            threads = 512;
            blocks = (particleCount + threads - 1) / threads;
            updateAccelerations_KernelShared <<<blocks, threads, 3 * threads * sizeof(float)>>> (gpu_particles_handle.d_struct);
            CUDA_CHECK_RET(cudaGetLastError());
            CUDA_CHECK_RET(cudaDeviceSynchronize());

            // Kernel przesuwający cząstki
            threads = 512;
            blocks = (particleCount + threads - 1) / threads;
            moveParticles_Kernel<<<blocks, threads>>> (gpu_particles_handle.d_struct, width, height, dt, drag);
            CUDA_CHECK_RET(cudaGetLastError());
            CUDA_CHECK_RET(cudaDeviceSynchronize());

            // Czyszczenie
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
    cudaGraphicsUnregisterResource(cudaTextureResource);
    freeParticlesOnGPU(&gpu_particles_handle);
    return 0;
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
    if (!*window) {
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

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    cudaFree(handle->d_x);
    cudaFree(handle->d_y);
    cudaFree(handle->d_vx);
    cudaFree(handle->d_vy);
    cudaFree(handle->d_ax);
    cudaFree(handle->d_ay);
    cudaFree(handle->d_q);
    cudaFree(handle->d_m);
    cudaFree(handle->d_struct);
}

// Rysowanie tekstury na oknie
void drawTexture(const GLuint& texture)
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

// Konwersja wartości pola do koloru na GPU
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

// Kernel obliczający pole elektryczne i zapisujący je do tekstury
__global__ void calculateFieldToTexture_KernelShared(const particles_t* particles, int width, int height, cudaSurfaceObject_t surface) {
    extern __shared__ float sh[];
    float* sh_x = sh;
    float* sh_y = sh + blockDim.x;
    float* sh_q = sh + 2 * blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= width * height) return;

    int x = i % width;
    int y = i / width;

    float value = 0.0f;

    int nParticles = particles->count;
    int batchSize = blockDim.x;

    for (int b = 0; b < nParticles; b += batchSize) {
        int idx = b + tid;
        if (idx < nParticles) {
            sh_x[tid] = particles->x[idx];
            sh_y[tid] = particles->y[idx];
            sh_q[tid] = particles->q[idx];
        }
        __syncthreads();

        int limit = min(batchSize, nParticles - b);
        for (int j = 0; j < limit; j++) {
            float dx = x - sh_x[j];
            float dy = y - sh_y[j];
            float dist2 = dx * dx + dy * dy + 1e-3f;
            value += sh_q[j] / dist2;
        }
        __syncthreads();
    }

    uchar4 color = value_to_color(value, CHARGE_ELECTRON, CHARGE_PROTON);
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

// Kernel nanoszący cząstki na teksturę CUDA
__global__ void drawParticlesToTexture_Kernel(const particles_t* particles, int width, int height, cudaSurfaceObject_t surface)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles->count) return;

    int x = (int)particles->x[i];
    int y = (int)particles->y[i];

    if (x < 0 || x >= width || y < 0 || y >= height) return;

    uchar4 black = make_uchar4(0, 0, 0, 255);
    surf2Dwrite(black, surface, x * sizeof(uchar4), y);
}

// Kernel obliczający przyspieszenia wynikające z oddziaływań cząstek
__global__ void updateAccelerations_KernelShared(const particles_t* particles)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i >= particles->count) return;

    float xi = particles->x[i];
    float yi = particles->y[i];
    float qi = particles->q[i];
    float mi = particles->m[i];

    float ax = 0.0f;
    float ay = 0.0f;

    int block_size = blockDim.x;
    extern __shared__ float sh[];
    float* sh_x = sh;
    float* sh_y = sh + block_size;
    float* sh_q = sh + 2 * block_size;

    int nParticles = particles->count;

    for (int b = 0; b < nParticles; b += block_size) {
        int idx = b + tid;

        if (idx < nParticles) {
            sh_x[tid] = particles->x[idx];
            sh_y[tid] = particles->y[idx];
            sh_q[tid] = particles->q[idx];
        }
        __syncthreads();

        int limit = min(block_size, nParticles - b);
        for (int j = 0; j < limit; j++) {
            if (i == b + j) continue;
            float dx = xi - sh_x[j];
            float dy = yi - sh_y[j];
            float dist2 = dx * dx + dy * dy + EPS;
            float invDist = rsqrtf(dist2);
            float invDist3 = invDist * invDist * invDist;
            float f = K * qi * sh_q[j] * invDist3 / mi;
            ax += f * dx;
            ay += f * dy;
        }
        __syncthreads();
    }

    particles->ax[i] = ax;
    particles->ay[i] = ay;
}

__global__ void moveParticles_Kernel(particles_t* particles, int width, int height, float dt, float drag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles->count) return;

    float x = particles->x[i];
    float y = particles->y[i];
    float vx = particles->vx[i];
    float vy = particles->vy[i];
    float ax = particles->ax[i];
    float ay = particles->ay[i];

    // Opory ruchu
    ax -= drag * vx;
    ay -= drag * vy;

    // Metoda Eulera-Cromera
    vx += ax * dt;
    vy += ay * dt;
    x += vx * dt;
    y += vy * dt;

    // Odbicia
    if (x < 0.0f && vx < 0.0f) {
        x = 0.0f;
        vx *= -1.0f;
    }
    if (x > width - 1.0f && vx > 0.0f) {
        x = width - 1.0f;
        vx *= -1.0f;
    }
    if (y < 0.0f && vy < 0.0f) {
        y = 0.0f;
        vy *= -1.0f;
    }
    if (y > height - 1.0f && vy > 0.0f) {
        y = height - 1.0f;
        vy *= -1.0f;
    }

    particles->x[i] = x;
    particles->y[i] = y;
    particles->vx[i] = vx;
    particles->vy[i] = vy;
    particles->ax[i] = ax;
    particles->ay[i] = ay;
}
