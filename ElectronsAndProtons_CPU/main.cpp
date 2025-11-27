#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

#include "particles_t.h"
#include "constants.h"

#define TITLE "ElectronsAndProtons_CPU"

void parseInputArgs(const int argc, char* const* argv, int* width, int* height, int* count);
int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, const int width, const int height);

void displayTextureFromPixels(const GLuint* texture, const uint8_t* pixels, const int width, const int height);

void calculateElectricField(float* field, const int width, const int height, const particles_t* particles);
void mapFieldValuesToPixels(const float* field, const int width, const int height, uint8_t* pixels);
void drawParticles(const particles_t* particles, uint8_t* pixels, const int width, const int height);
void updateAccelerations(particles_t* particles);
void moveParticles(particles_t* particles, const int width, const int height, const float dt, const float drag);
void value_to_color(const float value, const float min, const float max, unsigned char* r, unsigned char* g, unsigned char* b);

int main(int argc, char** argv) {
    // seed dla rand() wykorzystywanego do generacji losowych cząstek
    srand((unsigned int)time(nullptr));

    // Domyślne parametry symulacji
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int count = DEFAULT_PARTICLE_COUNT;
    float drag = DEFAULT_DRAG;

    // Parsowanie argumentów wejściowych
    parseInputArgs(argc, argv, &width, &height, &count);
    printf("Ustawienia symulacji:\nRozdzielczość = %d x %d\nLiczba cząstek = %d\nOpór = %f\n", width, height, count, drag);

    // Otworzenie okna i ustawienie kontekstu OpenGL
    GLFWwindow* window;
    GLuint texture;
    if(setupWindowAndOpenGLContext(&window, &texture, width, height) < 0){
        printf("Error during window setup\n");
        return -1;
    }

    float* field = new float[width * height];
    uint8_t* pixels = new uint8_t[width * height * 3];
    
    // Inicjalizacja losowych cząstek
    particles_t particles(count);
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

    while (!glfwWindowShouldClose(window))
    {
        // Wyznaczanie dt
        float currentFrameTime = glfwGetTime();
        float dt = currentFrameTime - prevFrameTime;
        prevFrameTime = currentFrameTime;
        dt = std::min(dt, 0.03f);
        //dt = myMin(dt, 0.03f);

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
            // Obliczenia natężeń pola
            calculateElectricField(field, width, height, &particles);

            // Przygotowanie pikseli do wyświetlenia
            mapFieldValuesToPixels(field, width, height, pixels);
            drawParticles(&particles, pixels, width, height);

            // Zapisywanie pikseli do tekstury
            displayTextureFromPixels(&texture, pixels, width, height);

            // Kinematyka cząstek
            updateAccelerations(&particles);
            moveParticles(&particles, width, height, dt, drag);

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

    glfwTerminate();
    particles.cleanup();
    delete[] field;
    delete[] pixels;
    return 0;
}

// Parsowanie argumentów
void parseInputArgs(const int argc, char* const* argv, int* width, int* height, int* count){
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
            *count = iarg1;
            *width = iarg2;
            *height = iarg3;
        }
        break;
    default:
        break;
    }
}

// Ustawienie okna i OpenGL
int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, int width, int height){
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

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    for (int i = 0; i < width * height; i++) {
        float value = 0;
        for (int j = 0; j < particles->count; j++) {
            int x = i % width;
            int y = i / width;
            float dx = x - particles->x[j];
            float dy = y - particles->y[j];
            float dist2 = dx * dx + dy * dy + EPS;
            value += particles->q[j] / dist2;
        }
        field[i] = value;
    }
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
        if (x < 0 || x >= width) continue;
        if (y < 0 || y >= height) continue;
        int idx = (y * width + x) * 3;
        if (idx < 0 || idx >= width * height * 3) continue;
        pixels[idx] = 0;
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 0;
    }
}

// Obliczanie przyspieszeń wynikających z pola elektrycznego
void updateAccelerations(particles_t* particles){
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
        if (x < 0 && vx < 0) { 
            x = 0;
            vx *= -1; 
        }
        if (x > width - 1 && vx > 0) { 
            x = width - 1; 
            vx *= -1; 
        }
        if (y < 0 && vy < 0) { 
            y = 0; 
            vy *= -1; 
        }
        if (y > height - 1 && vy > 0) { 
            y = height - 1;
            vy *= -1; 
        }

        particles->x[i] = x;
        particles->y[i] = y;
        particles->vx[i] = vx;
        particles->vy[i] = vy;
    }
}

// Konwersja wartości z zakresu [min, max] na podwójny gradient RGB (blue -> white -> red)
void value_to_color(const float value, const float min, const float max, unsigned char* r, unsigned char* g, unsigned char* b) {
    float t = (value - min) / (max - min);
    t = std::clamp(t, 0.0f, 1.0f);

    if (t < 0.5) {
        float f = 2 * t;
        float ratio = 1 - std::sqrt(1 - f);
        unsigned char byte = ratio * 255;
        *r = byte;
        *g = byte;
        *b = 255;
    }
    else {
        float f = 2 * (t - 0.5);
        float ratio = 1 - std::sqrt(f);
        unsigned char byte = ratio * 255;
        *r = 255;
        *g = byte;
        *b = byte;
    }
}