#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

#include "particles_t.h"
#include "i2xy_t.h"
#include "constants.h"

#define TITLE "ElectronsAndProtons_CPU"

void value_to_color(double value, double min, double max, unsigned char* r, unsigned char* g, unsigned char* b);
int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, int width, int height);
void parseInputArgs(int argc, char** argv, int* width, int* height, int* count);
void calculateElectricField(double* field, int width, int height, i2xy_t* i2xy, particles_t* particles);
void mapFieldValuesToPixels(double* field, int width, int height, uint8_t* pixels);
void drawParticles(particles_t* particles, uint8_t* pixels, int width, int height);
void updateAccelerations(particles_t* particles);
void moveParticles(particles_t* particles, int width, int height, double dt, double damping);
void displayTextureFromPixels(GLuint* texture, uint8_t* pixels, int width, int height);

int main(int argc, char** argv)
{
    // seed dla rand() wykorzystywanego do generacji losowych cząstek
    srand((unsigned int)time(nullptr));

    // Domyślne parametry symulacji
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int count = DEFAULT_PARTICLE_COUNT;
    double drag = DEFAULT_DRAG;

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

    // Inicjalizacja tablic reprezentujących pole i piksele
    i2xy_t i2xy(width, height);
    double* field = new double[width * height];
    uint8_t* pixels = new uint8_t[width * height * 3];
    
    // Inicjalizacja losowych cząstek
    particles_t particles(count);
    generateRandomlyMovingParticles(&particles, 0, width, 0, height, -2, +2, -2, +2);

    // Dodanie dużych cząsteczek
    modifyParticleData(&particles, 0, width / 4.0, height / 4.0, 0, 0, 100 * CHARGE_PROTON, 1e9 * MASS_PROTON);
    modifyParticleData(&particles, 1, width / 4.0 * 3, height / 4.0 * 3, 0, 0, 100 * CHARGE_ELECTRON, 1e9 * MASS_PROTON);

    uint64_t frames = 0;
    double frameRefreshInterval = 0.2;
    double prevFrameTime = glfwGetTime();
    double prevTime = prevFrameTime;

    bool paused = false;
    bool spaceWasPressed = false;

    while (!glfwWindowShouldClose(window))
    {
        // Wyznaczanie dt
        double currentFrameTime = glfwGetTime();
        double dt = currentFrameTime - prevFrameTime;
        prevFrameTime = currentFrameTime;
        dt = std::min(dt, 0.03);

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
            // Obliczenia natężeń pola
            calculateElectricField(field, width, height, &i2xy, &particles);

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
            double fps = frames / (currentFrameTime - prevTime);
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
    i2xy.cleanup();
    delete[] field;
    delete[] pixels;
    return 0;
}

// Parsowanie argumentów
void parseInputArgs(int argc, char** argv, int* width, int* height, int* count){
    int iarg1, iarg2, iarg3;
    //double darg;
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

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    return 0;
}

void displayTextureFromPixels(GLuint* texture, uint8_t* pixels, int width, int height){
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
void calculateElectricField(double* field, int width, int height, i2xy_t* i2xy, particles_t* particles){
    for (int i = 0; i < width * height; i++) {
        double value = 0;
        for (int j = 0; j < particles->count; j++) {
            //int x = i2xy->x[i];
            //int y = i2xy->y[i];
            int x = i % width;
            int y = i / width;
            double dx = x - particles->x[j];
            double dy = y - particles->y[j];
            //if (dx > 100 || dy > 100) continue;
            double dist2 = dx * dx + dy * dy + EPS;
            value += particles->q[j] / dist2;
        }
        field[i] = value;
    }
}

// Mapowanie wartości natężenia pola elektrycznego na gradient
void mapFieldValuesToPixels(double* field, int width, int height, uint8_t* pixels){
    for (int i = 0; i < width * height; i++) {
        unsigned char r, g, b;
        value_to_color(field[i], CHARGE_ELECTRON, CHARGE_PROTON, &r, &g, &b);
        pixels[i * 3 + 0] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }
}

// Nanoszenie cząstek do rysowanej tablicy pixeli
void drawParticles(particles_t* particles, uint8_t* pixels, int width, int height){
    for (int i = 0; i < particles->count; i++)
    {
        int x = (int)particles->x[i];
        int y = (int)particles->y[i];
        if(x < 0 || x >= width){
            fprintf(stderr, "ERR: x = %d\n", x);
            continue;
        }
        if(y < 0 || y >= height){
            fprintf(stderr, "ERR: y = %d\n", y);
            continue;
        }
        int idx = (y * width + x) * 3;
        if (idx < 0 || idx >= width*height*3){
            fprintf(stderr, "ERR: idx = %d\n", idx);
            continue;
        }
        pixels[idx] = 0;     // R
        pixels[idx + 1] = 0; // G
        pixels[idx + 2] = 0; // B
    }
}

// Obliczanie przyspieszeń wynikających z pola elektrycznego
void updateAccelerations(particles_t* particles){
    for (int i = 0; i < particles->count; i++) {
        double ax = 0.0;
        double ay = 0.0;
        for (int j = 0; j < particles->count; j++) {
            if (i == j) continue;

            double dx = particles->x[i] - particles->x[j];
            double dy = particles->y[i] - particles->y[j];
            double dist2 = dx * dx + dy * dy + EPS;
            double invDist = 1.0 / sqrt(dist2);
            double invDist3 = invDist * invDist * invDist;

            double f = K * particles->q[i] * particles->q[j] * invDist3 / particles->m[i];
            ax += f * dx;
            ay += f * dy;
        }

        particles->ax[i] = ax;
        particles->ay[i] = ay;
    }
}

// Zmiana prędkości i położenia cząstek na podstawie przyspieszeń
void moveParticles(particles_t* particles, int width, int height, double dt, double drag){
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
static void value_to_color(double value, double min, double max, unsigned char* r, unsigned char* g, unsigned char* b) {
    double t = (value - min) / (max - min);
    t = std::clamp(t, 0.0, 1.0);

    if (t < 0.5) {
        double f = 2 * t;
        double ratio = 1 - sqrt(1 - f);
        unsigned char byte = ratio * 255;
        *r = byte;
        *g = byte;
        *b = 255;
    }
    else {
        double f = 2 * (t - 0.5);
        double ratio = 1 - sqrt(f);
        unsigned char byte = ratio * 255;
        *r = 255;
        *g = byte;
        *b = byte;
    }
}