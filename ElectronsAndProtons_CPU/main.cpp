#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

#include "particles_t.h"
#include "i2xy_t.h"

#define TITLE "ElectronsAndProtons_CPU"

#define WIDTH 800
#define HEIGHT 600

#define EPS 1e-6

#define K 1e3 // stała elektrostatyczna, wprost proporcjonalna do siły z jaką cząstki odzdziałują na siebie
#define MASS_PROTON 1.0
#define MASS_ELECTRON 1.0
#define CHARGE_PROTON 1.0
#define CHARGE_ELECTRON -1.0

#define PARTICLE_COUNT 100

// Konwersja wartości z zakresu [min, max] na podwójny gradient RGB (blue -> white -> red)
static void value_to_color(double value, double min, double max, unsigned char* r, unsigned char* g, unsigned char* b)
{
    min = min;
    max = max;
    double t = (value - min) / (max - min);
    t = std::clamp(t, 0.0, 1.0);
    
    double f, ratio;
    if (t < 0.5) {
        f = 2 * t;
        ratio = 1 - sqrt(1 - f);
        *r = (unsigned char)(ratio * 255);
        *g = (unsigned char)(ratio * 255);
        *b = 255;
    }
    else {
        f = 2 * (t - 0.5);
        ratio = 1 - sqrt(f);
        *r = 255;
        *g = (unsigned char)(ratio * 255);
        *b = (unsigned char)(ratio * 255);
    }
}

// Zapełnia strukturę nieruchomymi cząstkami
static void generateStationaryParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y)
{
    for (int i = 0; i < particles->count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)RAND_MAX * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)RAND_MAX * (max_y - min_y);
        particles->vx[i] = 0;
        particles->vy[i] = 0;
        particles->ax[i] = 0;
        particles->ay[i] = 0;
        particles->q[i] = (rand() % 2) ? CHARGE_ELECTRON : CHARGE_PROTON;
    }
}

// Zapełnia strukturę losowo poruszającymi się cząstkami
static void generateRandomlyMovingParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y, double min_vx, double max_vx, double min_vy, double max_vy)
{
    for (int i = 0; i < particles->count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)RAND_MAX * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)RAND_MAX * (max_y - min_y);
        particles->vx[i] = min_vx + (double)rand() / (double)RAND_MAX * (max_vx - min_vx);
        particles->vy[i] = min_vy + (double)rand() / (double)RAND_MAX * (max_vy - min_vy);
        particles->ax[i] = 0;
        particles->ay[i] = 0;
        particles->q[i] = (rand() % 2) ? CHARGE_ELECTRON : CHARGE_PROTON;
    }
}

int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, int width, int height);
void calculateElectricField(double* field, int width, int height, i2xy_t* i2xy, particles_t* particles);
void mapFieldValuesToPixels(double* field, int width, int height, uint8_t* pixels);
void drawParticles(particles_t* particles, uint8_t* pixels, int width, int height);
void updateAccelerations(particles_t* particles);
void moveParticles(particles_t* particles, double dt);
void displayTextureFromPixels(GLuint* texture, uint8_t* pixels, int width, int height);

int main(int argc, char** argv)
{
    srand((unsigned int)time(nullptr));

    int width = WIDTH;
    int height = HEIGHT;
    int count = PARTICLE_COUNT;

    GLFWwindow* window;
    GLuint texture;
    if(setupWindowAndOpenGLContext(&window, &texture, width, height) < 0){
        printf("Error during window setup\n");
        return -1;
    }

    i2xy_t i2xy(width, height);
    double* field = new double[width * height];
    uint8_t* pixels = new uint8_t[width * height * 3];
    particles_t particles(count);
    generateRandomlyMovingParticles(&particles, 0, width, 0, height, -2, +2, -2, +2);

    uint64_t frames = 0;
    double frameRefreshInterval = 0.2;
    double prevFrameTime = glfwGetTime();
    double prevTime = prevFrameTime;

    while (!glfwWindowShouldClose(window))
    {
        double currentFrameTime = glfwGetTime();
        double dt = currentFrameTime - prevFrameTime;
        prevFrameTime = currentFrameTime;
        dt = std::min(dt, 0.05);

        // --- Obliczenia ---
        calculateElectricField(field, width, height, &i2xy, &particles);
        mapFieldValuesToPixels(field, width, height, pixels);
        drawParticles(&particles, pixels, width, height);
        
        // --- Wyświetlanie ---
        displayTextureFromPixels(&texture, pixels, width, height); 
        
        // --- Obliczenia ---
        updateAccelerations(&particles);
        moveParticles(&particles, dt);

        // --- Wyznaczanie FPS ---
        frames++;
        if (currentFrameTime - prevTime >= frameRefreshInterval) {
            double fps = frames / (currentFrameTime - prevTime);
            char title[128];
            snprintf(title, sizeof(title), "%s - FPS: %.1f", TITLE, fps);
            glfwSetWindowTitle(window, title);
            frames = 0;
            prevTime = currentFrameTime;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    delete[] field;
    delete[] pixels;
    return 0;
}

// Ustawienie okna i OpenGL
int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, int width, int height){
    if (!glfwInit()) {
        return -1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    *window = glfwCreateWindow(width, height, TITLE, nullptr, nullptr);
    if (!window)
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
        for (int j = 0; j < particles->count; j++)
        {
            int x = i2xy->x[i];
            int y = i2xy->y[i];
            double dx = x - particles->x[j];
            double dy = y - particles->y[j];
            double dist2 = dx * dx + dy * dy;
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
        int idx = (y * width + x) * 3;
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

            double f = K * particles->q[i] * particles->q[j] * invDist3;
            ax += f * dx;
            ay += f * dy;
        }

        particles->ax[i] = ax;
        particles->ay[i] = ay;
    }
}

// Zmiania prędkości i położenia cząstek na podstawie przyspieszeń
void moveParticles(particles_t* particles, double dt){
    for (int i = 0; i < particles->count; i++) {
        // Zmiana prędkości
        particles->vx[i] += particles->ax[i] * dt;
        particles->vy[i] += particles->ay[i] * dt;

        // Zmiana położenia
        particles->x[i] += particles->vx[i] * dt;
        particles->y[i] += particles->vy[i] * dt;

        // Odbicia
        if (particles->x[i] < 0) { 
            particles->x[i] = 0; 
            particles->vx[i] *= -1; 
        }
        if (particles->x[i] > WIDTH - 1) { 
            particles->x[i] = WIDTH - 1; 
            particles->vx[i] *= -1; 
        }
        if (particles->y[i] < 0) { 
            particles->y[i] = 0; 
            particles->vy[i] *= -1; 
        }
        if (particles->y[i] > HEIGHT - 1) { 
            particles->y[i] = HEIGHT - 1; 
            particles->vy[i] *= -1; 
        }
    }
}
