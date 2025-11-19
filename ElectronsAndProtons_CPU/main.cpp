#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <getopt.h>

#include "particles_t.h"
#include "i2xy_t.h"

#define EPS 1e-6
#define K 1e3 // stała elektrostatyczna, wprost proporcjonalna do siły z jaką cząstki odzdziałują na siebie
#define MASS_PROTON 1.0
#define MASS_ELECTRON 1.0
#define CHARGE_PROTON 1.0
#define CHARGE_ELECTRON -1.0

#define DEFAULT_WIDTH 600
#define DEFAULT_HEIGHT 600
#define DEFAULT_PARTICLE_COUNT 100
#define DEFAULT_DAMPING 0.999
#define DEFAULT_DRAG 0.8

#define TITLE "ElectronsAndProtons_CPU"

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

// Zapełnia strukturę nieruchomymi cząstkami
static void generateStationaryParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y) {
    for (int i = 0; i < particles->count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)(RAND_MAX + 1.0) * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)(RAND_MAX + 1.0) * (max_y - min_y);
        particles->vx[i] = 0;
        particles->vy[i] = 0;
        particles->ax[i] = 0;
        particles->ay[i] = 0;
        particles->q[i] = (rand() % 2) ? CHARGE_ELECTRON : CHARGE_PROTON;
    }
}

// Zapełnia strukturę losowo poruszającymi się cząstkami
static void generateRandomlyMovingParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y, double min_vx, double max_vx, double min_vy, double max_vy) {
    for (int i = 0; i < particles->count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)(RAND_MAX + 1.0) * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)(RAND_MAX + 1.0) * (max_y - min_y);
        particles->vx[i] = min_vx + (double)rand() / (double)(RAND_MAX + 1.0) * (max_vx - min_vx);
        particles->vy[i] = min_vy + (double)rand() / (double)(RAND_MAX + 1.0) * (max_vy - min_vy);
        particles->ax[i] = 0;
        particles->ay[i] = 0;
        particles->q[i] = (rand() % 2) ? CHARGE_ELECTRON : CHARGE_PROTON;
    }
}

void parseInputArgs(int argc, char** argv, int* width, int* height, int* count, double* damping);
int setupWindowAndOpenGLContext(GLFWwindow** window, GLuint* texture, int width, int height);
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
    double damping = DEFAULT_DAMPING;

    // Parsowanie argumentów wejściowych
    parseInputArgs(argc, argv, &width, &height, &count, &damping);
    printf("Ustawienia symulacji:\nRozdzielczość = %d x %d\nLiczba cząstek = %d\nWsp. tłumienia = %f\n", width, height, count, damping);

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

    uint64_t frames = 0;
    double frameRefreshInterval = 0.2;
    double prevFrameTime = glfwGetTime();
    double prevTime = prevFrameTime;

    while (!glfwWindowShouldClose(window))
    {
        // Wyznaczanie dt
        double currentFrameTime = glfwGetTime();
        double dt = currentFrameTime - prevFrameTime;
        prevFrameTime = currentFrameTime;
        dt = std::min(dt, 0.05);

        // Obliczenia natężeń pola
        calculateElectricField(field, width, height, &i2xy, &particles);
        
        // Przygotowanie pikseli do wyświetlenia
        mapFieldValuesToPixels(field, width, height, pixels);
        drawParticles(&particles, pixels, width, height);
        
        // Wyświetlanie na ekran
        displayTextureFromPixels(&texture, pixels, width, height); 
        
        // Kinematyka cząstek
        updateAccelerations(&particles);
        moveParticles(&particles, width, height, dt, damping);

        // Wyznaczanie FPS
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

    // Czyszczenie
    glfwTerminate();
    // particles.cleanup();
    // i2xy.cleanup();
    delete[] field;
    delete[] pixels;
    return 0;
}

// Parsowanie argumentów
void parseInputArgs(int argc, char** argv, int* width, int* height, int* count, double* damping){
    int option_index = 0;
    int c, arg;
    double darg;

    static struct option long_options[] = {
        {"width",   required_argument, 0, 'w'},
        {"height",  required_argument, 0, 'h'},
        {"count",   required_argument, 0, 'c'},
        {"damping", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    while ((c = getopt_long(argc, argv, "w:h:c:d:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'w':
                arg = atoi(optarg);
                if (arg > 0) *width = arg;
                break;
            case 'h':
                arg = atoi(optarg);
                if (arg > 0) *height = arg;
                break;
            case 'c':
                arg = atoi(optarg);
                if (arg > 0) *count = arg;
                break;
            case 'd':
                darg = atof(optarg);
                if (arg >= 0.0) *damping = darg;
                break;
            case '?':
                break;
            default:
                abort();
        }
    }

    if (optind < argc) {
        printf("Non-option arguments:\n");
        while (optind < argc)
            printf("  %s\n", argv[optind++]);
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
        for (int j = 0; j < particles->count; j++)
        {
            int x = i2xy->x[i];
            int y = i2xy->y[i];
            double dx = x - particles->x[j];
            double dy = y - particles->y[j];
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

            double f = K * particles->q[i] * particles->q[j] * invDist3;
            ax += f * dx;
            ay += f * dy;
        }

        particles->ax[i] = ax;
        particles->ay[i] = ay;
    }
}

// Zmiania prędkości i położenia cząstek na podstawie przyspieszeń
void moveParticles(particles_t* particles, int width, int height, double dt, double damping){
    for (int i = 0; i < particles->count; i++) {
        // Opory ruchu
        double drag = DEFAULT_DRAG; // smaller = less friction
        particles->ax[i] -= drag * particles->vx[i];
        particles->ay[i] -= drag * particles->vy[i];

        // Zmiana prędkości
        particles->vx[i] += particles->ax[i] * dt;
        particles->vy[i] += particles->ay[i] * dt;

        // Tłumienie ruchu
        particles->vx[i] *= damping;
        particles->vy[i] *= damping;

        // Zmiana położenia
        particles->x[i] += particles->vx[i] * dt;
        particles->y[i] += particles->vy[i] * dt;

        // Odbicia
        if (particles->x[i] < 0 && particles->vx[i] < 0) { 
            // particles->x[i] = -particles->x[i]; 
            particles->x[i] = 0;
            particles->vx[i] *= -1; 
        }
        if (particles->x[i] > width - 1 && particles->vx[i] > 0) { 
            // particles->x[i] = -particles->x[i] + 2*(width - 1); 
            particles->x[i] = width - 1; 
            particles->vx[i] *= -1; 
        }
        if (particles->y[i] < 0 && particles->vy[i] < 0) { 
            // particles->y[i] = -particles->y[i]; 
            particles->y[i] = 0; 
            particles->vy[i] *= -1; 
        }
        if (particles->y[i] > height - 1 && particles->vy[i] > 0) { 
            // particles->y[i] = -particles->y[i] + 2*(height - 1);
            particles->y[i] = height - 1;
            particles->vy[i] *= -1; 
        }
    }
}
