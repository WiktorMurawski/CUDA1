//#ifdef _WIN32
//#include <windows.h>
//#endif

#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#define TITLE "ElectronsAndProtons_CPU"

#define WIDTH 800
#define HEIGHT 600

#define MINVAL 0.0f
#define MAXVAL 1.0f

struct particles_t
{
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* q;
    int count;

    ~particles_t()
    {
        delete[] this->x;
        delete[] this->y;
        delete[] this->vx;
        delete[] this->vy;
        delete[] this->q;
        this->count = 0;
    }
};

// Convert [min, max] value -> RGB (blue -> white -> red)
static void value_to_color(double value, double min, double max, unsigned char* r, unsigned char* g, unsigned char* b)
{
    double t = (value - min) / (max - min);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    if (t < 0.5f) {
        double f = t / 0.5f;
        *r = (unsigned char)(f * 255);
        *g = (unsigned char)(f * 255);
        *b = 255;
    }
    else {
        double f = (t - 0.5f) / 0.5f;
        *r = 255;
        *g = (unsigned char)((1.0f - f) * 255);
        *b = (unsigned char)((1.0f - f) * 255);
    }
}

static void generateRandomParticles(particles_t* particles, int count, double min_x, double max_x, double min_y, double max_y, double min_vx, double max_vx, double min_vy, double max_vy, double electronCharge, double protonCharge)
{
    particles->count = count;
    particles->x = new double[count];
    particles->y = new double[count];
    particles->vx = new double[count];
    particles->vy = new double[count];
    particles->q = new double[count];

    for (int i = 0; i < count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)RAND_MAX * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)RAND_MAX * (max_y - min_y);
        particles->vx[i] = min_vx + (double)rand() / (double)RAND_MAX * (max_vx - min_vx);
        particles->vy[i] = min_vy + (double)rand() / (double)RAND_MAX * (max_vy - min_vy);
        particles->q[i] = (rand() % 2) ? electronCharge : protonCharge;
    }
}

int main(void)
{
    srand((unsigned int)time(nullptr));

    int width = WIDTH;
    int height = HEIGHT;

    if (!glfwInit()) {
        return -1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(width, height, TITLE, nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glOrtho(0, 1, 0, 1, -1, 1);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    double* field = new double[width * height];
    unsigned char* pixels = new unsigned char[width * height * 3];
    particles_t particles;
    //int count = 10'000;
    int count = 100;
    generateRandomParticles(&particles, count, 0, width, 0, height, -5, +5, -5, +5, -1.0, 1.0);

    double prevTime = glfwGetTime();
    int frames = 0;
    double fps = 0.0;
    while (!glfwWindowShouldClose(window))
    {
        // --- Calculations ---
        // Calculate field values
        for (int i = 0; i < width * height; i++) {
            double value = 0;
            for (int j = 0; j < particles.count; j++)
            {
                int x = i % width;
                int y = i / width;
                double dist2 = (x - particles.x[j]) * (x - particles.x[j]) + (y - particles.y[j]) * (y - particles.y[j]);
                value += particles.q[j] / (dist2 + 1e-6);
            }
            field[i] = value;
        }

        // Map to colors
        for (int i = 0; i < width * height; i++) {
            unsigned char r, g, b;
            value_to_color(field[i], -0.1, 0.1, &r, &g, &b);
            pixels[i * 3 + 0] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }

        // Draw particles
        for (int i = 0; i < particles.count; i++)
        {
            int x = (int)particles.x[i];
            int y = (int)particles.y[i];
            int idx = (y * width + x) * 3;
            pixels[idx] = 0;     // R
            pixels[idx + 1] = 0; // G
            pixels[idx + 2] = 0;   // B
        }

        // --- Display ---
        // Upload to GPU
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

        // Draw textured quad
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(0, 0);
        glTexCoord2f(1, 1); glVertex2f(1, 0);
        glTexCoord2f(1, 0); glVertex2f(1, 1);
        glTexCoord2f(0, 0); glVertex2f(0, 1);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        // --- FPS calculation ---
        frames++;
        double t = glfwGetTime();
        double interval = 0.1;
        if (t - prevTime >= interval) {
            fps = frames / (t - prevTime);
            char title[128];
            snprintf(title, sizeof(title), "%s - FPS: %.1f", TITLE, fps);
            //printf("%.1f\n", fps);
            glfwSetWindowTitle(window, title);
            frames = 0;
            prevTime = t;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    delete[] field;
    delete[] pixels;
    glfwTerminate();
    return 0;
}