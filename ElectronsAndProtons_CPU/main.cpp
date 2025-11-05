#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#define TITLE "ElectronsAndProtons_CPU"

#define HEIGHT 720
#define WIDTH 1280

#define MINVAL 0.0f
#define MAXVAL 1.0f

// Convert [min, max] value → RGB (blue→red)
static void value_to_color(float value, float min, float max, unsigned char* r, unsigned char* g, unsigned char* b)
{
    float t = (value - min) / (max - min);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    *r = (unsigned char)(t * 255);
    *g = 0;
    *b = (unsigned char)((1.0f - t) * 255);
}

int main(void)
{
    srand((unsigned int)time(nullptr));

    int width = WIDTH;
    int height = HEIGHT;

    float* values = new float[width * height];
    unsigned char* pixels = new unsigned char[width * height * 3];

    if (!glfwInit()) {
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(width, height, TITLE, nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glOrtho(0, 1, 0, 1, -1, 1);
    glPixelZoom(1.0f, -1.0f);
    glRasterPos2f(0, 1);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    double prevTime = glfwGetTime();
    int frames = 0;
    double fps = 0.0;
    while (!glfwWindowShouldClose(window))
    {
        // --- Calculations ---
        // Generate random data
        for (int i = 0; i < width * height; i++) {
            values[i] = MINVAL + (float)rand() / (float)RAND_MAX * (MAXVAL - MINVAL);
        }

        // Map to colors
        for (int i = 0; i < width * height; i++) {
            unsigned char r, g, b;
            value_to_color(values[i], MINVAL, MAXVAL, &r, &g, &b);
            pixels[i * 3 + 0] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }

        // --- Display ---
        // Upload to GPU
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

        // Draw textured quad
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f(1, 0);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 1); glVertex2f(0, 1);
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

    free(values);
    free(pixels);
    glfwTerminate();
    return 0;
}