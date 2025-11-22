#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "particles_t.h"
#include "constants.h"

// Zapełnia strukturę nieruchomymi cząstkami
void generateStationaryParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y) {
    for (int i = 0; i < particles->count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)(RAND_MAX + 1.0) * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)(RAND_MAX + 1.0) * (max_y - min_y);
        particles->vx[i] = 0;
        particles->vy[i] = 0;
        particles->ax[i] = 0;
        particles->ay[i] = 0;
        particles->q[i] = (rand() % 2) ? CHARGE_ELECTRON : CHARGE_PROTON;
        particles->m[i] = (rand() % 2) ? MASS_ELECTRON : MASS_PROTON;
    }
}

// Zapełnia strukturę losowo poruszającymi się cząstkami
void generateRandomlyMovingParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y, double min_vx, double max_vx, double min_vy, double max_vy) {
    for (int i = 0; i < particles->count; i++) {
        particles->x[i] = min_x + (double)rand() / (double)(RAND_MAX + 1.0) * (max_x - min_x);
        particles->y[i] = min_y + (double)rand() / (double)(RAND_MAX + 1.0) * (max_y - min_y);
        particles->vx[i] = min_vx + (double)rand() / (double)(RAND_MAX + 1.0) * (max_vx - min_vx);
        particles->vy[i] = min_vy + (double)rand() / (double)(RAND_MAX + 1.0) * (max_vy - min_vy);
        particles->ax[i] = 0;
        particles->ay[i] = 0;
        particles->q[i] = (rand() % 2) ? CHARGE_ELECTRON : CHARGE_PROTON;
        particles->m[i] = (rand() % 2) ? MASS_ELECTRON : MASS_PROTON;
    }
}

// Modyfikuje dane pojedynczej cząstki
void modifyParticleData(particles_t* particles, int index, double x, double y, double vx, double vy, double q, double m)
{
    if (index < 0 || index > particles->count) return;
    particles->x[index] = x;
    particles->y[index] = y;
    particles->vx[index] = vx;
    particles->vy[index] = vy;
    particles->q[index] = q;
    particles->m[index] = m;
}