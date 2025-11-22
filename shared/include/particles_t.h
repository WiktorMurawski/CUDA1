#pragma once

// Struktura przechowująca informacje o cząstkach w tablicach; SoA
struct particles_t
{
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* ax;
    double* ay;
    double* q;
    double* m;
    int count;

    particles_t() : x(nullptr), y(nullptr), vx(nullptr), vy(nullptr),
        ax(nullptr), ay(nullptr), q(nullptr), m(nullptr), count(0) {
    }

    particles_t(int count)
    {
        this->count = count;
        this->x = new double[count];
        this->y = new double[count];
        this->vx = new double[count];
        this->vy = new double[count];
        this->ax = new double[count];
        this->ay = new double[count];
        this->q = new double[count];
        this->m = new double[count];
    }

    ~particles_t() {
        //this->cleanup();
    }

    void cleanup()
    {
        delete[] this->x;
        delete[] this->y;
        delete[] this->vx;
        delete[] this->vy;
        delete[] this->ax;
        delete[] this->ay;
        delete[] this->q;
        delete[] this->m;
        this->count = 0;
        this->x = nullptr;
        this->y = nullptr;
        this->vx = nullptr;
        this->vy = nullptr;
        this->ax = nullptr;
        this->ay = nullptr;
        this->q = nullptr;
        this->m = nullptr;
    }
};

void generateStationaryParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y);
void generateRandomlyMovingParticles(particles_t* particles, double min_x, double max_x, double min_y, double max_y, double min_vx, double max_vx, double min_vy, double max_vy);
void modifyParticleData(particles_t* particles, int index, double x, double y, double vx, double vy, double q, double m);