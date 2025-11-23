#ifndef particles_t_h
#define particles_t_h

// Struktura przechowująca informacje o cząstkach w tablicach; SoA
struct particles_t
{
    float* x;
    float* y;
    float* vx;
    float* vy;
    float* ax;
    float* ay;
    float* q;
    float* m;
    int count;

    particles_t() : x(nullptr), y(nullptr), vx(nullptr), vy(nullptr),
        ax(nullptr), ay(nullptr), q(nullptr), m(nullptr), count(0) {
    }

    particles_t(int count)
    {
        this->count = count;
        this->x = new float[count];
        this->y = new float[count];
        this->vx = new float[count];
        this->vy = new float[count];
        this->ax = new float[count];
        this->ay = new float[count];
        this->q = new float[count];
        this->m = new float[count];
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

void generateStationaryParticles(particles_t* particles, float min_x, float max_x, float min_y, float max_y);
void generateRandomlyMovingParticles(particles_t* particles, float min_x, float max_x, float min_y, float max_y, float min_vx, float max_vx, float min_vy, float max_vy);
void modifyParticleData(particles_t* particles, int index, float x, float y, float vx, float vy, float q, float m);

#endif
