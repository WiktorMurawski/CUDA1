#include "particles_t.h"

struct gpu_particles_handle_t{
    particles_t * d_struct; // wskaźnik do struktury GPU
    float * d_x,* d_y,* d_vx,* d_vy,* d_ax,* d_ay,* d_q,* d_m;
};