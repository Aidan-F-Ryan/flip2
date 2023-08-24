#include "particles.hpp"

int main(){
    Particles particles(1<<19);
    particles.setDomain(-100.0f, -100.0f, -100.0f, 100, 100, 100, 2.0f);
    particles.randomizeParticlePositions();
    particles.alignParticlesToGrid();

    return 0;
}