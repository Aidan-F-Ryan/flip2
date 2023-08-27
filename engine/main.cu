#include "testing.h"

int main(){
    ParticleSystemTester particles(
        // 33
        1<<19
        // 1<<23
        );
    particles.setDomain(-100.0f, -100.0f, -100.0f, 100, 100, 100, 2.0f);
    particles.randomizeParticlePositions();
    particles.run();

    return 0;
}