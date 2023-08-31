#include "testing.h"

int main(){
    ParticleSystemTester particles(
        // 33
        // (1<<19) - 24//512K
        (1<<25) - 12 //16M
        );
    particles.setDomain(-100.0f, -100.0f, -100.0f, 100, 100, 100, 2.0f);
    particles.randomizeParticlePositions();
    particles.run();
    // particles.runVerify();

    return 0;
}