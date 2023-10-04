//Copyright 2023 Aberrant Behavior LLC

#include "testing.h"

int main(){
    ParticleSystemTester particles(
        // 33
        // (1<<19) - 24//512K
        (1<<24) - 13 //16M
        // (1<<25) - 12 //32M
        // 1<<26
        );
    particles.setDomain(-100.0f, -100.0f, -100.0f, 256, 256, 256, 200.0f / 256.0f);
    particles.randomizeParticlePositions();
    particles.run();
    // particles.runVerify();

    return 0;
}