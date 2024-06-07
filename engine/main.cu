//Copyright 2023 Aberrant Behavior LLC

#include "testing.h"
#include <string>

int main(){
    ParticleSystemTester particles(
        // 33
        // 1024
        (1<<19) - 24//512K
        // (1<<24) - 13 //16M
        // (1<<25) - 12 //32M
        // 1<<26
        );
    // particles.setDomain(-100.0f, -100.0f, -100.0f, 256, 256, 256, 200.0f / 256.0f);
    particles.setDomain(-10.0, -10.0, -10.0, 100, 100, 100, 0.25);
    // particles.setDomain(-100.0f, -100.0f, -100.0f, 1024, 1024, 1024, 200.0f / 1024.0f);
    particles.randomizeParticlePositions();
    particles.initialize();
    particles.writePositionsToFile(std::to_string(0));
    for(int i = 0; i < 300; ++i){
        // particles.run();
        particles.solveFrame(24.0f);
        particles.writePositionsToFile(std::to_string(i+1));
    }
    return 0;
}