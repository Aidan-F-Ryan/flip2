//Copyright 2023 Aberrant Behavior LLC

#include "testing.h"
#include "tcpTransport.hu"
#ifdef FLIP2_WITH_NCCL
#include "ncclTransport.hu"
#endif
#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char** argv){
    //main [frames] [flipRatio] [densityCorrectionTime] [nodes | tank [height] [swirl]]
    int numFrames = argc > 1 ? std::stoi(argv[1]) : 300;    //24 fps
    double flipRatio = argc > 2 ? std::stod(argv[2]) : 0.95;    //the particle velocity update's share of FLIP, the rest PIC
    double densityCorrectionTime = argc > 3 ? std::stod(argv[3]) : 0.1;   //seconds to pull the particle density back to rest; 0 turns it off
    //"tank" runs a 0.25 m tank at the dam break's resolution instead, for checking the particle/grid transfers:
    //filled to height of its 32 voxels (default full), at rest or swirling at up to swirl m/s
    bool tank = argc > 4 && std::string(argv[4]) == "tank";
    uint height = argc > 5 ? std::stoi(argv[5]) : 32;
    uint nodes = argc > 4 && !tank ? std::stoi(argv[4]) : 32;    //the dam break's grid, nodes per side of its 1 m cube (4 voxels each)
    int partitions = std::getenv("FLIP2_PARTITIONS") ? std::stoi(std::getenv("FLIP2_PARTITIONS")) : 1;  //FLIP2_PARTITIONS splits the domain along z
    int devices = std::getenv("FLIP2_DEVICES") ? std::stoi(std::getenv("FLIP2_DEVICES")) : 0;  //FLIP2_DEVICES spreads the partitions over that many GPUs; every GPU by default
    //FLIP2_WORLD_SIZE over 1 makes this process rank FLIP2_RANK of that many, each one partition on GPU FLIP2_DEVICE (default 0), meeting through rank 0
    //at FLIP2_RENDEZVOUS (host:port) and joined over FLIP2_TRANSPORT: tcp (the default) or nccl, one GPU per rank, if flip2 was built with NCCL. Setting
    //FLIP2_TRANSPORT alone runs one such rank, to try the transport out. tools/launch-local.sh starts them
    int worldSize = std::getenv("FLIP2_WORLD_SIZE") ? std::stoi(std::getenv("FLIP2_WORLD_SIZE")) : 1;
    std::string transport = std::getenv("FLIP2_TRANSPORT") ? std::getenv("FLIP2_TRANSPORT") : "";
    uint numParticles = tank ? 32*32*height*8 :    //~8 per fluid voxel
        // 33
        // 1024
        // (1<<19) - 24//512K
        40*nodes*nodes*nodes;   //~8.4 per fluid voxel: 1.3M on 32^3 nodes
        // (1<<24) - 13 //16M
        // (1<<25) - 12 //32M
        // 1<<26
    std::unique_ptr<ParticleSystemTester> tester;
    if(worldSize > 1 || !transport.empty()){
        const char* rank = std::getenv("FLIP2_RANK");
        const char* rendezvous = std::getenv("FLIP2_RENDEZVOUS");
        if(rank == nullptr || rendezvous == nullptr){
            std::cerr<<"running as a rank needs FLIP2_RANK and FLIP2_RENDEZVOUS too\n";
            return 1;
        }
        int device = std::getenv("FLIP2_DEVICE") ? std::stoi(std::getenv("FLIP2_DEVICE")) : 0;
        std::unique_ptr<Transport> link;
        if(transport.empty() || transport == "tcp"){
            link = std::make_unique<TcpTransport>(std::stoi(rank), worldSize, rendezvous);
        }
        else if(transport == "nccl"){
#ifdef FLIP2_WITH_NCCL
            link = std::make_unique<NcclTransport>(std::stoi(rank), worldSize, rendezvous);
#else
            std::cerr<<"FLIP2_TRANSPORT=nccl, but this build has no NCCL: build with NCCL_HOME set to where it's installed\n";
            return 1;
#endif
        }
        else{
            std::cerr<<"FLIP2_TRANSPORT is tcp or nccl, not "<<transport<<"\n";
            return 1;
        }
        tester = std::make_unique<ParticleSystemTester>(numParticles, std::move(link), device);
    }
    else{
        tester = std::make_unique<ParticleSystemTester>(numParticles, partitions, devices);
    }
    ParticleSystemTester& particles = *tester;
    if(tank){
        particles.setDomain(0.0f, 0.0f, 0.0f, 8, 8, 8, 1.0f / 32.0f);
        particles.randomizeParticlePositions(make_double3(0.0, 0.0, 0.0), make_double3(1.0, height / 32.0, 1.0), argc > 6 ? std::stod(argv[6]) : 0.0);
    }
    else{
        // particles.setDomain(-100.0f, -100.0f, -100.0f, 256, 256, 256, 200.0f / 256.0f);
        // particles.setDomain(-100.0f, -100.0f, -100.0f, 32, 32, 32, 200.0f / 32.0f);
        particles.setDomain(-0.5f, -0.5f, -0.5f, nodes, nodes, nodes, 1.0f / nodes);    //1 m cube
        // particles.setDomain(-10.0, -10.0, -10.0, 100, 100, 100, 0.25);
        // particles.setDomain(-100.0f, -100.0f, -100.0f, 1024, 1024, 1024, 200.0f / 1024.0f);
        particles.randomizeParticlePositions();
    }
    particles.setFlipRatio(flipRatio);
    particles.setDensityCorrectionTime(densityCorrectionTime);
    if(const char* solver = std::getenv("FLIP2_SOLVER")){    //FLIP2_SOLVER picks the pressure solver: multigrid (the default), sor, cg or jacobi
        std::string name = solver;
        particles.setPressureSolver(name == "sor" ? PressureSolver::sor : name == "cg" ? PressureSolver::cg : name == "jacobi" ? PressureSolver::jacobi : PressureSolver::multigrid);
    }
    if(const char* cfl = std::getenv("FLIP2_CFL")){  //FLIP2_CFL sets how many voxels the fastest particle may move per substep (default 4)
        particles.setCfl(std::stod(cfl));
    }
    if(const char* dots = std::getenv("FLIP2_DOTS")){    //FLIP2_DOTS=blocks adds CG's dot products per thread block, as before 2026-09-30: repeatable only for one split of the nodes between GPUs
        particles.setDotProductSums(std::string(dots) == "blocks" ? DotProductSums::perBlock : DotProductSums::exact);
    }
    if(const char* advection = std::getenv("FLIP2_ADVECTION")){  //FLIP2_ADVECTION=euler moves particles straight along the grid velocity instead of RK3
        particles.setRungeKutta3(std::string(advection) != "euler");
    }
    if(const char* transfer = std::getenv("FLIP2_TRANSFER")){    //FLIP2_TRANSFER=apic carries each particle's velocity gradient (flipRatio still blends FLIP in)
        particles.setApic(std::string(transfer) == "apic");
    }
    if(const char* surface = std::getenv("FLIP2_FREE_SURFACE")){     //FLIP2_FREE_SURFACE=sharp puts the pressure solve's surface at the level set's (freesurface.cu)
        particles.setFreeSurface(std::string(surface) == "sharp" ? FreeSurface::sharp : FreeSurface::footprint);
    }
    //FLIP2_DIAGNOSTICS=file writes a line of JSON per frame there: the particles' energy, momentum and extent, how they fill the voxels, and a hash of
    //every particle's state, which tools/golden.py compares between runs (see diagnostics.hu)
    const char* diagnostics = std::getenv("FLIP2_DIAGNOSTICS");
    particles.initialize();
    if(diagnostics){
        particles.writeDiagnostics(diagnostics, 0);
    }
    particles.writePositionsToFile(std::to_string(0) + ".bin");
    for(int i = 0; i < numFrames; ++i){
        // particles.run();
        particles.solveFrame(24.0f);
        if(diagnostics){
            particles.writeDiagnostics(diagnostics, i + 1);
        }
        particles.writePositionsToFile(std::to_string(i+1) + ".bin");
    }
    return 0;
}
