//Copyright 2023 Aberrant Behavior LLC

#include "voxelSolveFunctions.hu"
#include "multigridFunctions.hu"    //Stencil: an unknown's row, and its product
#include <cmath>

__global__ void applyGravityKernel(uint numUsedVoxelsInGrid, const float dt, const char* solids, float* voxelsUz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxelsInGrid){
        if(!solids[index]){
            voxelsUz[index] += -9.8f*dt;
        }
    }
}
__global__ void removeGravityKernel(uint numUsedVoxelsInGrid, const float dt, const char* solids, float* voxelsUz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxelsInGrid){
        if(!solids[index]){
            voxelsUz[index] -= -9.8f*dt;
        }
    }
}

void applyGravity(const CudaVec<char>& solids, CudaVec<float>& voxelsUy, float dt, cudaStream_t stream){
    applyGravityKernel<<<voxelsUy.size() / WORKSIZE + 1, WORKSIZE, 0, stream>>>(voxelsUy.size(), dt, solids.devPtr(), voxelsUy.devPtr());
}

void removeGravity(const CudaVec<char>& solids, CudaVec<float>& voxelsUy, float dt, cudaStream_t stream){
    removeGravityKernel<<<voxelsUy.size() / WORKSIZE + 1, WORKSIZE, 0, stream>>>(voxelsUy.size(), dt, solids.devPtr(), voxelsUy.devPtr());
}

//P2G accumulated weights and weighted velocities into each owning voxel; turn them into velocities, walls stay at rest
//P2G leaves 32-bit fixed-point sums in these arrays' storage: turn the weights and counts into floats, and each face's momentum into its velocity.
//The momentum is each particle's whole share of the weight times its velocity in whole units, so over the weight it's the mean of their velocities
//exactly, and the division keeps that: the whole units first, in integers, then what's left over. (A double holds neither sum of a heavy face
//exactly, and their quotient alone would come out a rounding either side of the mean; and a double's division is slower here than a 64-bit
//integer's: 71 us for this kernel against 52, at 370k particles.) A face every particle reaches at one velocity gets that velocity, to the bit,
//whatever its weight; one no share reaches has none (0, until it's extrapolated)
__global__ void normalizeVoxelVelocities(uint numUsedVoxels, const char* solids, float* weightsX, float* weightsY, float* weightsZ, float* ux, float* uy, float* uz,
                                         const unsigned int* lowX, const unsigned int* lowY, const unsigned int* lowZ, float* particleCounts, double unscaleVelocity,
                                         float unscaleWeight){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        float* weights[3] = {weightsX, weightsY, weightsZ};
        float* velocities[3] = {ux, uy, uz};
        const unsigned int* lows[3] = {lowX, lowY, lowZ};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            int shares = ((int*)weights[dim])[index];
            long long momentum = ((int*)velocities[dim])[index]*(1LL << MOMENTUM_DIGIT) + lows[dim][index];
            float velocity = 0.0f;
            if(shares > 0 && !solids[index]){
                long long whole = momentum / shares;
                velocity = (float)(((double)whole + (double)(momentum - whole*shares) / shares)*unscaleVelocity);
            }
            weights[dim][index] = shares*unscaleWeight;
            velocities[dim][index] = velocity;
        }
        particleCounts[index] = ((int*)particleCounts)[index];
    }
}

void cudaNormalizeVoxelVelocities(const CudaVec<char>& solids, CudaVec<float>& voxelWeightsX, CudaVec<float>& voxelWeightsY, CudaVec<float>& voxelWeightsZ, CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz,
    const CudaVec<float>& lowX, const CudaVec<float>& lowY, const CudaVec<float>& lowZ, CudaVec<float>& particleCounts, float velocityScale, float weightScale, cudaStream_t stream){
    normalizeVoxelVelocities<<<solids.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solids.size(), solids.devPtr(), voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(),
        (const unsigned int*)lowX.devPtr(), (const unsigned int*)lowY.devPtr(), (const unsigned int*)lowZ.devPtr(), particleCounts.devPtr(), 1.0/(double)velocityScale, 1.0f/weightScale);
}

//P2G leaves the faces no particle reaches at 0, and on the edge of the fluid the pressure solve would then drag on every surface moving outward, speeding
//those faces back up each step. Give an unknown's unreached face the average of the faces of the same component around it that have a velocity: the
//reached ones, and then, in the passes after the first, the ones the passes before it filled. The unknowns are every voxel of the 3^3 round a
//particle's own, which is further than its weights go: a face of one can be three steps, one along each axis, from the nearest face any particle
//reaches, so it takes EXTRAPOLATION_PASSES to give every one a velocity. (One pass left 7 to 14 faces round a ball 8 voxels across at rest each
//substep, each a voxel's worth of divergence to the solve: the ball's particles' velocities spread by half a percent of its speed in a quarter of
//a second.) Then an unreached face on top of an unknown, which an air voxel stores, takes the unknown's own face value (extrapolateAirFaces, once
//the unknowns are done).
//marks says which faces have a velocity, two bits a component: 1 for one that has, 2 for one this pass has just given one, which settleFaceMarks
//makes 1 for the next. A thread writes its own voxel's faces and marks, and reads only faces marked 1, which nothing is writing
static const int EXTRAPOLATION_PASSES = 3;

__global__ void extrapolateUnreachedFaces(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                            const float* weightsX, const float* weightsY, const float* weightsZ, bool first, unsigned char* marks, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        const uint* const neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
        const float* weights[3] = {weightsX, weightsY, weightsZ};
        float* velocities[3] = {ux, uy, uz};
        unsigned char mine = first ? 0 : marks[index];
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            if(first && weights[dim][index] > 0.0f){
                mine |= 1 << 2*dim;
            }
            if(solveCodes[index] && !(mine >> 2*dim & 3)){
                float sum = 0.0f;
                int count = 0;
                #pragma unroll
                for(int face = 0; face < 6; ++face){
                    uint neighbor = neighbors[face][index];
                    if(neighbor < WALL_VOXEL && (first ? weights[dim][neighbor] > 0.0f : (marks[neighbor] >> 2*dim & 3) == 1)){
                        sum += velocities[dim][neighbor];
                        ++count;
                    }
                }
                if(count){
                    velocities[dim][index] = sum / count;
                    mine |= 2 << 2*dim;
                }
            }
        }
        marks[index] = mine;
    }
}

__global__ void settleFaceMarks(uint numUsedVoxels, unsigned char* marks){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        marks[index] = (marks[index] | marks[index] >> 1) & 0x15;
    }
}

//an air voxel above an unknown, whose face between them no particle reached, takes the unknown's (extrapolated) face value. The voxel updates its own
//face from its lower neighbour's, rather than the unknown writing it, so each voxel's values are computed where the voxel is stored
__device__ inline void extrapolateAirFace(uint index, uint lower, const float* weights, float* u){
    if(lower < WALL_VOXEL && weights[index] == 0.0f){
        u[index] = u[lower];
    }
}

__global__ void extrapolateAirFaces(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz,
                                    const float* weightsX, const float* weightsY, const float* weightsZ, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && !solveCodes[index]){
        extrapolateAirFace(index, neighborNx[index], weightsX, ux);
        extrapolateAirFace(index, neighborNy[index], weightsY, uy);
        extrapolateAirFace(index, neighborNz[index], weightsZ, uz);
    }
}

//ghost voxels take their owners' faces after each pass: the air faces pull from unknowns that may be ghosts, and the next steps read ghost faces
void cudaExtrapolateUnreachedFaces(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& voxelWeightsX, const CudaVec<float>& voxelWeightsY, const CudaVec<float>& voxelWeightsZ, CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz,
    PartitionContext& context, cudaStream_t stream){
    if(solveCodes.size() == 0){
        return;
    }
    unsigned char* marks;
    gpuErrchk(cudaMallocAsync((void**)&marks, solveCodes.size(), stream));
    for(int pass = 0; pass < EXTRAPOLATION_PASSES; ++pass){
        extrapolateUnreachedFaces<<<solveCodes.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solveCodes.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
            voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), pass == 0, marks, voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
        for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){
            context.fillGhosts(velocity->devPtr(), stream);
        }
        if(pass + 1 < EXTRAPOLATION_PASSES){    //and the ghosts' marks, as their owners have them, for the next pass
            settleFaceMarks<<<solveCodes.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solveCodes.size(), marks);
            context.fillGhosts((char*)marks, stream);
        }
    }
    gpuErrchk(cudaFreeAsync(marks, stream));
    extrapolateAirFaces<<<solveCodes.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solveCodes.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(),
        voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
    for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){
        context.fillGhosts(velocity->devPtr(), stream);
    }
}

//one pass of measuring how far each voxel is inside the fluid's footprint: level 1 gives unknowns depth 1, or 2 if all six neighbours are unknowns too, and
//flags whether any unknown touches air. Each later level deepens the voxels at that depth whose neighbours all are (walls count as deep). Depths only
//grow, so reading a neighbour mid-pass is safe
__global__ void deepenFootprint(uint numUsedVoxels, char level, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                char* depth, uint* freeSurface){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        char current = level == 1 ? solveCodes[index] != 0 : depth[index];
        if(current == level){
            const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
            bool deeper = true;
            #pragma unroll
            for(int face = 0; face < 6; ++face){
                uint neighbor = neighbors[face][index];
                char neighborDepth = neighbor == WALL_VOXEL ? level : neighbor == NO_VOXEL ? 0 : level == 1 ? solveCodes[neighbor] != 0 : depth[neighbor];
                deeper = deeper && neighborDepth >= level;
            }
            if(level == 1 && !deeper && solveCodes[index] <= 2){    //a ghost's (3, 4) edge is just where this partition's copy stops
                *freeSurface = 1;
            }
            current += deeper;
        }
        if(level == 1 || current > level){
            depth[index] = current;
        }
    }
}

//the footprint reaches about 2 voxels past the particles, so from depth CORRECTION_DEPTH on, a voxel and its six neighbours sit a voxel or more inside
//them, clear of the free surface's partly filled voxels. Holes in the particles are still footprint, so they don't stop the density correction
//each level reads its neighbours' depths from the one before, so the ghosts take their owners' after every level
void cudaFindFootprintDepth(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    CudaVec<char>& depth, CudaVec<uint>& freeSurface, PartitionContext& context, cudaStream_t stream){
    freeSurface.zeroDeviceAsync(stream);
    for(char level = 1; level < CORRECTION_DEPTH; ++level){
        deepenFootprint<<<solveCodes.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solveCodes.size(), level, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
            depth.devPtr(), freeSurface.devPtr());
        context.fillGhosts(depth.devPtr(), stream);
    }
}

//velocities sit on each voxel's negative faces, so an unknown's divergence reads its own three faces and its upper neighbours' (owning) copies;
//wall faces carry no flow and faces nothing stores read as 0
__global__ void calculateDivU(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                const float* ux, const float* uy, const float* uz, const float* particleCounts, const char* footprintDepth, float restParticlesPerVoxel, float correctionRate, float* divU){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        float divergence = 0.0f;
        if(solveCodes[index]){
            uint upperX = neighborPx[index];
            uint upperY = neighborPy[index];
            uint upperZ = neighborPz[index];
            divergence = (upperX < WALL_VOXEL ? ux[upperX] : 0.0f) - (neighborNx[index] == WALL_VOXEL ? 0.0f : ux[index])
                       + (upperY < WALL_VOXEL ? uy[upperY] : 0.0f) - (neighborNy[index] == WALL_VOXEL ? 0.0f : uy[index])
                       + (upperZ < WALL_VOXEL ? uz[upperZ] : 0.0f) - (neighborNz[index] == WALL_VOXEL ? 0.0f : uz[index]);
        }
        if(footprintDepth[index] >= CORRECTION_DEPTH){  //the solve can't see particles bunching up or spreading out, so ask it for the divergence that restores the rest density
            const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
            //the voxel weighs half and its neighbours 1/12 each: an even average over the 7 reads a checkerboard of dense and sparse voxels with its sign
            //flipped, so the correction would feed it and clump particles into every other voxel. These weights never flip a pattern's sign
            float count = 6.0f*particleCounts[index];
            float weight = 6.0f;
            #pragma unroll
            for(int face = 0; face < 6; ++face){
                uint neighbor = neighbors[face][index];
                if(neighbor < WALL_VOXEL){
                    count += particleCounts[neighbor];
                    weight += 1.0f;
                }
            }
            divergence -= (count / (weight*restParticlesPerVoxel) - 1.0f)*correctionRate;   //spread a denser than rest voxel, gather a sparser one
        }
        divU[index] = divergence;
    }
}

void cudaCalcDivU(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& voxelsUx, const CudaVec<float>& voxelsUy, const CudaVec<float>& voxelsUz, const CudaVec<float>& particleCounts, const CudaVec<char>& footprintDepth, float restParticlesPerVoxel, float correctionRate,
    CudaVec<float>& divU, cudaStream_t stream){
    calculateDivU<<<divU.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(divU.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
        voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), particleCounts.devPtr(), footprintDepth.devPtr(), restParticlesPerVoxel, correctionRate, divU.devPtr());
}

//one face of an unknown's pressure equation: -scale towards a neighbouring unknown, and scale on the diagonal unless the face is a wall (anything else is air, pinned at p = 0)
__device__ inline float faceCoefficient(uint neighbor, const char* solveCodes, float scale, float& diagonal){
    diagonal += neighbor == WALL_VOXEL ? 0.0f : scale;
    return neighbor < WALL_VOXEL && solveCodes[neighbor] ? -scale : 0.0f;
}

//coefficients of each unknown's pressure equation, Adiag*p + sum(A*neighbour p) = -divU
__global__ void generateA(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                            float* Anx, float* Apx, float* Any, float* Apy, float* Anz, float* Apz, float* Adiag, float scale){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        float diagonal = 0.0f;
        bool unknown = solveCodes[index];
        Anx[index] = unknown ? faceCoefficient(neighborNx[index], solveCodes, scale, diagonal) : 0.0f;
        Apx[index] = unknown ? faceCoefficient(neighborPx[index], solveCodes, scale, diagonal) : 0.0f;
        Any[index] = unknown ? faceCoefficient(neighborNy[index], solveCodes, scale, diagonal) : 0.0f;
        Apy[index] = unknown ? faceCoefficient(neighborPy[index], solveCodes, scale, diagonal) : 0.0f;
        Anz[index] = unknown ? faceCoefficient(neighborNz[index], solveCodes, scale, diagonal) : 0.0f;
        Apz[index] = unknown ? faceCoefficient(neighborPz[index], solveCodes, scale, diagonal) : 0.0f;
        Adiag[index] = diagonal;
    }
}

void cudaGetA(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    CudaVec<float>& Anx, CudaVec<float>& Apx, CudaVec<float>& Any, CudaVec<float>& Apy, CudaVec<float>& Anz, CudaVec<float>& Apz, CudaVec<float>& Adiag, float scale, cudaStream_t stream){
    generateA<<<Adiag.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(Adiag.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
        Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), scale);
}

//one face's term of the off-diagonal sum; a zero coefficient (air, walls) never touches the neighbour
__device__ inline float facePressure(uint index, const float* A, const uint* neighbor, const float* p){
    return A[index] != 0.0f ? A[index]*p[neighbor[index]] : 0.0f;
}

__device__ inline float offDiagonalSum(uint index, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                        const float* Anx, const float* Apx, const float* Any, const float* Apy, const float* Anz, const float* Apz, const float* p){
    return facePressure(index, Anx, neighborNx, p) + facePressure(index, Apx, neighborPx, p)
         + facePressure(index, Any, neighborNy, p) + facePressure(index, Apy, neighborPy, p)
         + facePressure(index, Anz, neighborNz, p) + facePressure(index, Apz, neighborPz, p);
}

//one red or black SOR half-sweep over the unknowns. Same-coloured voxels never share a face, so updating in place is race free
__global__ void GSiteration(uint numUsedVoxels, const char* solveCodes, char color, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                            const float* Anx, const float* Apx, const float* Any, const float* Apy, const float* Anz, const float* Apz, const float* Adiag, const float* divU, float* p, float w){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && solveCodes[index] == color){
        float sum = offDiagonalSum(index, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, p);
        p[index] += w*((-divU[index] - sum)/(Adiag[index] + 0.000000001f) - p[index]);
    }
}

__global__ void pressureResiduals(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                    const float* Anx, const float* Apx, const float* Any, const float* Apy, const float* Anz, const float* Apz, const float* Adiag, const float* divU, const float* p, float* residuals){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        Stencil A = {{neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz}, {Anx, Apx, Any, Apy, Anz, Apz}, Adiag};
        residuals[index] = solveCodes[index] ? -divU[index] - A.rowTimes(index, p) : 0.0f;    //the row as differences across its faces, as CG takes it
    }
}

//red/black SOR until the largest residual is below tolerance times the largest divergence; returns that relative residual
float cudaGSiteration(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, uint numOwnVoxels, PartitionContext& context, cudaStream_t stream){
    float w = 1.9f;
    uint batchCheckEvery = 16;
    uint numUsedVoxels = p.size();
    float maxDivergence = (float)context.maxOverPartitions(std::abs(divU.getMax(stream, true, numOwnVoxels)));   //over this partition's own voxels, then every partition's
    if(maxDivergence == 0.0f){
        return 0.0f;
    }
    float maxResidual = 1.0f;
    SolveProgress progress;
    for(uint iteration = 1; iteration <= maxIterations; ++iteration){
        for(char color = 1; color <= 2; ++color){  //red, then black
            GSiteration<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), color, neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
                Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), divU.devPtr(), p.devPtr(), w);
            context.fillGhosts(p.devPtr(), stream);     //the other colour's next half-sweep reads the ghosts' pressures
        }
        if(iteration % batchCheckEvery == 0){
            pressureResiduals<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
                Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), divU.devPtr(), p.devPtr(), residuals.devPtr());
            maxResidual = (float)context.maxOverPartitions(std::abs(residuals.getMax(stream, true, numOwnVoxels))) / maxDivergence;   //a float division, as before
            if(progress.done(maxResidual, tolerance)){  //converged, or stalled
                break;
            }
        }
    }
    return maxResidual;
}

//every voxel updates the faces it stores, its negative faces, in one dimension. An unknown's: u -= scale*(p - lower neighbour's p), held at 0 on a wall.
//An air voxel's above an unknown: the same, with the air's p = 0, so u += scale*(the unknown's p). Each voxel's values are computed where it's stored,
//from its lower neighbours', never written from next door
__device__ inline void accelerateOwnFace(uint index, uint lowerNeighbor, const float* p, float* u, float scale){
    if(lowerNeighbor == WALL_VOXEL){
        u[index] = 0.0f;
    }
    else{
        u[index] -= scale*(p[index] - (lowerNeighbor == NO_VOXEL ? 0.0f : p[lowerNeighbor]));
    }
}

__device__ inline void accelerateAirFace(uint lowerUnknown, uint index, const float* p, float* u, float scale){
    if(lowerUnknown < WALL_VOXEL){
        u[index] += scale*p[lowerUnknown];
    }
}

__global__ void pressureToAcceleration(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz,
                                        const float* p, float* ux, float* uy, float* uz, float scale){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        if(solveCodes[index]){
            accelerateOwnFace(index, neighborNx[index], p, ux, scale);
            accelerateOwnFace(index, neighborNy[index], p, uy, scale);
            accelerateOwnFace(index, neighborNz[index], p, uz, scale);
        }
        else{   //the lower neighbours only name an unknown if the face between is this voxel's
            accelerateAirFace(neighborNx[index], index, p, ux, scale);
            accelerateAirFace(neighborNy[index], index, p, uy, scale);
            accelerateAirFace(neighborNz[index], index, p, uz, scale);
        }
    }
}

void cudaVelocityUpdate(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz, const CudaVec<float>& p,
    CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz, float scale, cudaStream_t stream){
    pressureToAcceleration<<<p.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(p.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(),
        p.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), scale);
}
