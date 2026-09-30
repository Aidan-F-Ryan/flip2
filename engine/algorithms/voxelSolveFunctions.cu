//Copyright 2023 Aberrant Behavior LLC

#include "voxelSolveFunctions.hu"
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
    cudaStreamSynchronize(stream);
}

void removeGravity(const CudaVec<char>& solids, CudaVec<float>& voxelsUy, float dt, cudaStream_t stream){
    removeGravityKernel<<<voxelsUy.size() / WORKSIZE + 1, WORKSIZE, 0, stream>>>(voxelsUy.size(), dt, solids.devPtr(), voxelsUy.devPtr());
    cudaStreamSynchronize(stream);
}

//P2G accumulated weighted velocities and weights into each owning voxel; turn them into velocities, walls stay at rest
__global__ void normalizeVoxelVelocities(uint numUsedVoxels, const char* solids, const float* weightsX, const float* weightsY, const float* weightsZ, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        ux[index] = solids[index] ? 0.0f : ux[index] / (weightsX[index] + 0.0000001f);
        uy[index] = solids[index] ? 0.0f : uy[index] / (weightsY[index] + 0.0000001f);
        uz[index] = solids[index] ? 0.0f : uz[index] / (weightsZ[index] + 0.0000001f);
    }
}

void cudaNormalizeVoxelVelocities(const CudaVec<char>& solids, const CudaVec<float>& voxelWeightsX, const CudaVec<float>& voxelWeightsY, const CudaVec<float>& voxelWeightsZ, CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz, cudaStream_t stream){
    normalizeVoxelVelocities<<<solids.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solids.size(), solids.devPtr(), voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
}

//P2G leaves the faces no particle reaches at 0, and on the edge of the fluid the pressure solve would then drag on every surface moving outward, speeding
//those faces back up each step. Give an unknown's unreached face the average of the reached faces of the same component around it, and its unreached
//upper face, which sits on air, its own face's value. Only unreached faces are written and only reached ones read by other threads, so this is race free
__device__ inline void extrapolateFace(uint index, uint upper, const uint* const neighbors[6], const char* solveCodes, const float* weights, float* u){
    if(weights[index] == 0.0f){
        float sum = 0.0f;
        int count = 0;
        #pragma unroll
        for(int face = 0; face < 6; ++face){
            uint neighbor = neighbors[face][index];
            if(neighbor < WALL_VOXEL && weights[neighbor] > 0.0f){
                sum += u[neighbor];
                ++count;
            }
        }
        if(count){
            u[index] = sum / count;
        }
    }
    if(upper < WALL_VOXEL && !solveCodes[upper] && weights[upper] == 0.0f){
        u[upper] = u[index];
    }
}

__global__ void extrapolateUnreachedFaces(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                            const float* weightsX, const float* weightsY, const float* weightsZ, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && solveCodes[index]){
        const uint* const neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
        extrapolateFace(index, neighborPx[index], neighbors, solveCodes, weightsX, ux);
        extrapolateFace(index, neighborPy[index], neighbors, solveCodes, weightsY, uy);
        extrapolateFace(index, neighborPz[index], neighbors, solveCodes, weightsZ, uz);
    }
}

void cudaExtrapolateUnreachedFaces(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& voxelWeightsX, const CudaVec<float>& voxelWeightsY, const CudaVec<float>& voxelWeightsZ, CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz, cudaStream_t stream){
    extrapolateUnreachedFaces<<<solveCodes.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solveCodes.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
        voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
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
            if(level == 1 && !deeper){
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
void cudaFindFootprintDepth(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    CudaVec<char>& depth, CudaVec<uint>& freeSurface, cudaStream_t stream){
    freeSurface.zeroDeviceAsync(stream);
    for(char level = 1; level < CORRECTION_DEPTH; ++level){
        deepenFootprint<<<solveCodes.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(solveCodes.size(), level, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
            depth.devPtr(), freeSurface.devPtr());
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
        residuals[index] = solveCodes[index] ? -divU[index] - Adiag[index]*p[index] - offDiagonalSum(index, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, p) : 0.0f;
    }
}

//red/black SOR until the largest residual is below tolerance times the largest divergence; returns that relative residual
float cudaGSiteration(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, cudaStream_t stream){
    float w = 1.9f;
    uint batchCheckEvery = 16;
    uint numUsedVoxels = p.size();
    float maxDivergence = std::abs(divU.getMax(stream, true));
    if(maxDivergence == 0.0f){
        return 0.0f;
    }
    float maxResidual = 1.0f;
    for(uint iteration = 1; iteration <= maxIterations; ++iteration){
        for(char color = 1; color <= 2; ++color){  //red, then black
            GSiteration<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), color, neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
                Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), divU.devPtr(), p.devPtr(), w);
        }
        if(iteration % batchCheckEvery == 0){
            pressureResiduals<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
                Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), divU.devPtr(), p.devPtr(), residuals.devPtr());
            float previousResidual = maxResidual;
            maxResidual = std::abs(residuals.getMax(stream, true)) / maxDivergence;
            if(maxResidual < tolerance || std::abs(previousResidual - maxResidual) < tolerance / 1000){  //converged, or stalled
                break;
            }
        }
    }
    return maxResidual;
}

//an unknown updates the faces it's responsible for in one dimension: its own negative face, u -= scale*(p - lower neighbour's p) (held at 0 on a wall),
//and its upper face when the voxel storing that face isn't an unknown (air, at p = 0)
__device__ inline void accelerateFaces(uint index, uint lowerNeighbor, uint upperNeighbor, const char* solveCodes, const float* p, float* u, float scale){
    if(lowerNeighbor == WALL_VOXEL){
        u[index] = 0.0f;
    }
    else{
        u[index] -= scale*(p[index] - (lowerNeighbor == NO_VOXEL ? 0.0f : p[lowerNeighbor]));
    }
    if(upperNeighbor < WALL_VOXEL && !solveCodes[upperNeighbor]){
        u[upperNeighbor] += scale*p[index];
    }
}

__global__ void pressureToAcceleration(uint numUsedVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz,
                                        const float* p, float* ux, float* uy, float* uz, float scale){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && solveCodes[index]){
        accelerateFaces(index, neighborNx[index], neighborPx[index], solveCodes, p, ux, scale);
        accelerateFaces(index, neighborNy[index], neighborPy[index], solveCodes, p, uy, scale);
        accelerateFaces(index, neighborNz[index], neighborPz[index], solveCodes, p, uz, scale);
    }
}

void cudaVelocityUpdate(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz, const CudaVec<float>& p,
    CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz, float scale, cudaStream_t stream){
    pressureToAcceleration<<<p.size() / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(p.size(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(),
        p.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), scale);
    cudaStreamSynchronize(stream);
}
