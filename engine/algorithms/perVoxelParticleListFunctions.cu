//Copyright 2023 Aberrant Behavior LLC

#include "perVoxelParticleListFunctions.hu"
#include "parallelPrefixSumKernels.hu"

__global__ void sumParticlesPerNode(uint numGridNodes, uint numParticles, uint* gridNodeIndicesToFirstParticleIndex, uint* subCellsTouchedPerParticle,
        uint* subCellsDim, uint* numNonZeroVoxels, uint* numParticlesInVoxelLists, uint numVoxelsPerNode){
    extern __shared__ uint voxelCount[];
    __shared__ uint sums[WORKSIZE];
    __shared__ uint nonZeroVoxels[WORKSIZE];
    __shared__ uint maxParticleNum;

    if(threadIdx.x == 0){
        if(blockIdx.x < numGridNodes - 1){
            maxParticleNum = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
        else{
            maxParticleNum = numParticles;
        }
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        voxelCount[i] = 0;
    }

    __syncthreads();

    for(uint i = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; i < maxParticleNum; i += blockDim.x){
        uint firstSubCellsTouched;
        if(i == 0){
            firstSubCellsTouched = 0;
        }
        else{
            firstSubCellsTouched = subCellsTouchedPerParticle[i-1];
        }
        for(uint j = firstSubCellsTouched; j < subCellsTouchedPerParticle[i]; ++j){
            atomicAdd(voxelCount + subCellsDim[j], 1);
        }
    }

    sums[threadIdx.x] = 0;
    sums[threadIdx.x + blockDim.x] = 0;
    nonZeroVoxels[threadIdx.x] = 0;
    nonZeroVoxels[threadIdx.x + blockDim.x] = 0;
    __syncthreads();

    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sums[i % WORKSIZE] += voxelCount[i];
        if(voxelCount[i] != 0){
            ++nonZeroVoxels[i % WORKSIZE];
        }
    }

    for(uint i = blockDim.x; i > 0; i>>=1){
        __syncthreads();
        if(threadIdx.x < i){
            sums[threadIdx.x] += sums[threadIdx.x + i];
            nonZeroVoxels[threadIdx.x] += nonZeroVoxels[threadIdx.x + i];
        }
    }

    if(threadIdx.x == 0){
        numParticlesInVoxelLists[blockIdx.x] = sums[0];
        numNonZeroVoxels[blockIdx.x] = nonZeroVoxels[0];
    }
}

void cudaSumParticlesPerNodeAndWriteNumUsedVoxels(uint numGridNodes, uint numParticles, uint* gridNodeIndicesToFirstParticleIndex, uint* subCellsTouchedPerParticle, uint* subCellsDim, uint* numNonZeroVoxels,
    uint* numParticlesInVoxelLists, uint numVoxelsPerNode, cudaStream_t stream){
    sumParticlesPerNode<<<numGridNodes, BLOCKSIZE, sizeof(uint) * numVoxelsPerNode, stream>>>(
        numGridNodes, numParticles, gridNodeIndicesToFirstParticleIndex, subCellsTouchedPerParticle, subCellsDim, numNonZeroVoxels,
        numParticlesInVoxelLists, numVoxelsPerNode);

    gpuErrchk(cudaPeekAtLastError());
    cudaStream_t particleListStream;
    cudaStreamCreate(&particleListStream);

    cudaStreamSynchronize(stream);
    cudaParallelPrefixSum(numGridNodes, numNonZeroVoxels, stream);
    cudaParallelPrefixSum(numGridNodes, numParticlesInVoxelLists, particleListStream);
    
    gpuErrchk( cudaStreamSynchronize(particleListStream) );
    gpuErrchk( cudaStreamDestroy(particleListStream) );
}



__global__ void createParticleListStartIndices(uint totalNumberVoxelsDimension, uint numGridNodes, uint numParticles, uint numVoxelsPerNode, uint* voxelIDs,
    uint* perVoxelParticleListStartIndices, uint* numUsedVoxelsPerNode, uint* subCellsTouchedPerParticle, uint* subCellsDim,
    uint* firstParticleInNodeIndex, uint* particleLists, uint* firstParticleListIndexPerNode)   //need to revisit how particle list creation is done
{

    //firstParticleInNodeIndex used to index node->subCellsTouchedPerParticle
    //subCellsTouchedPerParticle used to index particle->subCellsDim
    extern __shared__ uint voxelUsedIndex[]; //sized to 2*numVoxelsPerNode
    __shared__ uint maxParticleNum;
    uint* voxelCount = voxelUsedIndex + numVoxelsPerNode;
    __shared__ uint numUsedVoxelsThisNode;

    __shared__ uint firstParticleListIndex;
    __shared__ uint voxelUsedMax;
    __shared__ uint voxelCountMax;

    if(threadIdx.x == 0){
        if(blockIdx.x < numGridNodes - 1){
            maxParticleNum = firstParticleInNodeIndex[blockIdx.x + 1];
        }
        else{
            maxParticleNum = numParticles;
        }
        if(blockIdx.x == 0){
            numUsedVoxelsThisNode = 0;
            firstParticleListIndex = 0;
        }
        else{
            numUsedVoxelsThisNode = numUsedVoxelsPerNode[blockIdx.x - 1];
            firstParticleListIndex = firstParticleListIndexPerNode[blockIdx.x-1];
        }
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        voxelUsedIndex[i] = 0;
        voxelCount[i] = 0;
    }

    __syncthreads();

    for(uint i = firstParticleInNodeIndex[blockIdx.x] + threadIdx.x; i < maxParticleNum; i += blockDim.x){
        uint firstSubCellsTouched;
        if(i == 0){
            firstSubCellsTouched = 0;
        }
        else{
            firstSubCellsTouched = subCellsTouchedPerParticle[i-1];
        }
        for(uint j = firstSubCellsTouched; j < subCellsTouchedPerParticle[i]; ++j){
            atomicMax(voxelUsedIndex + subCellsDim[j], 1);  //marking used voxels
            atomicAdd(voxelCount + subCellsDim[j], 1);
        }
    }

    __syncthreads();
    //now have number of particles per voxel, need blockWise prefix sum on voxelCount to get offset of each used voxel's particle list
    blockWiseExclusivePrefixSum(voxelUsedIndex, numVoxelsPerNode, voxelUsedMax);
    blockWiseExclusivePrefixSum(voxelCount, numVoxelsPerNode, voxelCountMax);

    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        uint nextVal = voxelUsedMax;
        uint nextCount = voxelCountMax;
        if(i != numVoxelsPerNode-1){
            nextVal = voxelUsedIndex[i+1];
            nextCount = voxelCount[i+1];
        }
        if(voxelUsedIndex[i] != nextVal){
            voxelIDs[numUsedVoxelsThisNode + voxelUsedIndex[i]] = i;   //store in-node voxel ID to voxelIDs array
            perVoxelParticleListStartIndices[numUsedVoxelsThisNode + voxelUsedIndex[i]] = voxelCount[i] + firstParticleListIndex;

            //need to reframe from per-particle thread to per-voxel thread, each thread iterates over all particles in node to generate per-voxel list of particles
            uint numParticlesWrittenToCurrentVoxel = 0;
            for(uint particleIndex = firstParticleInNodeIndex[blockIdx.x]; particleIndex < maxParticleNum; ++particleIndex){    //whole loop needs to be its own kernel, iterate only over used nodes and write accordingly
                int firstSubCellsTouched;
                if(particleIndex == 0){
                    firstSubCellsTouched = 0;
                }
                else{
                    firstSubCellsTouched = subCellsTouchedPerParticle[particleIndex-1];
                }
                for(uint subCellTouchedPerParticleIndex = firstSubCellsTouched;                            //current version wastes threads and unnecessarily prolongs runtime of each thread
                        subCellTouchedPerParticleIndex < subCellsTouchedPerParticle[particleIndex];
                        ++subCellTouchedPerParticleIndex)
                {
                    if(subCellsDim[subCellTouchedPerParticleIndex] == i){   //need to convert this to shared mem array 
                        particleLists[firstParticleListIndex + voxelCount[i] + numParticlesWrittenToCurrentVoxel++] = particleIndex;
                    }
                }
            }
        }
    }
}
    
void cudaParticleListCreate(uint totalNumberVoxelsDimension, uint numGridNodes, uint numParticles, uint numVoxelsPerNode,
    uint* voxelIDs, uint* perVoxelParticleListStartIndices, uint* numUsedVoxelsPerNode, uint* firstParticleListIndexPerNode,
    uint* subCellsTouchedPerParticle, uint* subCellsDim, uint* firstParticleInNodeIndex, uint* particleLists,
    cudaStream_t stream)
{
    createParticleListStartIndices<<<numGridNodes, BLOCKSIZE, 2*numVoxelsPerNode*sizeof(uint), stream>>>(
        totalNumberVoxelsDimension, numGridNodes, numParticles, numVoxelsPerNode, voxelIDs, perVoxelParticleListStartIndices,
        numUsedVoxelsPerNode, subCellsTouchedPerParticle, subCellsDim, firstParticleInNodeIndex, particleLists, firstParticleListIndexPerNode);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk( cudaStreamSynchronize(stream) );
}