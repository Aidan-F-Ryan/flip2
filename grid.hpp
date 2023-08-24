#ifndef GRID_H
#define GRID_H

#include "typedefs.h"

class Grid{
public:
    Grid(float nx = 0.0f, float ny = 0.0f, float nz = 0.0f, 
        uint numGridsX = 0, uint numGridsY = 0,
        uint numGridsZ = 0, float gridSize = 0.0f)
    : negX(nx)
    , negY(ny)
    , negZ(nz)
    , sizeX(numGridsX)
    , sizeY(numGridsY)
    , sizeZ(numGridsZ)
    , cellSize(gridSize)
    {}

    void setNegativeCorner(float nx, float ny, float nz){
        negX = nx;
        negY = ny;
        negZ = nz;
    }

    void setSize(uint x, uint y, uint z, float gridSize){
        cellSize = gridSize;
        sizeX = x;
        sizeY = y;
        sizeZ = z;
    }

    float negX, negY, negZ;
    uint sizeX, sizeY, sizeZ;
    float cellSize;
};

#endif