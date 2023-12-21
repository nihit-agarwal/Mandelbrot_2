#pragma once
#include <iostream>

// Constants
#define HEIGHT 2048
#define WIDTH 2048
#define ITERATIONS 100
#define SCALE 0.001

struct BmpHeader {
	char bitmapSignatureBytes[2] = {'B', 'M'};
	uint32_t sizeOfBitmapFile = 54 + HEIGHT * WIDTH * 3;
	uint32_t reservedBytes = 0;
	uint32_t pixelDataOffset = 54;

} bmpHeader;

struct BmpInfoHeader {
    uint32_t sizeOfThisHeader = 40;
    int32_t width = WIDTH; // in pixels
    int32_t height = HEIGHT; // in pixels
    uint16_t numberOfColorPlanes = 1; // must be 1
    uint16_t colorDepth = 24;
    uint32_t compressionMethod = 0;
    uint32_t rawBitmapDataSize = 0; // generally ignored
    int32_t horizontalResolution = 3780; // in pixel per meter
    int32_t verticalResolution = 3780; // in pixel per meter
    uint32_t colorTableEntries = 0;
    uint32_t importantColors = 0;
} bmpInfoHeader;

struct Pixel {
    uint8_t blue = 0;
    uint8_t green = 255;
    uint8_t red = 0;
} pixel;

struct Coordinates {
    double x;
    double y;
};