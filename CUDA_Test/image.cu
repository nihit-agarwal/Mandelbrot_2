// Importing Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper.h"
#include <fstream>
#include <SFML/Graphics.hpp>
#include <thrust/complex.h>
cudaError_t convergenceCuda(sf::Uint8** colors, Coordinates** points);
cudaError_t memoryAllocateCuda(sf::Uint8** colors, Coordinates** points);
cudaError_t memoryFreeCuda(sf::Uint8** colors, Coordinates** points);

__global__ void checkConverergence(sf::Uint8* colors, Coordinates* points)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < HEIGHT && j < WIDTH)
    {
        double x = points[i * WIDTH + j].x;
        double y = points[i * WIDTH + j].y;
        thrust::complex<double> num = thrust::complex<double>(x, y);
        thrust::complex<double> curr(0,0);
        int iteration = 0;
        while (abs(curr) <= 2 && iteration < ITERATIONS)
        {
            curr = curr * curr + num;
            iteration++;
        }
        if (iteration == ITERATIONS)
        {
            colors[4* (i * WIDTH + j)] = 0; // Blue
            colors[4 * (i * WIDTH + j) + 1] = 255; // Green
            colors[4 * (i * WIDTH + j) + 2] = 0; // Red
            colors[4 * (i * WIDTH + j) + 3] = 255; // Alpha
        }
        else
        {
            colors[4 * (i * WIDTH + j)] = 0; // Blue
            colors[4 * (i * WIDTH + j) + 1] = 0; // Green
            colors[4 * (i * WIDTH + j) + 2] = 0; // Red
            colors[4 * (i * WIDTH + j) + 3] = 255; // Alpha
        }
    }

}

int main() {
	std::cout << "Hello cuda !" << std::endl;
    const unsigned numPixels = HEIGHT * WIDTH;
    Coordinates* points = nullptr;
    sf::Uint8* pixels = nullptr;

    memoryAllocateCuda(&pixels, &points);
    
    convergenceCuda(&pixels, &points);
    
    cudaEvent_t startEvent, stopEvent;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float elapsed_time;


    sf::Image image;
    image.create(WIDTH, HEIGHT, pixels);
    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "SFML works!");
    bool flag = false;
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
            else if (event.type == sf::Event::KeyPressed)
            {
                //std::cout << "Key Pressed" << event.key.code << std::endl;
                // Move Up
                if (event.key.code == sf::Keyboard::A)
                {
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i * WIDTH + j].y -= 10 * SCALE;
                        }
                    }
                }
                else if (event.key.code == sf::Keyboard::S)
                {
                    //std::cout << "S pressed" << std::endl;
                    // Move down
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i * WIDTH + j].y += 10 * SCALE;
                        }
                    }
                }
                else if (event.key.code == sf::Keyboard::W)
                {
                    //std::cout << "W pressed" << std::endl;
                    // Zoom in
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i * WIDTH + j].y *= 0.5;
                            points[i * WIDTH + j].x *= 0.5;
                        }
                    }
                }
                else if (event.key.code == sf::Keyboard::D)
                {
                    //std::cout << "D pressed" << std::endl;
                    // Zoom out
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i * WIDTH + j].x *= 2;
                            points[i * WIDTH + j].y *= 2;
                        }
                    }
                }
                cudaEventRecord(startEvent, 0);
                cudaError_t status = convergenceCuda(&pixels, &points);
                image.create(WIDTH, HEIGHT, pixels);
                texture.loadFromImage(image);
                sprite.setTexture(texture);

                cudaEventRecord(stopEvent, 0);
                cudaEventSynchronize(stopEvent);
                cudaEventElapsedTime(&elapsed_time, startEvent, stopEvent);

                std::cout << "Update time: " << elapsed_time << std::endl;
                flag = true;
            }
        }
        cudaEventRecord(startEvent, 0);
        window.clear();
        window.draw(sprite);
        window.display();

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed_time, startEvent, stopEvent);
        if (flag)
        {
            std::cout << "Image draw time: " << elapsed_time << std::endl;
            flag = false;
        }
        
    }
    

    /*for (int i = 0; i < 50; i++)
    {
        for (int j = 0; j < 50; j++)
        {
            /*std::cout << pixels[i * WIDTH + j] << ","
                << pixels[i * WIDTH + j + 1] << "," 
                << pixels[i * WIDTH + j + 2] << ","
                << pixels[i * WIDTH + j + 3] << " ";

            std::cout << points[i][j].x << ","
                << points[i][j].y << " ";
        }
        std::cout << std::endl;
    }
    */

    memoryFreeCuda(&pixels, &points);
	return 0;

}

cudaError_t memoryAllocateCuda(sf::Uint8** colors, Coordinates** points)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMallocManaged(points, sizeof(Coordinates) * HEIGHT * WIDTH);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Memory allocation for points failed.\n";
    }
    cudaStatus = cudaMallocManaged(colors, sizeof(sf::Uint8) * 4 * HEIGHT * WIDTH);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Memory allocation for colors failed.\n";
    }

    // Initial setup 
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            (*points)[i * WIDTH + j].x = (j - WIDTH / 2) * SCALE;
            (*points)[i * WIDTH + j].y = (HEIGHT / 2 - i) * SCALE;
        }
    }

    return cudaStatus;
}

cudaError_t memoryFreeCuda(sf::Uint8** colors, Coordinates** points)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaFree(*colors);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Memory freeing for points failed.\n";
    }
    cudaStatus = cudaFree(*points);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Memory freeing for colors failed.\n";
    }
    return cudaStatus;
}
cudaError_t convergenceCuda(sf::Uint8** colors, Coordinates **points)
{
    
    // Number of threads per block
    //std::cout << "Memory copied from host to device" << std::endl;
    dim3 threadsPerBlock(32, 32);
    dim3 numberOfBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);

    checkConverergence <<<numberOfBlocks, threadsPerBlock >>> (*colors, *points);
    cudaError_t cudaStatus = cudaDeviceSynchronize();

    
   
    return cudaStatus;

}
