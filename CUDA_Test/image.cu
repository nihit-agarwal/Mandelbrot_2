// Importing Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitmapCreation.h"
#include <fstream>
#include <SFML/Graphics.hpp>
#include <thrust/complex.h>
cudaError_t convergenceCuda(dim3** color, Coordinates** points);

__global__ void checkConverergence(dim3** color, Coordinates** points)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < HEIGHT && j < WIDTH)
    {
        double x = points[i][j].x;
        double y = points[i][j].y;
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
            color[i][j] = dim3(255, 0, 0);
        }
        else
        {
            color[i][j] = dim3(0, 0, 0);
        }
    }

}

int main() {
	std::cout << "Hello cuda !" << std::endl;
	
    dim3** colors = new dim3*[HEIGHT];
    Coordinates** points = new Coordinates*[HEIGHT];

    for (int i = 0; i < HEIGHT; i++)
    {
        colors[i] = new dim3[WIDTH];
        points[i] = new Coordinates[WIDTH];
    }
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            points[i][j].x = (j - WIDTH/2) * SCALE;
            points[i][j].y = (HEIGHT/2 - i) * SCALE;
        }
    }
    
    std::cout << "Coordinates computation completed" << std::endl;
    convergenceCuda(colors, points);


    const unsigned numPixels = HEIGHT * WIDTH;
    sf::Uint8* pixels = new sf::Uint8[4 * numPixels];
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            pixels[i * WIDTH * 4 + j * 4] = colors[i][j].x;
            pixels[4 * (i * WIDTH + j) + 1] = colors[i][j].y;
            pixels[4 * (i * WIDTH + j) + 2] = colors[i][j].z;
            pixels[4 * (i * WIDTH + j) + 3] = 255;



        }
    }
    sf::Image image;
    image.create(WIDTH, HEIGHT, pixels);
    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "SFML works!");

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
                std::cout << "Key Pressed" << event.key.code << std::endl;
                if (event.key.code == sf::Keyboard::A)
                {
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i][j].y -= 10 * SCALE;
                        }
                    }
                }
                else if (event.key.code == sf::Keyboard::S)
                {
                    std::cout << "S pressed" << std::endl;
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i][j].y += 10 * SCALE;
                        }
                    }
                }
                else if (event.key.code == sf::Keyboard::W)
                {
                    std::cout << "W pressed" << std::endl;
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i][j].y *= 0.5;
                            points[i][j].x *= 0.5;
                        }
                    }
                }
                else if (event.key.code == sf::Keyboard::D)
                {
                    std::cout << "D pressed" << std::endl;
                    for (int i = 0; i < HEIGHT; i++)
                    {
                        for (int j = 0; j < WIDTH; j++)
                        {
                            points[i][j].y *= 2;
                            points[i][j].x *= 2;
                        }
                    }
                }
                cudaError_t status = convergenceCuda(colors, points);
                std::cout << "After function call " << status << std::endl;
                for (int i = 0; i < HEIGHT; i++)
                {
                    for (int j = 0; j < WIDTH; j++)
                    {
                        pixels[i * WIDTH * 4 + j * 4] = colors[i][j].x;
                        pixels[4 * (i * WIDTH + j) + 1] = colors[i][j].y;
                        pixels[4 * (i * WIDTH + j) + 2] = colors[i][j].z;
                        pixels[4 * (i * WIDTH + j) + 3] = 255;



                    }
                }
                image.create(WIDTH, HEIGHT, pixels);
                texture.loadFromImage(image);
                sprite.setTexture(texture);

            }
        }

        window.clear();
        window.draw(sprite);
        window.display();
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

    for (int i = 0; i < HEIGHT; i++)
    {
        delete[] colors[i];
        delete[] points[i];
    }

    delete[] colors;
    delete[] points;
    delete[] pixels;
	return 0;

}


cudaError_t convergenceCuda(dim3** color, Coordinates** points)
{
    Coordinates** dev_points;
    dim3** dev_colors;
    cudaError_t cudaStatus;

    dim3** host_colors = new dim3 * [HEIGHT];
    Coordinates** host_points = new Coordinates * [HEIGHT];

    cudaStatus = cudaMalloc((void**)&dev_points, sizeof(Coordinates*) * HEIGHT);
    //std::cout << "Allocating memory for points " << cudaStatus << std::endl;
    cudaStatus = cudaMalloc((void**)&dev_colors, sizeof(dim3*) * HEIGHT);

    
    //std::cout << "Allocating memory for colors " << cudaStatus << std::endl;
    for (int i = 0; i < HEIGHT; i++)
    {
        cudaStatus = cudaMalloc((void**)&(host_points[i]), sizeof(Coordinates) * WIDTH);
        cudaMemcpy(host_points[i], points[i], sizeof(Coordinates) * WIDTH, cudaMemcpyHostToDevice);
        cudaStatus = cudaMalloc((void**)&(host_colors[i]), sizeof(dim3) * WIDTH);
    }

    //std::cout << "Memory allocated" << std::endl;
    cudaStatus = cudaMemcpy(dev_points, host_points, HEIGHT * sizeof(Coordinates*), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_colors, host_colors, HEIGHT * sizeof(dim3*), cudaMemcpyHostToDevice);

    // Number of threads per block
    //std::cout << "Memory copied from host to device" << std::endl;
    dim3 threadsPerBlock(16, 16);
    dim3 numberOfBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);

    checkConverergence <<<numberOfBlocks, threadsPerBlock >>> (dev_colors, dev_points);
    cudaDeviceSynchronize();

    //std::cout << "Computation done" << std::endl;

    for (int i = 0; i < HEIGHT; i++)
    {
        cudaMemcpy(color[i], host_colors[i], sizeof(dim3) * WIDTH, cudaMemcpyDeviceToHost);
    }
    //cudaStatus = cudaMemcpy2D(color[0], WIDTH * sizeof(dim3),
     //   dev_colors[0], WIDTH * sizeof(dim3),
       // WIDTH * sizeof(dim3), HEIGHT,
        //cudaMemcpyDeviceToHost);

    //std::cout << "Memory copied from device to host" << std::endl;
    for (int i = 0; i < HEIGHT; i++)
    {
        cudaStatus = cudaFree(host_points[i]);
        cudaStatus = cudaFree(host_colors[i]);
    }
    cudaFree(dev_points);
    cudaFree(dev_colors);
    //std::cout << "Memory freed" << std::endl;
    return cudaStatus;

}
