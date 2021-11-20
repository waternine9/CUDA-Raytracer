
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "Header.cuh"
#define RESOLUTION 200
#define NUMSPHERES 100000
#define NUMTRIS 1

using namespace std;
using namespace cv;
cudaError_t addWithCuda(int *a, int *b, int *c, int size, int steps);
__global__ void raytrace(int* r, int* g, int* b, int* sphereData, int* triData, int steps)
{
    
    float i = threadIdx.x;
    float j = blockIdx.x;
    float speed = 1;
    float3 rayPos = { i - RESOLUTION / 2.0, j - RESOLUTION / 2.0, 0 };
    float3 rayVec = { (i / (RESOLUTION * 4) - 0.25) * speed, (j / (RESOLUTION * 4) - 0.25) * speed, speed };
    float3 rayCol = { 0, 0, 0 };
    float vecMagnitude = sqrtf(rayVec.x * rayVec.x + rayVec.y * rayVec.y + rayVec.z * rayVec.z);
    
    rayVec.x /= vecMagnitude;
    rayVec.y /= vecMagnitude;
    rayVec.z /= vecMagnitude;
    
    
    const int radius = 10;
    float totalDist = 10000;
    float3 actSpherePos;
    for (int x = 0;x < NUMSPHERES;x++)
    {
        
        float3 spherePos = { sphereData[x * 3], sphereData[x * 3 + 1], sphereData[x * 3 + 2] };
        float3 oc = { rayPos.x - spherePos.x, rayPos.y - spherePos.y, rayPos.z - spherePos.z };
        float a = rayVec.x * rayVec.x + rayVec.y * rayVec.y + rayVec.z * rayVec.z;
        float n = 2.0 * oc.x * rayVec.x + oc.y * rayVec.y + oc.z * rayVec.z;
        float p = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - radius * radius;
        float discriminant = n * n - 4 * a * p;

        if (discriminant >= 0)
        {
            float dist = (-n - sqrtf(discriminant)) / (2.0 * a);
            if (dist < totalDist)
            {
                totalDist = dist;
                actSpherePos.x = spherePos.x;
                actSpherePos.y = spherePos.y;
                actSpherePos.z = spherePos.z;
            }
        }
        
    }
    for (int x = 0;x < NUMTRIS;x++)
    {
        float3 v0 = { triData[x * 9], triData[x * 9 + 1], triData[x * 9 + 2] };
        float3 v1 = { triData[x * 9 + 3], triData[x * 9 + 4], triData[x * 9 + 5] };
        float3 v2 = { triData[x * 9 + 6], triData[x * 9 + 7], triData[x * 9 + 8] };
        // printf("%.6f", v0.x);
        float3 v0v1 = sub(v1, v0);
        float3 v0v2 = sub(v2, v0);
        // no need to normalize
        float3 N = cross(v0v1, v0v2); // N 
        float area2 = length(N);

        // Step 1: finding P

        // check if ray and plane are parallel ?
        float NdotRayDirection = dot(N, rayVec);
        
        if (fabs(NdotRayDirection) > 0.00001f)
        {
            // compute d parameter using equation 2
            float d = dot(N, v0);
            // printf("%.6f", NdotRayDirection);
            // compute t (equation 3)
            float t = (dot(N, rayPos) + d) / NdotRayDirection;
            // check if the triangle is in behind the ray
            
            if (t > 0)
            {// the triangle is behind 

            
                float3 P = { rayPos.x + t * rayVec.x, rayPos.y + t * rayVec.y, rayPos.z + t * rayVec.z };

                // Step 2: inside-outside test
                float3 C; // vector perpendicular to triangle's plane 

                // edge 0
                float3 edge0 = sub(v1, v0);
                float3 vp0 = sub(P, v0);
                C = cross(edge0, vp0);
                if (dot(N, C) > 0)
                {


                    // edge 1
                    float3 edge1 = sub(v2, v1);
                    float3 vp1 = sub(P, v1);
                    C = cross(edge1, vp1);
                    if (dot(N, C) > 0)
                    {
                        // edge 2
                        float3 edge2 = sub(v0, v2);
                        float3 vp2 = sub(P, v2);
                        C = cross(edge2, vp2);
                        if (dot(N, C) > 0)
                        {
        
                            float dist = length(sub(P, rayPos));
                            if (dist < totalDist)
                            {
                                totalDist = dist;
                                actSpherePos = P;
                                // printf("Bruh\n", totalDist);
                            }
                            
                        }
                    }
                }
            }
        }
    }
    if (totalDist < 10000)

    {

        float3 hitPoint = { rayVec.x * totalDist, rayVec.y * totalDist, rayVec.z * totalDist };
        float3 normal = { hitPoint.x - actSpherePos.x, hitPoint.y - actSpherePos.y, hitPoint.z - actSpherePos.z };
        float normalMagnitude = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        normal.x /= normalMagnitude;
        normal.y /= normalMagnitude;
        normal.z /= normalMagnitude;
        float3 lightDir = { sphereData[0] - hitPoint.x, sphereData[1] - hitPoint.y, sphereData[2] - hitPoint.z };
        float lightMagnitude = sqrtf(lightDir.x * lightDir.x + lightDir.y * lightDir.y + lightDir.z * lightDir.z);
        lightDir.x /= lightMagnitude;
        lightDir.y /= lightMagnitude;
        lightDir.z /= lightMagnitude;
        float3 bruh = { 0, 0, 0 };
        
        // printf("%.6f\n", dot(lightDir, normal));
        float3 d = { sphereData[0] - hitPoint.x, sphereData[1] - hitPoint.y, sphereData[2] - hitPoint.z };
        // lightDir = { -lightDir.x, -lightDir.y, -lightDir.z };
        float dot = ((normal.x * lightDir.x) + (normal.y * lightDir.y) + (normal.z * lightDir.z));
        float3 d2 = { (lightDir.x - 2 * dot * normal.x), (lightDir.y - 2 * dot * normal.y), (lightDir.z - 2 * dot * normal.z) };
        float dMagnitude = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z); 
        d.x /= dMagnitude;
        d.y /= dMagnitude;
        d.z /= dMagnitude;
        float lums = powf(fmaxf((d.x * d2.x + d.y * d2.y + d.z * d2.z), 0), 0.0) * 255;
        if (lums < 0) lums = 0; 
        if (lums > 255) lums = 255;
        rayCol.x = (lightDir.x) * 255;
        rayCol.y = (lightDir.y) * 255;
        rayCol.z = (lightDir.z) * 255;
    }
    int y = (int)j;
    int u = (int)i;
    r[y + u * RESOLUTION] = (int)rayCol.x;

    g[y + u * RESOLUTION] = (int)rayCol.y;
    b[y + u * RESOLUTION] = (int)rayCol.z;
}
class globalVars
{
public:
    int sphereData[NUMSPHERES * 3];
    int trigData[NUMTRIS * 9] = { 0 };
};
globalVars globVars = globalVars();
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    globVars.sphereData[0] = x - RESOLUTION / 2.0;
    globVars.sphereData[1] = y - RESOLUTION / 2.0;
    globVars.sphereData[2] = 0;
}
int main()
{
    /* ifstream f("C:/Users/arthu/ObjFiles/pigeon.obj");
    if (!f.is_open()) return 1;
    std::vector<std::vector<float>> verts;
    int triStep = 0;
    while (!f.eof())
    {
        char line[128];
        f.getline(line, 128);

        stringstream s;
        s << line;

        char junk;

        if (line[0] == 'v')
        {
            float3 v;
            s >> junk >> v.x >> v.y >> v.z;
            verts.push_back(std::vector<float> { v.x, v.y, v.z });
        }

        if (line[0] == 'f')
        {
            int f[3];
            s >> junk >> f[0] >> f[1] >> f[2];
            globVars.trigData[triStep * 9] = verts[f[0] - 1][0] * 10;
            globVars.trigData[triStep * 9 + 1] = verts[f[0] - 1][1] * -10;
            globVars.trigData[triStep * 9 + 2] = verts[f[0] - 1][2] * 10 + 100;
            globVars.trigData[triStep * 9 + 3] = verts[f[1] - 1][0] * 10;
            globVars.trigData[triStep * 9 + 4] = verts[f[1] - 1][1] * -10;
            globVars.trigData[triStep * 9 + 5] = verts[f[1] - 1][2] * 10 + 100;
            globVars.trigData[triStep * 9 + 6] = verts[f[2] - 1][0] * 10;
            globVars.trigData[triStep * 9 + 7] = verts[f[2] - 1][1] * -10;
            globVars.trigData[triStep * 9 + 8] = verts[f[2] - 1][2] * 10 + 100;
            triStep += 1;
        }
    } */

    srand(time(NULL));
    for (int x = 0;x < NUMSPHERES;x++)
    {
        globVars.sphereData[x * 3] = rand() % 200 - 100;
        globVars.sphereData[x * 3 + 1] = rand() % 200 - 100;
        globVars.sphereData[x * 3 + 2] = rand() % 200 + 40;
    }
    for (int x = 0;x < NUMTRIS * 9;x++)
    {
        globVars.sphereData[x] = rand() % 200 - 100;
    }
    const int arraySize = RESOLUTION * RESOLUTION;
    int r[arraySize] = { 0 };
    int g[arraySize] = { 0 };
    
    int b[arraySize] = { 0 };
    float steps = 10;
    // Add vectors in parallel.
    while (true) {
        // steps += 1;
        cudaError_t cudaStatus = addWithCuda(r, g, b, arraySize, (int)steps);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        Mat bruh = Mat::zeros(RESOLUTION, RESOLUTION, CV_8UC3);
        for (int y = 0;y < RESOLUTION;y++)
        {
            for (int x = 0;x < RESOLUTION;x++)
            {
                Vec3b& col = bruh.at<Vec3b>(x, y);
                col.val[0] = r[x + y * RESOLUTION];
                col.val[1] = g[x + y * RESOLUTION];
                col.val[2] = b[x + y * RESOLUTION];
            }
        }
        cv::imshow("output", bruh);
        setMouseCallback("output", CallBackFunc, NULL);
        waitKey(1);
        for (int x = 0;x < NUMTRIS;x++)
        {
            // globVars.trigData[x * 9 + 2] = steps;// (globVars.sphereData[0] + RESOLUTION / 2) / 8;
            // globVars.trigData[x * 9 + 5] = steps;
            // globVars.trigData[x * 9 + 8] = steps; 
        }
        // globVars.sphereData[3] = cosf(steps) * RESOLUTION / 2;

        // globVars.sphereData[4] = sinf(steps) * RESOLUTION / 2;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *a, int *b, int *c, int size, int steps)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int *dev_position = 0;
    int* dev_tris = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_position, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaMemcpy(dev_position, globVars.sphereData, size, cudaMemcpyHostToDevice);
    cudaStatus = cudaMalloc((void**)&dev_tris, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaMemcpy(dev_tris, globVars.trigData, size, cudaMemcpyHostToDevice);
    float f = clock();
    raytrace<<<RESOLUTION, RESOLUTION>>>(dev_a, dev_b, dev_c, dev_position, dev_tris, steps);

    
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    printf((to_string((clock() - f) / CLOCKS_PER_SEC) + "\n").c_str());
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_position);
    cudaFree(dev_tris);
    return cudaStatus;
}
