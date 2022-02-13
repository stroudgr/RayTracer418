
#ifndef __KERNEL__H
#define __KERNEL__H

#include "raytracer.cuh"
#include "util.h"
#include "world.h"

__global__ void render_kernel(Raytracer* d_raytracer, Camera* d_camera, Scene* d_scene, LightList* d_light_list, Image* d_image);

__host__
void CUDArender(Raytracer& r, Camera& camera, Scene& scene, LightList& light_list, Image& image);


#endif
