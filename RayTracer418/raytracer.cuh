/***********************************************************

	Starter code for Assignment 3

	This file contains the interface and datastructures of the raytracer.
	Simple traversal and addition code to the datastructures are given to you.

***********************************************************/
#pragma once

#include <stdlib.h>
#include "util.h"
#include "scene_object.h"
#include "light_source.h"
#include "raytracer.cuh"
#include "cuda_runtime.h"



/// <summary>
///  unused???????????????????????????
/// </summary>
enum RT_TYPE
{
	RT_BASIC, RT_AA, RT_DOF
};

class Raytracer {
public:

	__device__ __host__ Raytracer(const char* name, bool shadow) {
		type = name;
		if (name == "basic") {
			renderFunction = &Raytracer::basicRayTracing;
		}
		else if (name == "aa") {
			renderFunction = &Raytracer::antialiasing;
		}
		else if (name == "dof") {
			renderFunction = &Raytracer::depthOfField;
		}
		else {
			//fprintf(stderr, "Not a valid raytracing method. (Please don't edit the main files)\n");
			//exit(1);
		}

		shadows = shadow;
	}

	void (Raytracer::* renderFunction)(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j);
	//const char* renderFunctionSTR;


	__host__ __device__  void antialiasing(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j);

	__host__ __device__ void daa(Raytracer& d_raytracer, Image& image, int i, int j);

	// Renders 3D scene to an image given camera and lights setup.
	__host__ void render(Camera& camera, Scene& scene, LightList& light_list, Image& image);

private:

	// Return the color of the ray after intersection and shading, call
	// this function recursively for reflection and refraction.
	__host__ __device__ Color shadeRay(Ray3D& ray, Scene& scene, LightList& light_list, int depth);

	// Traversal code for the scene, the ray is transformed into
	// the object space of each node where intersection is performed.
	__host__ __device__ void traverseScene(Scene& scene, Ray3D& ray);

	// After intersection, calculate the color of the ray by shading it
	// with all light sources in the scene.
	__host__ __device__ void computeShading(Ray3D& ray, LightList& light_list, Scene& scene, int depth);

	// Precompute the modelToWorld and worldToModel transformations for each
	// object in the scene.
	__host__ __device__ void computeTransforms(Scene& scene);

	// Flags
	bool shadows;
	const char* type;


	// Different options for lighting functions to be .

	__host__ __device__  void depthOfField(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j);
	__host__ __device__  void basicRayTracing(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j);

};

