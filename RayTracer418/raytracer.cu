/***********************************************************

	Starter code for Assignment 3

	Implementations of functions in raytracer.h,
	and the main function which specifies the scene to be rendered.

***********************************************************/

#define ITERATIONS 5
//#define epsilon 0.0001

const float epsilon = 0.0001;

#include "raytracer.cuh"
#include "kernel.cuh"
#include "util.h"

#include "cuda_runtime.h"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <typeinfo>



// NOTE!!!!
//https://forums.developer.nvidia.com/t/the-cost-of-relocatable-device-code-rdc-true/47665
// using relocatable devide code (rdc) is better for readability, but potentially worse for speed
// RDC doesn't even seem to be working???




//int debug = 0;

__host__ __device__
void Raytracer::traverseScene(Scene& scene, Ray3D& ray) {

	double t_min;
	t_min = 1000000;

	Intersection intersect;
	intersect.none = true;


	for (size_t i = 0; i < scene.size(); ++i) {
		SceneNode* node = scene[i];
		
		//SceneObject* objs = node->obj;

		Matrix4x4 m1 = node->worldToModel;
		//Matrix4x4 m2 = node->modelToWorld;


		
		//if (node->obj->intersect(ray, node->worldToModel, node->modelToWorld, 0)) {

			/*if (ray.intersection.t_value > 0.0001 && ray.intersection.t_value < t_min) {

				ray.intersection.mat = node->mat;

				t_min = ray.intersection.t_value;
				intersect.point = ray.intersection.point;
				intersect.normal = ray.intersection.normal;
				intersect.mat = node->mat;
				intersect.none = false;

			}*/
		//}
	}
	// I think keeping a seperate Interection instance
	// is no longer necessary as intersection methods handle cases when something has already been hit.
	ray.intersection.point = intersect.point;
	ray.intersection.normal = intersect.normal;
	ray.intersection.mat = intersect.mat;
	ray.intersection.none = intersect.none;
	ray.intersection.t_value = t_min;
	ray.intersection.normal.normalize();

}

__host__ __device__
void Raytracer::computeTransforms(Scene& scene) {
	// right now this method might seem redundant. But if you decide to implement
	// scene graph this is where you would propagate transformations to child nodes

	for (size_t i = 0; i < scene.size(); ++i) {
		SceneNode* node = scene[i];

		node->modelToWorld = node->trans;
		node->worldToModel = node->invtrans;
	}
}

__host__ __device__
void Raytracer::computeShading(Ray3D& ray, LightList& light_list, Scene& scene, int depth) {

	for (size_t i = 0; i < light_list.size(); ++i) {
		LightSource* light = light_list[i];

		Vector3D l = light->get_position() - ray.intersection.point;
		l.normalize();

		Ray3D shadowRay;
		shadowRay.origin = ray.intersection.point;
		shadowRay.dir = light->get_position() - shadowRay.origin;
		shadowRay.dir.normalize();

		float ep = 0.1;
		for (size_t i = 0; i < scene.size(); ++i) {
			SceneNode* node = scene[i];
			if (node->obj->intersect(shadowRay, node->worldToModel, node->modelToWorld, 0)) {
				if (!shadowRay.intersection.none && (shadowRay.intersection.t_value > ep || shadowRay.intersection.t_value < -ep) /*&& (shadowRay.intersection.mat->transparent > 0.0)*/) {
					break;
				}
				else {
					shadowRay.intersection.none = true;
				}
			}
		}

		if (this->shadows) {
			if (!shadowRay.intersection.none)
				light->shade(ray, false, scene, shadows); // just ambient
			else
				light->shade(ray, true, scene, shadows); //all three
		}
		else
			light->shade(ray, true, scene, shadows);
	}
	ray.col = (1.0 / light_list.size()) * ray.col;
	ray.col.clamp();
}

__host__ __device__
Color Raytracer::shadeRay(Ray3D& ray, Scene& scene, LightList& light_list, int depth) {

	Color col(0.0, 0.0, 0.0);

	
	traverseScene(scene, ray);

	return Color(0.516228, 0.016228, 0.016228);

	// Don't bother shading if the ray didn't hit
	// anything.
	if (!ray.intersection.none) {
		computeShading(ray, light_list, scene, depth);

		Material mat = *(ray.intersection.mat);

		col = (1 - mat.transparent - mat.mirror) * ray.col;

		Vector3D normal;
		if (depth > 0) {

			// Refraction
			if (mat.transparent > 0) {

				Color refractColor(0, 0, 0);
				normal = ray.intersection.normal;
				normal.normalize();
				Vector3D incident = -ray.dir;
				incident.normalize();

				// cos(angle in)
				float c1 = incident.dot(normal);

				float n = 1;
				if (c1 <= 0) {
					//inside the object in question
					normal = -normal;
					c1 = incident.dot(normal);
					n = 1 / mat.refract;
				}
				else { //>=0
				 //outside
					n = mat.refract;
				}

				// Sin(angle out)^2
				float s22 = n * n * (1.0 - c1 * c1);
				// Cos(angle out)^2
				float c22 = 1.0 - n * n * (1.0 - c1 * c1);

				if (c22 >= 0) {

					// cos(angle out)
					float c2 = sqrt(c22);

					//refract = n*(-I) + (n*c_1 - c_2)N
					Vector3D refract = -n * incident + (n * c1 - c2) * normal;
					refract.normalize();

					Ray3D refractRay;
					refractRay.origin = ray.intersection.point + 0.0001 * refract;
					refractRay.dir = refract;

					refractColor = shadeRay(refractRay, scene, light_list, depth - 1);

				}
				col = col + (mat.transparent) * refractColor;
			}

			// Reflection
			if (mat.mirror > 0) {

				normal = ray.intersection.normal;
				normal.normalize();

				Vector3D incident = -ray.dir;
				incident.normalize();

				Vector3D reflect = ((2 * normal.dot(incident)) * normal - incident);
				reflect.normalize();

				Ray3D recRay;
				recRay.origin = ray.intersection.point + 0.0001 * reflect;
				recRay.dir = reflect;

				col = col + (mat.mirror) * shadeRay(recRay, scene, light_list, depth - 1);
			}
		}
		col.clamp();
	}

	// You'll want to call shadeRay recursively (with a different ray,
	// of course) here to implement reflection/refraction effects.

	return col;
}

__host__ __device__
void Raytracer::basicRayTracing(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j) {

	Matrix4x4 viewToWorld;
	viewToWorld = camera.initInvViewMatrix();

	Point3D origin(0, 0, 0);
	Point3D imagePlane;

	// Sets up ray origin and direction in view space,
	// image plane is at z = -1.
	imagePlane[0] = (-double(image.width) / 2 + 0.5 + j) / factor;
	imagePlane[1] = (-double(image.height) / 2 + 0.5 + i) / factor;
	imagePlane[2] = -1;

	Ray3D ray;
	ray.origin = viewToWorld * origin;
	ray.dir = (viewToWorld * imagePlane) - ray.origin;
	ray.dir.normalize();

	Color col = shadeRay(ray, scene, light_list, ITERATIONS);

	image.setColorAtPixel(i, j, col);

}

__host__ __device__
void Raytracer::depthOfField(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j) {

#define F 5 //5
#define R 0.3 //0.7
#define SAMPLES 50

	Matrix4x4 viewToWorld;
	//double factor = (double(image.height)/2)/tan(camera.fov*M_PI/360.0);
	viewToWorld = camera.initInvViewMatrix();

	Point3D origin(0, 0, 0);
	Point3D imagePlane;
	imagePlane[0] = (-double(image.width) / 2 + 0.5 + j) / factor;
	imagePlane[1] = (-double(image.height) / 2 + 0.5 + i) / factor;
	imagePlane[2] = -1;
	Color col(0, 0, 0);

	origin = viewToWorld * origin;
	Vector3D dir = (viewToWorld * imagePlane) - origin;
	dir.normalize();

	Point3D C = origin + F * dir;

	for (int k = 0; k < SAMPLES; k++) {
		//double x = rand() / (float)RAND_MAX; // sampled from [0, 1] at random
		//double y = rand() / (float)RAND_MAX;
		//double z = rand() / (float)RAND_MAX;
		//x = x * 2 * R - R; y = y * 2 * R - R; z = z * 2 * R - R;  // x,y,z in [-R, R]

		double x = 0, y = 0, z = 0;

		Ray3D ray;
		ray.origin = origin + Vector3D(x, y, z);

		ray.dir = C - ray.origin;
		ray.dir.normalize();

		//col = col + shadeRay(ray, scene, light_list, ITERATIONS);

	}
	col = (1.0 / SAMPLES) * col;
	col.clamp();

	image.setColorAtPixel(i, j, col);

}









__host__ __device__
void Raytracer::antialiasing(Camera& camera, Scene& scene, LightList& light_list, Image& image, double factor, int i, int j) {

#define N 3

	int b_index = i * image.width + j;
	/*if (j >= 512) {
		image.rbuffer[b_index] = int(0.516228 * 255);
		image.gbuffer[b_index] = int(0.016228 * 255);
		image.bbuffer[b_index] = int(0.016228 * 255);
	}
	else {
		image.rbuffer[b_index] = int(0.75164 * 255);
		image.gbuffer[b_index] = int(0.60648 * 255);
		image.bbuffer[b_index] = int(0.22648 * 255);
		return;
	}*/

	//return;

	Matrix4x4 viewToWorld;
	//double factor = (double(image.height)/2)/tan(camera.fov*M_PI/360.0);
	viewToWorld = camera.initInvViewMatrix();

	Point3D origin(0, 0, 0);
	Point3D imagePlane;
	imagePlane[2] = -1;

	imagePlane[0] = (-double(image.width) / 2 + (0 + 0) / N + j) / factor;
	imagePlane[1] = (-double(image.height) / 2 + (0 + 0) / N + i) / factor;

	Color col(0, 0, 0);

	int k;
	int l;

	for (k = 0; k < N; k++) {
		for (l = 0; l < N; l++) {

			
			//double x = rand() / (float)RAND_MAX;
			//double y = rand() / (float)RAND_MAX;

			double x = 0;
			double y = 0;

			imagePlane[0] = (-double(image.width) / 2 + (k + x) / N + j) / factor; // unchanged
			imagePlane[1] = (-double(image.height) / 2 + (l + y) / N + i) / factor; // unchanged

			Ray3D ray;
			ray.origin = viewToWorld * origin;
			ray.dir = (viewToWorld * imagePlane) - ray.origin;

			col = col + shadeRay(ray, scene, light_list, 5); //ITERATIONS
	
		}
	}
	col = (1.0 / (N * N)) * col;
	col.clamp();
	image.setColorAtPixel(i, j, col);


	//image.rbuffer[b_index] = int(0.516228 * 255);
	//image.gbuffer[b_index] = int(0.016228 * 255);
	//image.bbuffer[b_index] = int(0.016228 * 255);

}
















































__host__
void Raytracer::render(Camera& camera, Scene& scene, LightList& light_list, Image& image) {
	computeTransforms(scene);

	double factor = (double(image.height) / 2) / tan(camera.fov * M_PI / 360.0);
	// Construct a ray for each pixel.


	// See kernel.cu
	CUDArender(*this, camera, scene, light_list, image);




	/*#pragma omp parallel for collapse(2)
	for (int i = 0; i < image.height; i++) {
		for (int j = 0; j < image.width; j++) {

			// Calls one of the functions below:
		 //* - Basic raytracing :
		  //*
		  //* - jitter algorithm for AA :
		 // *
		 // * - approximate depth of field
			(this->*renderFunction)(camera, scene, light_list, image, factor, i, j);

		}
	}*/
}


