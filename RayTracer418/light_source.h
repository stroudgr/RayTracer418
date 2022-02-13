/***********************************************************

	Starter code for Assignment 3

	Light source classes

***********************************************************/
#pragma once

#include "util.h"
#include "scene_object.h"
#include <vector>
#include <typeinfo>

// Base class for a light source.  You could define different types
// of lights here, but point light is sufficient for most scenes you
// might want to render.  Different light sources shade the ray
// differently.
class LightSource {
public:
	__host__ __device__
	virtual void shade(Ray3D&, bool phong, Scene& scene, bool shadows) = 0;
	__host__ __device__
	virtual Point3D get_position() const = 0;
	virtual ~LightSource() {}
};

// List of all light sources in your scene
//typedef std::vector<LightSource*> LightList;
typedef ContainerVec<LightSource*> LightList;


// A point light is defined by its position in world space and its
// color.
class PointLight : public LightSource {
public:
	__host__ __device__
	PointLight(Point3D pos, Color col)
		:
		pos(pos), col_ambient(col), col_diffuse(col), col_specular(col) {}

	__host__ __device__
	PointLight(Point3D pos, Color ambient, Color diffuse, Color specular)
		:
		pos(pos), col_ambient(ambient), col_diffuse(diffuse), col_specular(specular) {}

	__host__ __device__
	void shade(Ray3D& ray, bool phong, Scene& scene, bool shadows);

	//Color ray_diffuse(Ray3D ray);

	__host__ __device__
	Point3D get_position() const { return pos; }
	__host__ __device__
	Color get_col_ambient() const { return col_ambient; }
	__host__ __device__
	Color get_col_diffuse() const { return col_diffuse; }
	__host__ __device__
	Color get_col_specular() const { return col_specular; }

private:
	Point3D pos;
	Color col_ambient;
	Color col_diffuse;
	Color col_specular;
};

class AreaLight : public LightSource {
public:
	__host__ __device__
	AreaLight(int N, float dist, Point3D cent, Color ambient, Color diffuse, Color specular)
		:
		N(N), dist(dist), centre(cent), col_ambient(ambient), col_diffuse(diffuse), col_specular(specular) {}

	__host__ __device__
	AreaLight(int N, float dist, Point3D cent, Color col)
		:
		N(N), dist(dist), centre(cent), col_ambient(col), col_diffuse(col), col_specular(col) {}

	__host__ __device__
	void shade(Ray3D& ray, bool phong, Scene& scene, bool shadows);

	__host__ __device__
	Point3D get_position() const { return centre; }
	__host__ __device__
	Color get_col_ambient() const { return col_ambient; }
	__host__ __device__
	Color get_col_diffuse() const { return col_diffuse; }
	__host__ __device__
	Color get_col_specular() const { return col_specular; }


private:
	int N;
	float dist;
	Point3D centre;
	Color col_ambient;
	Color col_diffuse;
	Color col_specular;
};


/*
class LightList {

public:
	__host__ __device__ LightList() : arr_size{ 0 } {
		capacity = 0;
		arr = nullptr;
	}

	__host__ void push_back(LightSource* t) {
		if (arr_size == 0) {
			arr = new LightSource * [4];
			arr[0] = t;
			arr_size++;
			capacity = 4;
		}
		else if (arr_size % 2) {
			LightSource** old = arr;
			capacity *= 2;
			arr = new LightSource * [capacity];

			for (int i = 0; i < arr_size; i++) {
				arr[i] = old[i];
			}

			delete[] old;

			arr[arr_size] = t;
			arr_size++;
		}
	}

	__host__ __device__ LightSource*& operator[](int i) {
		return arr[i];
	}

	__host__ __device__ int size() {
		return arr_size;
	}

	LightSource** arr;

private:

	int capacity;
	int arr_size;
};
*/