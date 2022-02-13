/***********************************************************

	 Starter code for Assignment 3


	Classes defining primitives in the scene

***********************************************************/
#pragma once

#include "util.h"
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// All primitives should provide an intersection function.
// To create more primitives, inherit from SceneObject.
// Namely, you can create, Sphere, Cylinder, etc... classes
// here.
class SceneObject {
public:
	// Returns true if an intersection occured, false otherwise.
	__host__ __device__
	virtual bool intersect(Ray3D&, const Matrix4x4&, const Matrix4x4&, int debug) = 0;
	__host__ __device__
	virtual ~SceneObject() {}
};

// Scene node containing information about an object: geometry, material,
// tranformations.
struct SceneNode {
	__host__ __device__
	SceneNode()
		:
		obj(NULL), mat(NULL) {}

	__host__ __device__
	SceneNode(SceneObject* obj, Material* mat)
		:
		obj(obj), mat(mat) {}

	__host__ __device__
	~SceneNode() {
		if (obj) delete obj;
	}

	// Apply rotation about axis 'x', 'y', 'z' angle degrees to node.
	__host__ __device__
	void rotate(char axis, double angle);

	// Apply translation in the direction of trans to node.
	__host__ __device__
	void translate(Vector3D trans);

	// Apply scaling about a fixed point origin.
	__host__ __device__
	void scale(Point3D origin, double factor[3]);

	// Pointer to geometry primitive, used for intersection.
	SceneObject* obj;

	// Pointer to material of the object, used in shading.
	Material* mat;

	// Each node maintains a transformation matrix, which maps the
	// geometry from object space to world space and the inverse.
	Matrix4x4 trans;
	Matrix4x4 invtrans;
	Matrix4x4 modelToWorld;
	Matrix4x4 worldToModel;
};

// Scene is simply implemented as a list of nodes. Doesnt support hierarchy(scene graph).
//typedef std::vector<SceneNode*> Scene;
//typedef thrust::host_vector<SceneNode*> HScene;
//typedef thrust::device_vector<SceneNode*> Scene;


typedef ContainerVec<SceneNode*> Scene;



// Example primitive you can create, this is a unit square on
// the xy-plane.
class UnitSquare : public SceneObject {
public:
	__host__ __device__
	bool intersect(Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld, int debug);
};

class UnitSphere : public SceneObject {
public:
	__host__ __device__
	bool intersect(Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld, int debug);
};

class UnitCone : public SceneObject {
public:
	__host__ __device__
	bool intersect(Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld, int debug);
};




/*class Scene {

public:
	__host__ __device__ Scene() : arr_size{ 0 } {
		capacity = 0;
		arr = nullptr;
	}

	__host__ void push_back(SceneNode* t) {
		if (arr_size == 0) {
			arr = new SceneNode * [4];
			arr[0] = t;
			arr_size++;
			capacity = 4;
		}
		else if (arr_size % 2) {
			SceneNode** old = arr;
			capacity *= 2;
			arr = new SceneNode * [capacity];

			for (int i = 0; i < arr_size; i++) {
				arr[i] = old[i];
			}

			delete[] old;

			arr[arr_size] = t;
			arr_size++;
		}
	}

	__host__ __device__ SceneNode*& operator[](int i) {
		return arr[i];
	}

	__host__ __device__ int size() {
		return arr_size;
	}

	SceneNode** arr;

private:

	int capacity;
	int arr_size;
};*/