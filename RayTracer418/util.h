/***********************************************************

	Starter code for Assignment 3

	Utility functions and structures (based on code from CGL, University of Waterloo),
	modify this file as you see fit.

***********************************************************/
//#pragma once


#ifndef __UTIL__H
#define __UTIL__H

#include <iostream>
#include <cmath>
#include "bmp_io.h"

#include "cuda_runtime.h"

#ifndef M_PI
#define M_PI	3.14159265358979323846
#endif

class Point3D {
public:
	__host__ __device__
	Point3D();
	__host__ __device__
	Point3D(double x, double y, double z);
	__host__ __device__
	Point3D(const Point3D& other);

	__host__ __device__
	Point3D& operator =(const Point3D& other);
	__host__ __device__
	double& operator[](int i);
	__host__ __device__
	double operator[](int i) const;

private:
	double m_data[3];
};

class Vector3D {
public:
	__host__ __device__
	Vector3D();
	__host__ __device__
	Vector3D(double x, double y, double z);
	__host__ __device__
	Vector3D(const Vector3D& other);

	__host__ __device__
	Vector3D& operator =(const Vector3D& other);
	__host__ __device__
	double& operator[](int i);
	__host__ __device__
	double operator[](int i) const;

	__host__ __device__
	double length() const;
	__host__ __device__
	double normalize();
	__host__ __device__
	double dot(const Vector3D& other) const;
	__host__ __device__
	Vector3D cross(const Vector3D& other) const;

private:
	double m_data[3];
};

// standard operators on points and vectors
__host__ __device__
Vector3D operator *(double s, const Vector3D& v);
__host__ __device__
Vector3D operator +(const Vector3D& u, const Vector3D& v);
__host__ __device__
Point3D operator +(const Point3D& u, const Vector3D& v);
__host__ __device__
Vector3D operator -(const Point3D& u, const Point3D& v);
__host__ __device__
Vector3D operator -(const Vector3D& u, const Vector3D& v);
__host__ __device__
Vector3D operator -(const Vector3D& u);
__host__ __device__
Point3D operator -(const Point3D& u, const Vector3D& v);
__host__ __device__
Vector3D cross(const Vector3D& u, const Vector3D& v);
std::ostream& operator <<(std::ostream& o, const Point3D& p);
std::ostream& operator <<(std::ostream& o, const Vector3D& v);

class Vector4D {
public:
	__host__ __device__
	Vector4D();
	__host__ __device__
	Vector4D(double w, double x, double y, double z);
	__host__ __device__
	Vector4D(const Vector4D& other);

	__host__ __device__
	Vector4D& operator =(const Vector4D& other);
	__host__ __device__
	double& operator[](int i);
	__host__ __device__
	double operator[](int i) const;

private:
	double m_data[4];
};

class Matrix4x4 {
public:
	__host__ __device__
	Matrix4x4();
	__host__ __device__
	Matrix4x4(const Matrix4x4& other);
	__host__ __device__
	Matrix4x4& operator=(const Matrix4x4& other);

	__host__ __device__
	Vector4D getRow(int row) const;
	__host__ __device__
	double* getRow(int row);
	__host__ __device__
	Vector4D getColumn(int col) const;

	__host__ __device__
	Vector4D operator[](int row) const;
	__host__ __device__
	double* operator[](int row);
	__host__ __device__
	Matrix4x4 transpose() const;

private:
	double m_data[16];
};

__host__ __device__
Matrix4x4 operator *(const Matrix4x4& M, const Matrix4x4& N);
__host__ __device__
Vector3D operator *(const Matrix4x4& M, const Vector3D& v);
__host__ __device__
Point3D operator *(const Matrix4x4& M, const Point3D& p);
// Multiply n by the transpose of M, useful for transforming normals.
// Recall that normals should be transformed by the inverse transpose
// of the matrix.
__host__ __device__
Vector3D transNorm(const Matrix4x4& M, const Vector3D& n);
std::ostream& operator <<(std::ostream& os, const Matrix4x4& M);

class Color {
public:
	__host__ __device__
	Color();
	__host__ __device__
	Color(double r, double g, double b);
	__host__ __device__
	Color(const Color& other);

	__host__ __device__
	Color& operator =(const Color& other);
	__host__ __device__
	Color operator *(const Color& other);
	__host__ __device__
	double& operator[](int i);
	__host__ __device__
	double operator[](int i) const;

	__host__ __device__
	void clamp();

private:
	double m_data[3];
};

__host__ __device__
Color operator *(double s, const Color& c);
__host__ __device__
Color operator +(const Color& u, const Color& v);
__host__
std::ostream& operator <<(std::ostream& o, const Color& c);

struct Material {
	__host__ __device__
	Material(Color ambient, Color diffuse, Color specular, double exp, double mirror, double refract, double transparent) :
		ambient(ambient), diffuse(diffuse), specular(specular),
		specular_exp(exp), mirror(mirror), refract(refract), transparent(transparent) { }

	// Ambient components for Phong shading.
	Color ambient;
	// Diffuse components for Phong shading.
	Color diffuse;
	// Specular components for Phong shading.
	Color specular;
	// Specular expoent.
	double specular_exp;
	//mirrorness
	double mirror;
	// i.e. n2/n1
	double refract;
	double transparent;

};

struct Intersection {
	// Location of intersection.
	Point3D point;
	// Normal at the intersection.
	Vector3D normal;
	// Material at the intersection.
	Material* mat;
	// Position of the intersection point on your ray.
	// (i.e. point = ray.origin + t_value * ray.dir)
	// This is used when you need to intersect multiply objects and
	// only want to keep the nearest intersection.
	double t_value;
	// Set to true when no intersection has occured.
	bool none;
};

// Ray structure.
struct Ray3D {
	__host__ __device__
	Ray3D() {
		intersection.none = true;
	}
	__host__ __device__
	Ray3D(Point3D p, Vector3D v) : origin(p), dir(v) {
		intersection.none = true;
	}
	// Origin and direction of the ray.
	Point3D origin;
	Vector3D dir;
	// Intersection status, should be computed by the intersection
	// function.
	Intersection intersection;
	// Current colour of the ray, should be computed by the shading
	// function.
	Color col;
};

struct Camera {
	// Camera positioned at eye, with view vector view, up vector up, and
	// field of view fov.
	__host__ __device__
	Camera(Point3D eye, Vector3D view, Vector3D up, double fov)
		:
		eye(eye), view(view), up(up), fov(fov)
	{}

	// Constructs a view to world transformation matrix based on the
	// camera parameters.
	__host__ __device__
	Matrix4x4 initInvViewMatrix() {
		Matrix4x4 mat;
		Vector3D w;
		view.normalize();
		up = up - up.dot(view) * view;
		up.normalize();
		w = view.cross(up);

		mat[0][0] = w[0];
		mat[1][0] = w[1];
		mat[2][0] = w[2];
		mat[0][1] = up[0];
		mat[1][1] = up[1];
		mat[2][1] = up[2];
		mat[0][2] = -view[0];
		mat[1][2] = -view[1];
		mat[2][2] = -view[2];
		mat[0][3] = eye[0];
		mat[1][3] = eye[1];
		mat[2][3] = eye[2];

		return mat;
	}

	Point3D eye;
	Vector3D view;
	Vector3D up;
	double fov;
};









struct Image {

	// Create an image with dimensions (width, height)
	__host__
	Image(int width, int height)
		:
		width(width), height(height), rbuffer(NULL), gbuffer(NULL), bbuffer(NULL) {
		initPixelBuffer();
	}

	~Image() {
		if (rbuffer) delete rbuffer;
		if (gbuffer) delete gbuffer;
		if (bbuffer) delete bbuffer;
	}

	// Allocates and initializes the pixel buffer for rendering, you
	// could add an interesting background to your scene by modifying
	// this function.
	__host__
	void initPixelBuffer() {
		int numbytes = width * height * sizeof(unsigned char);
		rbuffer = new unsigned char[numbytes];
		std::fill_n(rbuffer, numbytes, 0);
		gbuffer = new unsigned char[numbytes];
		std::fill_n(gbuffer, numbytes, 0);
		bbuffer = new unsigned char[numbytes];
		std::fill_n(bbuffer, numbytes, 0);
	}

	// Saves the pixel buffer as image to a file at path
	__host__
	void flushPixelBuffer(std::string path) {
		bmp_write(path.c_str(), width, height, rbuffer, gbuffer, bbuffer);
	}

	// Set color of pixel (i,j) to col
	__host__ __device__
	void setColorAtPixel(int i, int j, Color& col) {
		rbuffer[i * width + j] = int(col[0] * 255);
		gbuffer[i * width + j] = int(col[1] * 255);
		bbuffer[i * width + j] = int(col[2] * 255);
	}

	Image* toCuda() {

		return nullptr;
	}


	int width;
	int height;

//private:
	// Pixel buffer.
	unsigned char* rbuffer; // red channel
	unsigned char* gbuffer; // green channel
	unsigned char* bbuffer; // blue channel
};




enum ShadeType { ambientDiffuse, ambient, specular };






template <class T>
class ContainerVec {

public:
	__host__ __device__ ContainerVec() : arr_size{ 0 } {
		capacity = 8;
		//capacity = 0;
	}

	/*__host__ void push_back(T t) {
		if (arr_size == 0) {
			arr = new T[4];
			arr[0] = t;
			arr_size++;
			capacity = 4;
		}
		else if (arr_size % 2) {
			T* old = arr;
			capacity *= 2;
			arr = new T[capacity];

			for (int i = 0; i < arr_size; i++) {
				arr[i] = old[i];
			}

			delete[] old;

			arr[arr_size] = t;
			arr_size++;
		}
	}*/



	__host__ __device__ void push_back(T t) {
		if (arr_size < capacity) {
			arr[arr_size] = t;
			arr_size++;
		}
	}

	__host__ __device__ T& operator[](int i) {
		return arr[i];
	}

	__host__ __device__ int size() {
		return arr_size;
	}

	T arr[8]; 
	//T*arr;

private:
	
	int capacity;
	int arr_size;
};



#endif
