/***********************************************************

	Starter code for Assignment 3

	Implements scene_object.h

***********************************************************/

#include <cmath>
#include "scene_object.h"

__host__ __device__
bool UnitSquare::intersect(Ray3D& ray, const Matrix4x4& worldToModel,
	const Matrix4x4& modelToWorld, int debug) {
	// TODO: implement intersection code for UnitSquare, which is
	// defined on the xy-plane, with vertices (0.5, 0.5, 0),
	// (-0.5, 0.5, 0), (-0.5, -0.5, 0), (0.5, -0.5, 0), and normal
	// (0, 0, 1).
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point,
	// intersection.normal, intersection.none, intersection.t_value.
	//
	// HINT: Remember to first transform the ray into object space
	// to simplify the intersection test.

	Vector3D rayDir = worldToModel * ray.dir;
	Point3D rayOrg = worldToModel * ray.origin;
	// ray is rayOrg + t*rayDir

	Vector3D n = Vector3D(0.0, 0.0, 1.0);

	double dn = n.dot(rayDir); //== rayDir[2]

	if (dn == 0.0) {

		return false;
	}
	else {

		Vector3D rayOriginVec = rayOrg - Point3D(0.0, 0.0, 0.0);
		double t = -(rayOriginVec.dot(n)) / dn;
		Point3D inter = rayOrg + t * rayDir;

		if (t < 0 || inter[1] > 0.5 || inter[1] < -0.5 || inter[0] > 0.5 || inter[0] < -0.5) {
			//ray.intersection.none = true;
			return false;
		}

		if (ray.intersection.none || ray.intersection.t_value > t) {

			ray.intersection.none = false;
			ray.intersection.normal = worldToModel.transpose() * n;
			ray.intersection.point = modelToWorld * inter;
			ray.intersection.t_value = t;

			return true;

		}
		return false;
	}
	return false;
}

__host__ __device__
bool UnitSphere::intersect(Ray3D& ray, const Matrix4x4& worldToModel,
	const Matrix4x4& modelToWorld, int debug) {
	// TODO: implement intersection code for UnitSphere, which is centred
	// on the origin.
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point,
	// intersection.normal, intersection.none, intersection.t_value.
	//
	// HINT: Remember to first transform the ray into object space
	// to simplify the intersection test.

	return false;

	Vector3D rayDir = worldToModel * ray.dir;
	Vector3D o = worldToModel * ray.origin - Point3D(0.0, 0.0, 0.0);

	Point3D rayOrg = Point3D(o[0], o[1], o[2]);

	// p(t) = t*rayDir + rayOrg   is the ray,  and (p-cent)*(p-cent) - 1 = 0  is the plane
	// (t*rayDir + rayOrg)*(t*rayDir + rayOrg) -1 = 0
	// t^2(rayDir^2) + 2trayDir*rayDir + rayOrg^2 - 1 = 0

	// (d*d)t^2 + 2d(e-centre)t + (e-cent)^2 - 1 = 0

	float A = rayDir.dot(rayDir);
	float B = 2 * (rayDir.dot(o));
	float C = o.dot(o) - 1;

	if (B * B - 4 * A * C < 0.0) {
		return false;
	}

	float t1 = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
	float t2 = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);


	float t = -1;
	if (t1 <= 0 && t2 <= 0)
		return false;
	else if (t1 > 0 && t2 <= 0)
		t = t1;
	else if (t1 <= 0 && t2 > 0)
		t = t2;
	else
		t = fmin(t1, t2);

	if (t < 0) {
		return false;
	}

	//if (debug){printf("Hit sphere at %f\n", t);}
	if (ray.intersection.none || ray.intersection.t_value > t) {

		Point3D inter = worldToModel * ray.origin + (t * rayDir);

		ray.intersection.none = false;
		ray.intersection.t_value = t;
		ray.intersection.point = modelToWorld * inter;

		//Normal is grad(x^2 + y^2 + z^z - 1) = (2x,2y,2z)

		ray.intersection.normal = Vector3D(inter[0], inter[1], inter[2]);
		ray.intersection.normal = worldToModel.transpose() * ray.intersection.normal;
		ray.intersection.normal.normalize();

	}

	return true;
}
__host__ __device__
bool UnitCone::intersect(Ray3D& ray, const Matrix4x4& worldToModel,
	const Matrix4x4& modelToWorld, int debug) {

	bool hitBottom = false;
	float t_value = 0;

	Vector3D rayDir = worldToModel * ray.dir;
	Point3D rayOrg = worldToModel * ray.origin;

	// The cone's tip is at (0,0,0)
	// and the base circle is (cos(t), sin(t), 1)
	// z^2 - x^2 - y^2 = 0

	// Intersect circle at the bottom (r*cost, r*sint, 1)
	// p = rayOrg + t*rayDir
	// rayOrg[2] + t*rayDir[2] = 1
	//t = (1 - rayOrg[2] )/rayDir[2]

	//NOTE didn't bother to finish implementing hitting the bottom of the cone
	/*if (rayDir[2]) {
		//normal = (0,0,1)
		t_value = (1 - rayOrg[2])/rayDir[2];

		Point3D inter = rayOrg + t_value*rayDir;

		if (t_value > 0 && inter[0]*inter[0] + inter[1]*inter[1] < 1 && (ray.intersection.none || ray.intersection.t_value > t_value) ){
			ray.intersection.none = false;
			ray.intersection.t_value = t_value;
			ray.intersection.point = inter;

			ray.intersection.normal = Vector3D(0,0,1);
			ray.intersection.normal.normalize();
			ray.intersection.normal = worldToModel.transpose() *ray.intersection.normal;

			hitBottom = true;
		}
	}*/


	// z^2 = x^2 + y^2
	// (rayOrg[2] + t*rayDir[2])^2 = (rayOrg[0] + t*rayDir[0])^2 + (rayOrg[1] + t*rayDir[1])^2
	// (rayOrg[2])^2 + t*2rayDir[2]rayOrg[2]  + t^2*rayDir[2]^2 =
	//              rayOrg[0]^2 + t*2rayDir[0]rayOrg[0]  + t^2*rayDir[0]^2 +  rayOrg[1]^2 + t*2rayDir[1]rayOrg[1]  + t^2*rayDir[1]^2

	// t^2(rayDir[2])^2 - rayDir[0]^2 - rayDir[1]^2)
  // + t*2(rayDir[2]rayOrg[2] - rayDir[0]rayOrg[0] - rayDir[1]rayOrg[1])
	// + ((1-rayOrg[2])^2 - rayOrg[0]^2 -  rayOrg[1]^2) = 0

	float A = rayDir[2] * rayDir[2] - rayDir[0] * rayDir[0] - rayDir[1] * rayDir[1];
	float B = 2 * (rayDir[2] * rayOrg[2] - rayDir[0] * rayOrg[0] - rayDir[1] * rayOrg[1]);
	float C = (rayOrg[2]) * (rayOrg[2]) - rayOrg[0] * rayOrg[0] - rayOrg[1] * rayOrg[1];

	if (B * B - 4 * A * C < 0.0) {
		return hitBottom; //hitBottom always is false, since detection is not implemented
	}

	float t1 = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
	float t2 = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);

	float t;
	if (t1 <= 0 && t2 <= 0) {
		return hitBottom;
	}
	else if (t1 > 0 && t2 < 0)
		t = t1;
	else if (t1 < 0 && t2 > 0)
		t = t2;
	else
		t = fmin(t1, t2);

	if (t < 0) {
		return hitBottom;
	}

	Point3D intersection = rayOrg + (t * rayDir);

	/*if ((intersection[0] < 0 || intersection[0] > 1)) {
		return false;
	}*/

	if ((intersection[2] < 0 || intersection[2] > 1)) {
		return hitBottom;
	}

	if (hitBottom && ray.intersection.t_value < t) {
		return true;
	}

	ray.intersection.none = false;
	ray.intersection.t_value = t;
	ray.intersection.point = modelToWorld * intersection;

	//Normal is grad(z^2- x^2 - y^2) = (-2x,-2y,2z)

	ray.intersection.normal = Vector3D(intersection[0], intersection[1], -intersection[2]);
	ray.intersection.normal.normalize();
	ray.intersection.normal = worldToModel.transpose() * ray.intersection.normal;

	return true;
}



void SceneNode::rotate(char axis, double angle) {
	Matrix4x4 rotation;
	double toRadian = 2 * M_PI / 360.0;
	int i;

	for (i = 0; i < 2; i++) {
		switch (axis) {
		case 'x':
			rotation[0][0] = 1;
			rotation[1][1] = cos(angle * toRadian);
			rotation[1][2] = -sin(angle * toRadian);
			rotation[2][1] = sin(angle * toRadian);
			rotation[2][2] = cos(angle * toRadian);
			rotation[3][3] = 1;
			break;
		case 'y':
			rotation[0][0] = cos(angle * toRadian);
			rotation[0][2] = sin(angle * toRadian);
			rotation[1][1] = 1;
			rotation[2][0] = -sin(angle * toRadian);
			rotation[2][2] = cos(angle * toRadian);
			rotation[3][3] = 1;
			break;
		case 'z':
			rotation[0][0] = cos(angle * toRadian);
			rotation[0][1] = -sin(angle * toRadian);
			rotation[1][0] = sin(angle * toRadian);
			rotation[1][1] = cos(angle * toRadian);
			rotation[2][2] = 1;
			rotation[3][3] = 1;
			break;
		}
		if (i == 0) {
			this->trans = this->trans * rotation;
			angle = -angle;
		}
		else {
			this->invtrans = rotation * this->invtrans;
		}
	}
}

void SceneNode::translate(Vector3D trans) {
	Matrix4x4 translation;

	translation[0][3] = trans[0];
	translation[1][3] = trans[1];
	translation[2][3] = trans[2];
	this->trans = this->trans * translation;
	translation[0][3] = -trans[0];
	translation[1][3] = -trans[1];
	translation[2][3] = -trans[2];
	this->invtrans = translation * this->invtrans;
}

void SceneNode::scale(Point3D origin, double factor[3]) {
	Matrix4x4 scale;

	scale[0][0] = factor[0];
	scale[0][3] = origin[0] - factor[0] * origin[0];
	scale[1][1] = factor[1];
	scale[1][3] = origin[1] - factor[1] * origin[1];
	scale[2][2] = factor[2];
	scale[2][3] = origin[2] - factor[2] * origin[2];
	this->trans = this->trans * scale;
	scale[0][0] = 1 / factor[0];
	scale[0][3] = origin[0] - 1 / factor[0] * origin[0];
	scale[1][1] = 1 / factor[1];
	scale[1][3] = origin[1] - 1 / factor[1] * origin[1];
	scale[2][2] = 1 / factor[2];
	scale[2][3] = origin[2] - 1 / factor[2] * origin[2];
	this->invtrans = scale * this->invtrans;
}
