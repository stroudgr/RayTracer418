/***********************************************************

	Starter code for Assignment 3

	Implements light_source.h

***********************************************************/
#include "util.h"
#include <cmath>
#include "light_source.h"

__host__ __device__
void PointLight::shade(Ray3D& ray, bool phong, Scene& scene, bool shadows) {
	// TODO: implement this function to fill in values for ray.col
	// using phong shading.  Make sure your vectors are normalized, and
	// clamp colour values to 1.0.
	//
	// It is assumed at this point that the intersection information in ray
	// is available.  So be sure that traverseScene() is called on the ray
	// before this function.

	Vector3D sourceOfLight = this->get_position() - ray.intersection.point;
	sourceOfLight.normalize();

	Vector3D view = -ray.dir;
	view.normalize();

	Vector3D normal = ray.intersection.normal;
	normal.normalize();

	float normalDotLight = sourceOfLight.dot(normal);
	Vector3D reflect = (2 * normalDotLight) * normal - sourceOfLight;
	reflect.normalize();
	float reflectDotView = fmax(0.0, view.dot(reflect));

	normalDotLight = fmax(normalDotLight, 0.0);

	Color ray_ambient = ray.intersection.mat->ambient * this->col_ambient;
	Color ray_diffuse = normalDotLight * ray.intersection.mat->diffuse * this->col_diffuse;
	Color ray_specular = pow(reflectDotView, ray.intersection.mat->specular_exp) * (ray.intersection.mat->specular * this->col_specular);

	if (phong) {
		ray.col = ray.col + ray_diffuse;
		ray.col = ray.col + ray_ambient;
		ray.col = ray.col + ray_specular;
	}

	//Just ambient
	else {
		ray.col = ray.col + ray_ambient;
	}

	//Done in raytracer
	//ray.col.clamp();
}

__host__ __device__
void AreaLight::shade(Ray3D& ray, bool phong, Scene& scene, bool shadows) {

	for (int i = 0; i < this->N; i++) {
		for (int j = 0; j < this->N; j++) {
			Vector3D cent(this->centre[0], this->centre[1], this->centre[2]);
			PointLight light(Point3D(this->dist * (i - N / 2), this->dist * (j - N / 2), 0) + cent,
				get_col_ambient(), get_col_diffuse(), get_col_specular());

			// Shadows for point lights are handled in the raytracer (computeShading), and are handled here for
			//   AreaLight, because soft shadows aren't produced by without the shadow detecting code below.

			Vector3D l = light.get_position() - ray.intersection.point;
			l.normalize();

			Ray3D shadowRay;
			shadowRay.origin = ray.intersection.point;
			shadowRay.dir = light.get_position() - shadowRay.origin;
			shadowRay.dir.normalize();

			// Shadow detect
			float ep = 0.1;
			for (size_t i = 0; i < scene.size(); ++i) {
				SceneNode* node = scene[i];
				if (node->obj->intersect(shadowRay, node->worldToModel, node->modelToWorld, 0)) {
					if (!shadowRay.intersection.none && (shadowRay.intersection.t_value > ep || shadowRay.intersection.t_value < -ep)) {
						break;
					}
					else {
						shadowRay.intersection.none = true;
					}
				}
			}

			if (shadows) {
				if (!shadowRay.intersection.none)
					light.shade(ray, false, scene, shadows); // just ambient
				else
					light.shade(ray, true, scene, shadows); //all three
			}
			else
				light.shade(ray, true, scene, shadows);
		}
	}
	ray.col = (1.0 / (this->N * this->N)) * ray.col;
}
