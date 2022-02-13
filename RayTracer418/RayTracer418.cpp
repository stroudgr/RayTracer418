/***********************************************************

	Starter code for Assignment 3

***********************************************************/

#include <cstdlib>
#include "raytracer.cuh"
#include "world.h"



int main(int argc, char* argv[])
{
	// Build your scene and setup your camera here, by calling
	// functions from Raytracer.  The code here sets up an example
	// scene and renders it from two different view points, DO NOT
	// change this if you're just implementing part one of the
	// assignment.


	//World w(true);

	Raytracer raytracer("aa", false);
	LightList light_list;
	Scene scene;

	int width = 64;//1024;//320; //1080;
	int height = 64;//1024;//240; //810;

	if (argc == 3) {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
	}

	// Define materials for shading.
	Material gold(Color(0.3, 0.3, 0.3), Color(0.75164, 0.60648, 0.22648),
		Color(0.628281, 0.555802, 0.366065),
		51.2, 0.05, 0, 0);
	Material jade(Color(0, 0, 0), Color(0.54, 0.89, 0.63),
		Color(0.316228, 0.316228, 0.316228),
		12.8, 0.05, 0, 0);
	Material red(Color(0, 0, 0), Color(0.54, 0.09, 0.03),
		Color(0.516228, 0.016228, 0.016228),
		12.8, 0.3, 0, 0);
	Material blue(Color(0, 0, 0), Color(0.04, 0.09, 0.73),
		Color(0.016228, 0.86228, 0.016228),
		12.8, 0.0, 1 / 0.97, 0.75);

	Material grey(Color(0.2, 0.2, 0.2), Color(0.5, 0.5, 0.5),
		Color(0.5, 0.5, 0.5),
		12.8, 0.4, 0, 0);

	Material orange(0.3 * Color(1, 0.39, 0.33), Color(1, 0.39, 0.33),
		Color(0.5, 0.5, 0.5),
		12.8, 0.3, 0, 0);

	// Defines a point light source.
	PointLight* pLight = new PointLight(Point3D(0, 0, 5), Color(0.9, 0.9, 0.9));
	light_list.push_back(pLight);

	// Add a unit square into the scene with material mat.
	SceneNode* sphere = new SceneNode(new UnitSphere(), &gold);
	//scene.push_back(sphere);

	SceneNode* sphere2 = new SceneNode(new UnitSphere(), &red);
	scene.push_back(sphere2);
	SceneNode* sphere3 = new SceneNode(new UnitSphere(), &jade);
	//scene.push_back(sphere3);
	SceneNode* sphere4 = new SceneNode(new UnitSphere(), &gold);
	//scene.push_back(sphere4);
	SceneNode* sphere5 = new SceneNode(new UnitSphere(), &blue);
	//scene.push_back(sphere5);

	SceneNode* cone = new SceneNode(new UnitCone(), &orange);
	//scene.push_back(cone);


	SceneNode* plane = new SceneNode(new UnitSquare(), &grey);
	//scene.push_back(plane);

	// Apply some transformations to the sphere and unit square.
	double factor1[3] = { 0.5, 0.5, 0.25 };
	//sphere->translate(Vector3D(0, 0, -5));
	//sphere->rotate('x', -45);
	//sphere->rotate('z', 45);
	//sphere->scale(Point3D(0, 0, 0), factor1);

	sphere2->translate(Vector3D(0, 0, -5));
	sphere3->translate(Vector3D(2, 0, -4.5));
	sphere4->translate(Vector3D(-2.2, 0, -4.5));
	sphere5->translate(Vector3D(1, -0.5, -2));
	sphere5->scale(Point3D(0, 0, 0), factor1);

	double factorc[3] = { 0.75, 0.5, 0.5 };
	cone->translate(Vector3D(-1, -0.5, -2));
	cone->rotate('z', -90);
	cone->rotate('y', 90);
	cone->scale(Point3D(0, 0, 0), factorc);

	double factor2[3] = { 14.0, 14.0, 14.0 };
	plane->translate(Vector3D(0, -1, -4));
	//plane->rotate('z', 45);
	plane->rotate('x', -90);
	plane->scale(Point3D(0, 0, 0), factor2);

	// Render the scene, feel free to make the image smaller for
	// testing purposes.
	Camera camera1(Point3D(0, 0, 1), Vector3D(0, 0, -1), Vector3D(0, 1, 0), 60.0);
	Image image1(width, height);



	raytracer.render(camera1, scene, light_list, image1); //render 3D scene to image

	image1.flushPixelBuffer("view1.bmp"); //save rendered image to file

	// Render it from a different point of view.
	//Camera camera2(Point3D(4, 2, 1), Vector3D(-4, -2, -6), Vector3D(0, 1, 0), 60.0);
	//Image image2(width, height);
	//raytracer.render(camera2, scene, light_list, image2);
	//image2.flushPixelBuffer("images/view2.bmp");

	// Free memory
	for (size_t i = 0; i < scene.size(); ++i) {
		delete scene[i];
	}

	for (size_t i = 0; i < light_list.size(); ++i) {
		delete light_list[i];
	}

	return 0;
}












/*int main(int argc, char* argv[])
{
	// Build your scene and setup your camera here, by calling
	// functions from Raytracer.  The code here sets up an example
	// scene and renders it from two different view points, DO NOT
	// change this if you're just implementing part one of the
	// assignment.
	Raytracer raytracer("aa", false);
	LightList light_list;
	Scene scene;

	int width = 64;//1024;//320; //1080;
	int height = 64;//1024;//240; //810;

	if (argc == 3) {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
	}

	// Define materials for shading.
	Material gold(Color(0.3, 0.3, 0.3), Color(0.75164, 0.60648, 0.22648),
		Color(0.628281, 0.555802, 0.366065),
		51.2, 0.05, 0, 0);
	Material jade(Color(0, 0, 0), Color(0.54, 0.89, 0.63),
		Color(0.316228, 0.316228, 0.316228),
		12.8, 0.05, 0, 0);
	Material red(Color(0, 0, 0), Color(0.54, 0.09, 0.03),
		Color(0.516228, 0.016228, 0.016228),
		12.8, 0.3, 0, 0);
	Material blue(Color(0, 0, 0), Color(0.04, 0.09, 0.73),
		Color(0.016228, 0.86228, 0.016228),
		12.8, 0.0, 1 / 0.97, 0.75);

	Material grey(Color(0.2, 0.2, 0.2), Color(0.5, 0.5, 0.5),
		Color(0.5, 0.5, 0.5),
		12.8, 0.4, 0, 0);

	Material orange(0.3 * Color(1, 0.39, 0.33), Color(1, 0.39, 0.33),
		Color(0.5, 0.5, 0.5),
		12.8, 0.3, 0, 0);

	// Defines a point light source.
	PointLight* pLight = new PointLight(Point3D(0, 0, 5), Color(0.9, 0.9, 0.9));
	light_list.push_back(pLight);

	// Add a unit square into the scene with material mat.
	SceneNode* sphere = new SceneNode(new UnitSphere(), &gold);
	//scene.push_back(sphere);

	SceneNode* sphere2 = new SceneNode(new UnitSphere(), &red);
	scene.push_back(sphere2);
	SceneNode* sphere3 = new SceneNode(new UnitSphere(), &jade);
	//scene.push_back(sphere3);
	SceneNode* sphere4 = new SceneNode(new UnitSphere(), &gold);
	//scene.push_back(sphere4);
	SceneNode* sphere5 = new SceneNode(new UnitSphere(), &blue);
	//scene.push_back(sphere5);

	SceneNode* cone = new SceneNode(new UnitCone(), &orange);
	//scene.push_back(cone);


	SceneNode* plane = new SceneNode(new UnitSquare(), &grey);
	//scene.push_back(plane);

	// Apply some transformations to the sphere and unit square.
	double factor1[3] = { 0.5, 0.5, 0.25 };
	//sphere->translate(Vector3D(0, 0, -5));
	//sphere->rotate('x', -45);
	//sphere->rotate('z', 45);
	//sphere->scale(Point3D(0, 0, 0), factor1);

	sphere2->translate(Vector3D(0, 0, -5));
	sphere3->translate(Vector3D(2, 0, -4.5));
	sphere4->translate(Vector3D(-2.2, 0, -4.5));
	sphere5->translate(Vector3D(1, -0.5, -2));
	sphere5->scale(Point3D(0, 0, 0), factor1);

	double factorc[3] = { 0.75, 0.5, 0.5 };
	cone->translate(Vector3D(-1, -0.5, -2));
	cone->rotate('z', -90);
	cone->rotate('y', 90);
	cone->scale(Point3D(0, 0, 0), factorc);

	double factor2[3] = { 14.0, 14.0, 14.0 };
	plane->translate(Vector3D(0, -1, -4));
	//plane->rotate('z', 45);
	plane->rotate('x', -90);
	plane->scale(Point3D(0, 0, 0), factor2);

	// Render the scene, feel free to make the image smaller for
	// testing purposes.
	Camera camera1(Point3D(0, 0, 1), Vector3D(0, 0, -1), Vector3D(0, 1, 0), 60.0);
	Image image1(width, height);



	raytracer.render(camera1, scene, light_list, image1); //render 3D scene to image
	
	image1.flushPixelBuffer("view1.bmp"); //save rendered image to file

	// Render it from a different point of view.
	//Camera camera2(Point3D(4, 2, 1), Vector3D(-4, -2, -6), Vector3D(0, 1, 0), 60.0);
	//Image image2(width, height);
	//raytracer.render(camera2, scene, light_list, image2);
	//image2.flushPixelBuffer("images/view2.bmp");

	// Free memory
	for (size_t i = 0; i < scene.size(); ++i) {
		delete scene[i];
	}

	for (size_t i = 0; i < light_list.size(); ++i) {
		delete light_list[i];
	}

	return 0;
}
*/