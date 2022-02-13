#pragma once

#if 0

#include "util.h"
#include "light_source.h"
#include "scene_object.h"
#include "kernel.cuh"


typedef ContainerVec<Material> MaterialList;
typedef ContainerVec<Material> MaterialList;



class World {

public:

	World(bool isCuda = true) {

		this->isCuda = isCuda;




		if (isCuda) {

		}
		else {

		}

	}

	void render() {

		if (isCuda) {
			//CUDArender(*this, raytracer, camera, scene, light_list, image);
		}

		else {
		
		
		}

	}



	void addMaterial(Material m);
	void addMaterial(Material m);




	Raytracer raytracer;
	Camera camera; 
	Scene scene; LightList& light_list; 
	Image image;



private:
	bool isCuda;


};

#endif