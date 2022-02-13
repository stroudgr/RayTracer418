#include "raytracer.cuh"
#include "kernel.cuh"
#include "util.h"

#include "cuda_runtime.h"

//__device__ void Raytracer::daa(Raytracer& d_raytracer, Image& image, int i, int j) { }


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert code %d: %s %s %d\n", code, cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


#define XINDEXPRINT 0
#define YINDEXPRINT 0


__global__ void render_kernel2() {

	//Camera d_camera(Point3D eye, Vector3D view, Vector3D up, double fov);
	//Camera camera1(Point3D(0, 0, 1), Vector3D(0, 0, -1), Vector3D(0, 1, 0), 60.0);
}


__global__ void addToScene(Scene* d_scene, SceneNode **sn) {
	//d_scene->push_back(sn);
	//d_scene->arr = sn;
}



__global__ void render_kernel(Raytracer* d_raytracer, Camera* d_camera, Scene* d_scene, LightList* d_light_list, Image* d_image) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Raytracer d_ray("aa", false);

	
	if (i == XINDEXPRINT && j == YINDEXPRINT)
		printf("Index %d, %d \n", i, j);

	
	double factor = (double(d_image->height) / 2) / tan(d_camera->fov * M_PI / 360.0);

	//double factor = (double(image.height) / 2) / tan(camera.fov * M_PI / 360.0);

	//(d_ray.*Raytracer::renderFunction)(camera, scene, light_list, image, factor, i, j);

	int num = d_image->width * d_image->height;

	int b_index = i * d_image->width + j;


	if (i == XINDEXPRINT && j == YINDEXPRINT)
		printf("Colour before: %d \n", d_image[b_index]);

	d_raytracer->antialiasing(*d_camera, *d_scene, *d_light_list, *d_image, factor, i, j);
	
	if (i == XINDEXPRINT && j == YINDEXPRINT)
		printf("Colour after: %d \n", d_image[b_index]);


}
__host__
void CUDArender(Raytracer& raytracer, Camera& camera, Scene& scene, LightList& light_list, Image& image) {

	// Device constants
	Raytracer* d_raytracer;
	Camera* d_camera = nullptr;
	Scene* d_scene = nullptr;
	LightList* d_light_list = nullptr;

	// Device variables (to modify).
	Image* d_image = nullptr;

	//https://www.reddit.com/r/CUDA/comments/l64vpg/how_to_work_with_classes_in_a_cuda_kernel/gkyljqo/

	// This is for global memory
	// shared keyword in device kernel. copying done in kernel. proabably won't do so (at least at first). If I do it, remember to worry about bank conflicts.



	//************************************************************************************************************************
	// Allocating and copying to device memory.
	//************************************************************************************************************************
	cudaMalloc(&d_image, sizeof(Image));
	cudaMemcpy(d_image, &image, sizeof(Image), cudaMemcpyHostToDevice);

	gpuErrchk(cudaPeekAtLastError());

	cudaMalloc(&d_raytracer, sizeof(Raytracer));
	cudaMemcpy(d_raytracer, &raytracer, sizeof(Raytracer), cudaMemcpyHostToDevice);

	gpuErrchk(cudaPeekAtLastError());


	printf("sizeof(Camera) = %d\n", sizeof(Camera));

	cudaMalloc(&d_camera, sizeof(Camera));
	cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

	gpuErrchk(cudaPeekAtLastError());

	//thrust::device_vector<SceneNode*> d_scene;
	//thrust::copy(scene.begin(), scene.end(), d_scene.begin());

	cudaMalloc(&d_scene, sizeof(Scene));
	cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

	gpuErrchk(cudaPeekAtLastError());

	cudaMalloc(&d_light_list, sizeof(LightList));
	cudaMemcpy(d_light_list, &light_list, sizeof(LightList), cudaMemcpyHostToDevice);


	// ************************************************************************************************************************
	// Copying array fields from host Image to device Image.
	// ************************************************************************************************************************
	int numbytes = image.width * image.height * sizeof(unsigned char);

	unsigned char* d_rbuffer;
	unsigned char* d_gbuffer;
	unsigned char* d_bbuffer;

	gpuErrchk(cudaMalloc(&d_rbuffer, numbytes));
	gpuErrchk(cudaMalloc(&d_gbuffer, numbytes));
	gpuErrchk(cudaMalloc(&d_bbuffer, numbytes));

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(&(d_image->rbuffer), &d_rbuffer, sizeof(unsigned char*), cudaMemcpyHostToDevice)); // change the value of d_image->rbuffer to equal the same value as d_rbuffer ( a device memory location).
	gpuErrchk(cudaMemcpy(d_rbuffer, image.rbuffer, numbytes, cudaMemcpyHostToDevice));						// Then we can copy the cpu data into the gpu location. Could have put (d_image->rbuffer) instead of d_rbuffer since they're the same, 
																								// as is, the two lines of code are interchangeable.

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(&(d_image->gbuffer), &d_gbuffer, sizeof(unsigned char*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_gbuffer, image.gbuffer, numbytes, cudaMemcpyHostToDevice));

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(&(d_image->bbuffer), &d_bbuffer, sizeof(unsigned char*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_bbuffer, image.rbuffer, numbytes, cudaMemcpyHostToDevice));


	gpuErrchk(cudaPeekAtLastError());

	//cudaMalloc(&(d_image->rbuffer), numbytes); // Why this a mistake https://stackoverflow.com/questions/14790999/how-to-pass-a-c-class-with-array-of-pointers-to-cuda
	//cudaMemcpy(d_image->rbuffer, image.rbuffer, numbytes, cudaMemcpyHostToDevice);


	// ************************************************************************************************************************
	// Copying array fields from host Scene to device Scene.
	// ************************************************************************************************************************


	for (int i = 0; i < scene.size(); i++) {

		SceneNode* s = scene[i];
	
		
	}

	SceneNode** nodes;
	gpuErrchk(cudaMalloc(&nodes, sizeof(SceneNode*) * scene.size()));
	
	//gpuErrchk(cudaMemcpy(&(d_scene->arr), &nodes, sizeof(SceneNode**), cudaMemcpyHostToDevice)); // Why this no work?

	
	gpuErrchk(cudaMemcpy(nodes, scene.arr, sizeof(SceneNode*) * scene.size(), cudaMemcpyHostToDevice));
	
	addToScene << <1, 1 >> > (d_scene, nodes);





	
	LightSource** l_sources;
	gpuErrchk(cudaMalloc(&l_sources, sizeof(LightSource*) * light_list.size()));
	

	gpuErrchk(cudaPeekAtLastError());


	//gpuErrchk(cudaMemcpy(&(d_light_list->arr), &l_sources, sizeof(LightSource**), cudaMemcpyHostToDevice));
	
	//gpuErrchk(cudaPeekAtLastError());

	//gpuErrchk(cudaMemcpy(l_sources, light_list.arr, sizeof(LightSource*) * light_list.size(), cudaMemcpyHostToDevice));

	//gpuErrchk(cudaPeekAtLastError());


	dim3 gridsize(8, 8); // something to do with  image.height; image.width;
	dim3 blocksize(8, 8);

	render_kernel << <gridsize, blocksize >> > (d_raytracer, d_camera, d_scene, d_light_list, d_image);

	//render_kernel2 << <gridsize, blocksize >> > ();

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//
	// Copy image back.
	//

	//cudaMemcpy(image, d_image, sizeof(Image), cudaMemcpyDeviceToHost); // not necessary
	//gpuErrchk(cudaMemcpy(image.rbuffer, d_rbuffer, numbytes, cudaMemcpyDeviceToHost));
	//cudaMemcpy(image.rbuffer, d_image->rbuffer, numbytes, cudaMemcpyDeviceToHost); /// This isn't allowed apparently.
	//gpuErrchk(cudaMemcpy(image.gbuffer, d_gbuffer, numbytes, cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(image.bbuffer, d_bbuffer, numbytes, cudaMemcpyDeviceToHost));











	// This isn't good enough!
	// need to copy the buffers.
	// maybe need alternate data structures. Look up what Thrust is.







	//cudaMalloc(&d_camera, sizeof(Camera));
	//cudaMalloc(&d_scene, sizeof(Camera));
	//cudaMalloc(&d_light_list, sizeof(LightList));

	// Lightlist is a vector, cannot use STL on device. 
	// Look into something called "Thrust"
	// https://docs.nvidia.com/cuda/thrust/index.html
	// https://stackoverflow.com/questions/10375680/using-stdvector-in-cuda-device-code


	//cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_light_list, light_list, sizeof(LightList), cudaMemcpyHostToDevice);





}








__device__ void Raytracer::daa(Raytracer& d_raytracer, Image& image, int i, int j) {}