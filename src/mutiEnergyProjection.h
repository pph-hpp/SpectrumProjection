#pragma once
#include "RayCast/lightSource.h"
#include "SpectralProjection/mutiEnergyProcess.h"
#include "config.h"
#include "cudaFunction.hpp"
#include <filesystem>
#include <mutex>

enum ProcessType {
	ForwardPorjedction,
	MEProcess,
	ForwardAndMEProcess
};

class mutiEnergyProjection
{
private:
	std::vector<MaterialType>m_material_list;
	std::unordered_map<MaterialType, float*>img_materials;
	std::unordered_map<MaterialType, float*>sgm_materials;
	float* sinogram = nullptr;			// sinogram for final output
	
	std::unordered_map<MaterialType, float*>img_materials_cpu;
	std::unordered_map<MaterialType, float*>sgm_materials_cpu;
	float* sinogram_cpu = nullptr;			// sinogram for final output

	lightSource* m_light_process;
	mutiEnergyProcess* m_energy_process;

	ProcessType m_process_type = ProcessType::ForwardAndMEProcess;
	Timer timer;
	int m_num_streams = 0;
	std::mutex fileMutex;

	cudaStream_t* m_streams = nullptr;
	void CreateStreams();
	void ClearStreams();

public:
	bool m_use_stream = true;
	cudaDeviceProp prop;
	FPConfig m_FPConfig;
	MEConfig m_MEConfig;

	mutiEnergyProjection(const char* config_path, const cudaDeviceProp &prop);
	~mutiEnergyProjection();

	// Read config file
	void ReadConfigFile(const char* filename);

	void ReadImageFile(const char* filename);

	// Save sinogram data to file (raw format)
	void SaveSinogram(const char* filename, float* data, float* data_cpu);

	std::vector<std::string> GetOutputFileNames(const std::vector<std::string>& inputFileNames,
		std::pair<std::string, std::string>& replace, const std::string& prefix);

	std::vector<std::string> GetInputFileNames(const std::string& dir,
		const std::string& filter);

	void readData(const char* filepath, float *data_cpu, float *data_gpu, int length, cudaStream_t stream);
	void writeData(const char* filepath, float* data_gpu, float* data_cpu, int length, cudaStream_t stream);

	void MallocData();
	void MallocStaticData();

	bool do_forward_projection(int img_offset, int sgm_offset, cudaStream_t stream);
	bool do_forward_projection(float *img, float *sgm, cudaStream_t stream);
	bool do_energy_process(int e_idx, int offset, cudaStream_t stream);
	bool process();
private:
};

