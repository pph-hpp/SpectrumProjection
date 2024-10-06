#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <list>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <time.h>
#include<random>
#include<iostream>
#include <fstream>
#include <regex>
#include "noise.h"
#include "test.h"
#include "../config.h"

class mutiEnergyProcess {
private:
	std::unordered_map<MaterialType, float> p_materials;

	/*static float uBone;
	static float uWater;*/
	struct MEConfig config;
	int sgm_num = 0;
	int m_energy_num;
	std::unordered_map<std::string, float> allSpectrum;

	float* coefficient = nullptr;
	std::unordered_map<MaterialType, float*>coef_materials;
	
	float* spectrum = nullptr;
	std::unordered_map<std::string, float*>spectrums;

	float* SpCpu = nullptr;
	TestClass* testclassCpu;
	TestClass* testclass;

public:

	/*mutiEnergyProcess(int num, int width, int height, int enery, bool ifNoise, std::string waterPath, std::string bonePath,
		std::string highEnergyPath, std::string lowEnergyPath) :sgm_num(num), sgmWidth(width),
		sgmHeight(height), energyNum(enery), insertNoise(ifNoise) {

		this->Dose = 2e5;
		this->boneFilter = "sgmb_*.raw";
		this->waterFilter = "sgmw_*.raw";
		this->prefix = "";
		this->replace = { "sgmb_", "sgm_" };

		this->inputBoneDir = "./sgm/bone/";
		this->inputWaterDir = "./sgm/water/";
		this->outputHighDir = "./sinogram/120kvp/";
		this->outputLowDir = "./sinogram/80kvp/";

		Init(waterPath, bonePath, highEnergyPath, lowEnergyPath);
	}*/

	mutiEnergyProcess(const char* configFilePath) {
		Init();
	}

	mutiEnergyProcess(MEConfig config);

	~mutiEnergyProcess();
	void readMateriasCoefficient(std::vector<MaterialType>materials, std::unordered_map<MaterialType, std::string>paths);
	void readSpectrum(std::unordered_map<std::string, std::string>filenames);
	void sgmToSinogram(int e_idx, std::vector<MaterialType>materials,
		std::unordered_map<MaterialType, float*>sgm_materials, float* sinogram_high);

	void InitSgm();
	void setFilter(std::string filBone, std::string filWater);
	void test();
	void Init();
	void MallocData();
	bool SpectralPhotonCounting(int e_idx, std::vector<MaterialType>use_list,
		std::unordered_map<MaterialType, float*>sgm_materials, float* sinogram, int offset, cudaStream_t stream);

};

