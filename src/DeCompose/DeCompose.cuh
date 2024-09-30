#pragma once
#include <string>
#include <vector>
#include <stdio.h>
#include <cmath>
#include<random>
#include<iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>


class DeCompose
{
public:

	std::string sgmLowDir;
	std::string sgmHighDir;
	std::string sgmBoneDir;
	std::string sgmWaterDir;

	std::string waterPath;
	std::string bonePath;
	std::string highEnergyPath;
	std::string lowEnergyPath;
	std::string lowFilter;
	std::string highFilter;

	std::vector<std::string>replaceBone;
	std::vector<std::string>replaceWater;

	int sgmWidth;
	int sgmHeight;
	int energyNum;

	DeCompose(const char* filename, int o) {
		Init(filename, o);
	}
	DeCompose(int w, int h, int o);
	~DeCompose();
	void material_decompose();
	void readSgm(const char* fileNameLow, const char* fileNameHigh);
	void saveSgm(const char* fileNameBone, const char* fileNameWater);
	void readParam(std::vector<float>&paramBone, std::vector<float>&paramWater);
	void readParam(std::vector<std::vector<float>>& params);
	void readConfigFile(const char* filename);
	void readEnergyData(std::string filename, float* data, float p);
	/*void Init(const mutiEnergyProcess* proc, int o);*/
	void Init(const char* filename, int o);
	void use();
	std::vector<std::string> GetInputFileNames(const std::string& dir, const std::string& filter);
	std::vector<std::string> GetOutputFileNames(const std::vector<std::string>& inputFileNames, const std::vector<std::string>& replace, const std::string& prefix);

private:
	float* sgmLowEnergy;
	float* sgmHighEnergy;
	float* sgmLowEnergyCpu;
	float* sgmHighEnergyCpu;
	int width;
	int height;
	int paramNum;	//拟合参数数量
	int order;

	
	float* paramFitBone;	//拟合参数
	float* paramFitWater;
	
	float* sgmBone;	//正弦图
	float* sgmWater;
	float* sgmBoneCpu;
	float* sgmWaterCpu;

};

