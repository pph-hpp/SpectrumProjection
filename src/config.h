#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>


enum MaterialType {
	Bone = 0,
	Water,
	SoftIssue
	//wait add
};

struct FPConfig {
	struct FPPath {
		std::vector<MaterialType>Materials;
		std::unordered_map<MaterialType, std::string>FPInputImageMaterialDir;
		std::unordered_map<MaterialType, std::string>MaterialFilter;
		// 
		std::unordered_map<MaterialType, std::string>FPOutputSgmMaterialDir;

		std::unordered_map<MaterialType, std::string> prefix;
		std::unordered_map<MaterialType, std::pair<std::string, std::string>>replace;
	};
	FPPath FpPath;
	int	imgDim;					// number of rows/cols of reconstructed images
	float pixelSize;				// image pixel size [mm]
	int	sliceCount;				// number of slice in each image file
	bool coneBeam = false;		// whether the fpj is conebeam or fanbeam
	float sliceThickness;			// (for conebeam) slice thickness of each image

	float sid;					// source to isocenter distance [mm]
	float sdd;					// source to detector distance [mm]

	float startAngle = 0;			// angle position of source for the first view [degree]
	int	detEltCount;			// number of detector elements
	int	detZEltCount = 1;			// (for bone beam) number of detector elements in Z direction
	int	views;					// number of views
	float totalScanAngle;			// total scan angle for short scan [degree]

	float detEltSize;				// physical size of detector element [mm]

	float detEltHeight;			// (for cone beam) height of detector element [mm]

	bool converToHU = false;		// whether the image has been conver to HU
	float waterMu = 0.2;				// mu of water
};

struct MEConfig {
	struct MEPath {
		std::vector<MaterialType>Materials;
		std::unordered_map<MaterialType, std::string>MEInputSgmMaterialsDir;
		std::unordered_map<MaterialType, std::string>MEMaterialFilter;
		std::unordered_map<MaterialType, std::string>MECoefficientPath;

		std::vector<std::string>SpectrumEnergys;
		std::unordered_map<std::string, std::string>MEOutputDirs;
		std::unordered_map<std::string, std::string>SpectrumPaths;
		std::unordered_map<std::string, std::string>outputFilesNamePrefix;

		std::pair<std::string, std::string> outputFilesReplace;
		MaterialType materialToReplace;
	};
	MEPath MePath;

	std::unordered_map<MaterialType, float> pMaterials;

	int sgmWidth;	//1
	int sgmHeight;

	int startEnergy = 10;
	int endEnergy = 140;

	float Dose;
	bool insertNoise;	//1

	std::vector<std::string>inputFiles;
	std::vector<std::string>outputFiles;
};
