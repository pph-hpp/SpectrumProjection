#include "mutiEnergyProcess.h"
#include "MEProcess.cuh"
#include <filesystem>
#include <io.h>
#include "../cudaFunction.hpp"

mutiEnergyProcess::mutiEnergyProcess(MEConfig config) {
	this->config = config;
	m_energy_num = config.endEnergy - config.startEnergy + 1;
	Init();
	//wait add
}

mutiEnergyProcess::~mutiEnergyProcess() {
	MemoryAgent::FreeMemory(coefficient);
	for (auto material : config.MePath.Materials){
		coef_materials[material] = nullptr;
	}
	MemoryAgent::FreeMemory(spectrum);
	for (auto energy : config.MePath.SpectrumEnergys) {
		spectrums[energy] = nullptr;
	}
}

void mutiEnergyProcess::MallocData() {

}

void mutiEnergyProcess::readMateriasCoefficient(std::vector<MaterialType>materials,
	std::unordered_map<MaterialType, std::string>paths) {

	std::ifstream inputFile;
	float* param_cpu = new float[m_energy_num];
	cudaMalloc((void**)&coefficient, sizeof(float) * m_energy_num * materials.size());
	int e_idx = 0;
	for (int i = 0; i < materials.size(); i++) {
		auto material = materials[i];
		auto filename = paths[material];
		//Opens data file for reading.
		inputFile.open(filename);

		//Creates vector, initially with 0 points.
		/*vector<Point> data(0);*/
		int temp_x;
		double temp_y;
		int e = 0;
		float p = config.pMaterials[material];

		//Read contents of file till EOF.
		while (inputFile.good()) {
			inputFile >> temp_x >> temp_y;
			param_cpu[e] = float(temp_y) * p;
			e += 1;
		}

		if (!inputFile.eof())
			if (inputFile.fail()) std::cout << "Type mismatch during parsing." << std::endl;
			else std::cout << "Unknow problem during parsing." << std::endl;

		coef_materials[material] = coefficient + e_idx * m_energy_num;
		cudaMemcpy(coef_materials[material], param_cpu, sizeof(float) * m_energy_num, cudaMemcpyHostToDevice);
		e_idx++;
		
	}
	//Close data file.
	try {
		inputFile.close();
	}
	catch (const std::ios_base::failure& e) {
		std::cerr << "File operation failed: " << e.what() << std::endl;
	}

	// 绑定数据到纹理内存
	//bind_coefficient_texture(param_cpu, m_energy_num, materials.size());
}


void mutiEnergyProcess::readSpectrum(std::unordered_map<std::string, std::string>filenames) {
	std::ifstream inputFile;
	float* spec_cpu = new float[m_energy_num];
	int e_idx = 0;
	cudaMalloc((void**)&spectrum, sizeof(float) * m_energy_num * config.MePath.SpectrumEnergys.size());
	for (int i = 0; i < config.MePath.SpectrumEnergys.size(); i++) {
		std::string energy = config.MePath.SpectrumEnergys[i];
		inputFile.open(filenames[energy]);

		int temp_x;
		double temp_y;
		int e = 0;

		//Read contents of file till EOF.
		while (inputFile.good()) {
			inputFile >> temp_x >> temp_y;
			spec_cpu[e] = temp_y;
			e += 1;
		}

		if (!inputFile.eof())
			if (inputFile.fail()) std::cout << "Type mismatch during parsing." << std::endl;
			else std::cout << "Unknow problem during parsing." << std::endl;

		spectrums[energy] = spectrum + e_idx * m_energy_num;
		cudaMemcpy(spectrums[energy], spec_cpu, sizeof(float) * m_energy_num, cudaMemcpyHostToDevice);

		allSpectrum[energy] = 0;
		for (int i = 0; i < m_energy_num; i++) {
			allSpectrum[energy] += spec_cpu[i];
		}
		e_idx++;
	}

	try {
		inputFile.close();
	}
	catch (const std::ios_base::failure& e) {
		std::cerr << "File operation failed: " << e.what() << std::endl;
	}

	//bind_spectrum_texture(spec_cpu, m_energy_num, config.MePath.SpectrumEnergys.size());
}



void mutiEnergyProcess::sgmToSinogram(int e_idx, std::vector<MaterialType>materials,
	std::unordered_map<MaterialType, float*>sgm_materials, float* sinogram_high) {

	

}



void mutiEnergyProcess::InitSgm() {
	SpCpu = new float[config.sgmWidth * config.sgmHeight];
}

void mutiEnergyProcess::setFilter(std::string filBone, std::string filWater) {

}

void mutiEnergyProcess::test(){
	dim3 grid(16, 16);
	dim3 block(16, 16);
	cudaMalloc((void**)&testclass, sizeof(TestClass));
	cudaMemcpy(testclass, testclassCpu, sizeof(TestClass), cudaMemcpyHostToDevice);
	//testFunc << < grid, block >> > (testclass);
	cudaMemcpy(testclassCpu, testclass, sizeof(TestClass), cudaMemcpyDeviceToHost);
}


void mutiEnergyProcess::Init() {
	//std::cout << (config.MePath.MEInputSgmMaterialsDir.begin()->second) << std::endl;
	this->readMateriasCoefficient(config.MePath.Materials, config.MePath.MECoefficientPath);
	this->readSpectrum(config.MePath.SpectrumPaths);
}



bool mutiEnergyProcess::SpectralPhotonCounting(int e_idx, std::vector<MaterialType>materials,
	std::unordered_map<MaterialType, float*>sgm_materials, float* sinogram, int offset, cudaStream_t stream) {
	if (materials.size() == 0) {
		printf("Material list is empty");
		return false;
	}
	else if (materials.size() > 3){
		printf("Now supports up to three materials");
		return false;
	}

	if (materials.size() == 1) {
		auto sgm_m1 = sgm_materials[*materials.begin()] + offset;

		std::string energy = config.MePath.SpectrumEnergys[e_idx];
		phone_count_agent({ sgm_m1 }, sinogram + offset, spectrums[energy], coefficient, energy, config.sgmWidth,
			config.sgmHeight, m_energy_num, config.Dose, allSpectrum[energy], config.insertNoise, stream);
	}
	else if (materials.size() == 2) {
		auto it = materials.begin();
		auto sgm_m1 = sgm_materials[*it] + offset;

		it++;
		auto sgm_m2 = sgm_materials[*it] + offset;

		std::string energy = config.MePath.SpectrumEnergys[e_idx];
		phone_count_agent({ sgm_m1, sgm_m2 }, sinogram + offset, spectrums[energy], coefficient, energy,
			config.sgmWidth, config.sgmHeight, m_energy_num, config.Dose, allSpectrum[energy], config.insertNoise, stream);

	}
	else if (materials.size() == 3) {
		auto it = materials.begin();
		auto sgm_m1 = sgm_materials[*it] + offset;
		it++;
		auto sgm_m2 = sgm_materials[*it] + offset;
		it++;
		auto sgm_m3 = sgm_materials[*it] + offset;

		std::string energy = config.MePath.SpectrumEnergys[e_idx];
		phone_count_agent({ sgm_m1, sgm_m2, sgm_m3 }, sinogram + offset, spectrums[energy], coefficient, energy,
			config.sgmWidth, config.sgmHeight, m_energy_num, config.Dose, allSpectrum[energy], config.insertNoise, stream);
	}
	cudaDeviceSynchronize();

	/*if (insertNoise)
	{
		RadonNoiseGenerator pph_noise(0);
		pph_noise.add_noise(sinogram, Dose, false, sgmWidth, sgmHeight);
	}*/
	//cudaMemcpy(sinogramHighCpu, sinogramHigh, sizeof(float) * config.sgmWidth * config.sgmHeight, cudaMemcpyDeviceToHost);

	// if (insertNoise)
	// {
	// 	std::default_random_engine generator((unsigned int)time(NULL));

	// 	for (int i = 0; i < sgmWidth * sgmHeight; i++)
	// 	{
	// 		std::poisson_distribution<int> poissonDistribution(sinogramCpu[i]); // 参数为泊松分布的 lambda
	// 		int randomValue = poissonDistribution(generator);	// 生成泊松随机数
	// 		if (randomValue <= 0)
	// 		{
	// 			randomValue = 1;
	// 		}
	// 		sinogramCpu[i] = log(Dose / randomValue);
	// 	}
	// }

}