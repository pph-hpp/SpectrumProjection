#include "mutiEnergyProcess.h"
#include "MEProcess.cuh"
#include <filesystem>
#include <io.h>
#include "../cudaFunction.hpp"


//__global__ void testFunc(TestClass* testclass){
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	if (row == 1 && col == 1){
//		int sum = testclass->x + testclass->y;
//		testclass->x = sum;
//		testclass->y = sum;
//	}
//}


mutiEnergyProcess::mutiEnergyProcess(MEConfig config) {
	this->config = config;
	m_energy_num = config.endEnergy - config.startEnergy + 1;
	Init();
	//wait add
}

mutiEnergyProcess::~mutiEnergyProcess() {
	for (auto material : config.MePath.Materials){
		MemoryAgent::FreeMemory(u_materials[material]);
	}
	MemoryAgent::FreeMemory(spectrum);
	for (auto item : u_materials_cpu){
		if (item.second) {
			delete[] item.second;
		}
	}
}

void mutiEnergyProcess::MallocData() {

}

void mutiEnergyProcess::readMateriasCoefficient(std::vector<MaterialType>materials,
	std::unordered_map<MaterialType, std::string>paths) {

	std::ifstream inputFile;
	float* param_cpu = new float[m_energy_num];
	for (auto material : materials){
		auto filename = paths[material];
		//Opens data file for reading.
		inputFile.open(filename);

		//Creates vector, initially with 0 points.
		/*vector<Point> data(0);*/
		int temp_x;
		double temp_y;
		int i = 0;
		float p = config.pMaterials[material];

		//Read contents of file till EOF.
		while (inputFile.good()) {
			inputFile >> temp_x >> temp_y;
			param_cpu[i] = float(temp_y) * p;
			i += 1;
		}

		if (!inputFile.eof())
			if (inputFile.fail()) std::cout << "Type mismatch during parsing." << std::endl;
			else std::cout << "Unknow problem during parsing." << std::endl;

		//Close data file.
		inputFile.close();
		cudaMalloc((void**)&u_materials[material], sizeof(float) * m_energy_num);
		cudaMemcpy(u_materials[material], param_cpu, sizeof(float) * m_energy_num, cudaMemcpyHostToDevice);

	}
}


void mutiEnergyProcess::readSpectrum(std::unordered_map<std::string, std::string>filenames) {
	std::ifstream inputFile;
	float* spec_cpu = new float[m_energy_num];
	for (auto path : filenames){
		std::string energy = path.first;
		inputFile.open(path.second);

		int temp_x;
		double temp_y;
		int i = 0;

		//Read contents of file till EOF.
		while (inputFile.good()) {
			inputFile >> temp_x >> temp_y;
			spec_cpu[i] = temp_y;
			i += 1;
		}

		if (!inputFile.eof())
			if (inputFile.fail()) std::cout << "Type mismatch during parsing." << std::endl;
			else std::cout << "Unknow problem during parsing." << std::endl;

		inputFile.close();

		cudaMalloc((void**)&spectrums[energy], sizeof(float) * m_energy_num);
		cudaMemcpy(spectrums[energy], spec_cpu, sizeof(float) * m_energy_num, cudaMemcpyHostToDevice);
		allSpectrum[energy] = 0;
		for (int i = 0; i < m_energy_num; i++) {
			allSpectrum[energy] += spec_cpu[i];
		}
	}
}



void mutiEnergyProcess::sgmToSinogram(std::string energy, std::vector<MaterialType>materials,
	std::unordered_map<MaterialType, float*>sgm_materials, float* sinogram_high) {

	dim3 grid((config.sgmWidth + 15) / 16, (config.sgmHeight + 15) / 16);
	dim3 block(16, 16);
	if (materials.size() == 1) {
		auto sgm_m1 = sgm_materials[*materials.begin()];
		auto u1 = u_materials[*materials.begin()];
		phone_count_1material << < grid, block >> > (sgm_m1, sinogram_high, spectrums[energy],
			u1, config.sgmWidth, config.sgmHeight, m_energy_num, config.Dose, allSpectrum[energy], config.insertNoise);
	}
	else if (materials.size() == 2) {
		auto it = materials.begin();
		auto sgm_m1 = sgm_materials[*it];
		auto u1 = u_materials[*it];

		it++;
		auto sgm_m2 = sgm_materials[*it];
		auto u2 = u_materials[*it];

		phone_count_2materials << < grid, block >> > (sgm_m1, sgm_m2, sinogram_high, spectrums[energy],
			u1, u2, config.sgmWidth, config.sgmHeight, m_energy_num, config.Dose, allSpectrum[energy], config.insertNoise);
	}
	else if (materials.size() == 3) {
		auto it = materials.begin();
		auto sgm_m1 = sgm_materials[*it];
		auto u1 = u_materials[*it];

		it++;
		auto sgm_m2 = sgm_materials[*it];
		auto u2 = u_materials[*it];

		it++;
		auto sgm_m3 = sgm_materials[*it];
		auto u3 = u_materials[*it];

		phone_count_3materials << < grid, block >> > (sgm_m1, sgm_m2, sgm_m3, sinogram_high, spectrums[energy],
			u1, u2, u3, config.sgmWidth, config.sgmHeight, m_energy_num, config.Dose, allSpectrum[energy], config.insertNoise);
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



bool mutiEnergyProcess::SpectralPhotonCounting(std::string energy, std::vector<MaterialType>use_list,
	std::unordered_map<MaterialType, float*>sgm_materials, float* sinogram) {
	if (use_list.size() == 0) {
		printf("Material list is empty");
		return false;
	}
	else if (use_list.size() > 3){
		printf("Now supports up to three materials");
		return false;
	}

	this->sgmToSinogram(energy, use_list, sgm_materials, sinogram);
}