#include "mutiEnergyProjection.h"
#include "SpectralProjection/MEProcess.cuh"
#include "cudaFunction.hpp"

mutiEnergyProjection::mutiEnergyProjection(const char* config_path)
{
	ReadConfigFile(config_path);
	MallocData();
	m_light_process = new lightSource(m_FPConfig);
	std::cout << "lightSource has been created" << std::endl;
	m_energy_process = new mutiEnergyProcess(m_MEConfig);
	std::cout << "mutiEnergyProcess has been created" << std::endl;
	// malloc image and sinogram
}

mutiEnergyProjection::~mutiEnergyProjection()
{
	if (m_process_type == ProcessType::ForwardPorjedction) {
		for (auto material : m_material_list) {
			MemoryAgent::FreeMemoryCpu(img_materials_cpu[material]);
			MemoryAgent::FreeMemoryCpu(sgm_materials_cpu[material]);

			MemoryAgent::FreeMemory(img_materials[material]);
			MemoryAgent::FreeMemory(sgm_materials[material]);
		}
	}
	else if (m_process_type == ProcessType::MEProcess) {
		for (auto material : m_material_list) {
			MemoryAgent::FreeMemoryCpu(sgm_materials_cpu[material]);
			MemoryAgent::FreeMemory(sgm_materials[material]);
		}
		MemoryAgent::FreeMemory(sinogram);
		MemoryAgent::FreeMemoryCpu(sinogram_cpu);
	}
	else {
		for (auto material : m_material_list) {
			MemoryAgent::FreeMemoryCpu(img_materials_cpu[material]);
			MemoryAgent::FreeMemoryCpu(sgm_materials_cpu[material]);

			MemoryAgent::FreeMemory(img_materials[material]);
			MemoryAgent::FreeMemory(sgm_materials[material]);
		}
		MemoryAgent::FreeMemoryCpu(sinogram_cpu);

		MemoryAgent::FreeMemory(sinogram);
	}

	if (m_light_process)
		delete m_light_process;
	m_light_process = nullptr;

	if (m_energy_process)
		delete m_energy_process;
	m_energy_process = nullptr;
}

void mutiEnergyProjection::ReadImageFile(const char* filename) {

}

// Save sinogram data to file (raw format)
void mutiEnergyProjection::SaveSinogram(const char* filename, float *data, float *data_cpu) {
#pragma warning (disable : 4996)
	FILE* fp = NULL;
	fp = fopen(filename, "wb");

	if (fp == NULL){
		fprintf(stderr, "Cannot save to file %s!\n", filename);
		exit(4);
	}
	cudaMemcpy(data_cpu, data, sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight, cudaMemcpyDeviceToHost);
	fwrite(data_cpu, sizeof(float), m_MEConfig.sgmWidth * m_MEConfig.sgmHeight, fp);
	
	fclose(fp);
}

void mutiEnergyProjection::ReadConfigFile(const char* filename, ConfigType type) {
	namespace fs = std::filesystem;
	namespace js = rapidjson;

	// load the config file
	std::ifstream ifs(filename);
	if (!ifs)
	{
		printf("Cannot open config file '%s'!\n", filename);
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);

	if (m_process_type == ProcessType::ForwardPorjedction || m_process_type == ProcessType::ForwardAndMEProcess) {
		const js::Value& projection_Js = doc["ForwardProjectionConfig"];

		m_material_list.clear();
		m_FPConfig.FpPath.Materials.clear();
		const js::Value& material_type_Js = projection_Js["Materials"];
		std::vector<std::string>str_materials;
		const js::Value& materials_path_Js = projection_Js["MaterialPath"];
		for (js::SizeType i = 0; i < material_type_Js.Size(); i++) {
			std::string type_str = material_type_Js[i].GetString();
			const js::Value& material_Js = materials_path_Js[type_str.c_str()];
			MaterialType type;
			if (type_str == "Bone"){
				type = MaterialType::Bone;
			}
			else if (type_str == "Water"){
				type = MaterialType::Water;
			}
			else if (type_str == "SoftIssue"){
				type = MaterialType::SoftIssue;
			}
			m_material_list.push_back(type);
			m_FPConfig.FpPath.Materials.push_back(type);
			m_FPConfig.FpPath.FPInputImageMaterialDir[type] = material_Js["InputDir"].GetString();
			m_FPConfig.FpPath.FPOutputSgmMaterialDir[type] = material_Js["OutputDir"].GetString();
			m_FPConfig.FpPath.MaterialFilter[type] = material_Js["InputFilesFilter"].GetString();
			m_FPConfig.FpPath.prefix[type] = material_Js["OutputFilePrefix"].GetString();
			const js::Value& replace_Js = material_Js["OutputFileReplace"];
			m_FPConfig.FpPath.replace[type] = { material_Js["OutputFileReplace"][0].GetString(),
				material_Js["OutputFileReplace"][1].GetString() };

			// check inputDir and outputDir
			fs::path inDir(m_FPConfig.FpPath.FPInputImageMaterialDir[type]);
			fs::path outDir(m_FPConfig.FpPath.FPOutputSgmMaterialDir[type]);
			if (!fs::exists(inDir)){
				fprintf(stderr, "Input directory %s does not exist!\n", m_FPConfig.FpPath.FPInputImageMaterialDir[type].c_str());
				exit(1);
			}
			if (!fs::exists(outDir)){
				fprintf(stderr, "Output directory %s does not exist!\n", m_FPConfig.FpPath.FPOutputSgmMaterialDir[type].c_str());
				exit(1);
			}
		}

		m_FPConfig.imgDim = projection_Js["ImageDimension"].GetInt();

		// get pixel size
		if (projection_Js.HasMember("PixelSize")){
			m_FPConfig.pixelSize = projection_Js["PixelSize"].GetFloat();
		}
		else if (projection_Js.HasMember("ImageSize")){
			m_FPConfig.pixelSize = projection_Js["ImageSize"].GetFloat() / m_FPConfig.imgDim;
		}
		else{
			fprintf(stderr, "Did not find PixelSize or ImageSize! Please check your config file: %s.\n", filename);
			exit(1);
		}

		m_FPConfig.sliceCount = projection_Js["SliceCount"].GetInt();


		//wait add, if add, need change readData and writeData function
		/*if (projection_Js.HasMember("ConeBeam")){
			m_FPConfig.coneBeam = projection_Js["ConeBeam"].GetBool();
		}*/


		m_FPConfig.sid = projection_Js["SourceIsocenterDistance"].GetFloat();

		m_FPConfig.sdd = projection_Js["SourceDetectorDistance"].GetFloat();

		if (projection_Js.HasMember("StartAngle")){
			m_FPConfig.startAngle = projection_Js["StartAngle"].GetFloat();
		}

		if (projection_Js.HasMember("TotalScanAngle")){
			m_FPConfig.totalScanAngle = projection_Js["TotalScanAngle"].GetFloat();
		}
		else{
			m_FPConfig.totalScanAngle = 360.0f;
		}

		if (abs(m_FPConfig.totalScanAngle - 360.0f) < 0.001f){
			printf("--FULL scan--\n");
		}
		else{
			printf("--SHORT scan (%.1f degrees)--\n", m_FPConfig.totalScanAngle);
		}

		m_FPConfig.detEltCount = projection_Js["DetectorElementCount"].GetInt();
		m_FPConfig.views = projection_Js["Views"].GetInt();

		m_FPConfig.detEltSize = projection_Js["DetectorElementSize"].GetFloat();


		if (m_FPConfig.coneBeam == true)
		{
			printf("--CONE beam--\n");
			if (projection_Js.HasMember("ImageSliceThickness")){
				m_FPConfig.sliceThickness = projection_Js["ImageSliceThickness"].GetFloat();
			}
			if (projection_Js.HasMember("DetectorZElementCount")){
				m_FPConfig.detZEltCount = projection_Js["DetectorZElementCount"].GetInt();
			}
			if (projection_Js.HasMember("DetectorElementHeight")){
				m_FPConfig.detEltHeight = projection_Js["DetectorElementHeight"].GetFloat();
			}
		}
		else{
			m_FPConfig.detZEltCount = m_FPConfig.sliceCount;
		}

		if (projection_Js.HasMember("WaterMu")){
			printf("--Images are in HU values--\n");
			m_FPConfig.converToHU = true;
			m_FPConfig.waterMu = projection_Js["WaterMu"].GetFloat();
		}
		else{
			m_FPConfig.converToHU = false;
		}
	}
	if (m_process_type == ProcessType::MEProcess || m_process_type == ProcessType::ForwardAndMEProcess){
		const js::Value& MEProcess_Js = doc["MutiEnergyProcessConfig"];

		m_material_list.clear();
		m_MEConfig.MePath.Materials.clear();
		const js::Value& material_type_Js = MEProcess_Js["Materials"];
		std::vector<std::string>str_materials;
		const js::Value& materials_path_Js = MEProcess_Js["MaterialPath"];
		for (js::SizeType i = 0; i < material_type_Js.Size(); i++) {
			std::string type_str = material_type_Js[i].GetString();
			const js::Value& material_Js = materials_path_Js[type_str.c_str()];
			MaterialType type;
			if (type_str == "Bone") {
				type = MaterialType::Bone;
			}
			else if (type_str == "Water") {
				type = MaterialType::Water;
			}
			else if (type_str == "SoftIssue") {
				type = MaterialType::SoftIssue;
			}
			m_material_list.push_back(type);
			m_MEConfig.MePath.Materials.push_back(type);
			m_MEConfig.MePath.MEInputSgmMaterialsDir[type] = material_Js["InputDir"].GetString();
			m_MEConfig.MePath.MEMaterialFilter[type] = material_Js["InputFilesFilter"].GetString();
			m_MEConfig.MePath.MECoefficientPath[type] = material_Js["CoefficientPath"].GetString();
			m_MEConfig.pMaterials[type] = material_Js["Density"].GetFloat();
			if (material_Js.HasMember("OutputFileReplace")){
				m_MEConfig.MePath.materialToReplace = type;
				m_MEConfig.MePath.outputFilesReplace.first = material_Js["OutputFileReplace"][0].GetString();
				m_MEConfig.MePath.outputFilesReplace.second = material_Js["OutputFileReplace"][1].GetString();
			}

			// check inputDir and outputDir
			fs::path inDir(m_MEConfig.MePath.MEInputSgmMaterialsDir[type]);
			if (!fs::exists(inDir)) {
				fprintf(stderr, "Input directory %s does not exist!\n", m_FPConfig.FpPath.FPInputImageMaterialDir[type].c_str());
				exit(1);
			}
		}

		m_MEConfig.sgmWidth = MEProcess_Js["sgmWidth"].GetInt();
		m_MEConfig.sgmHeight = MEProcess_Js["sgmHeight"].GetInt();

		if (MEProcess_Js.HasMember("SpectrumPath")) {
			m_MEConfig.MePath.SpectrumEnergys.clear();
			const js::Value& spectrums_Js = MEProcess_Js["SpectrumPath"];
			for (js::SizeType i = 0; i < spectrums_Js["energy"].Size(); i++) {
				m_MEConfig.MePath.SpectrumEnergys.push_back(spectrums_Js["energy"][i].GetString());
			}
			for (auto energy : m_MEConfig.MePath.SpectrumEnergys){
				const js::Value& spectrumInfo_Js = spectrums_Js[energy.c_str()];
				m_MEConfig.MePath.SpectrumPaths[energy] = spectrumInfo_Js["path"].GetString();
				m_MEConfig.MePath.outputFilesNamePrefix[energy] = spectrumInfo_Js["FileNamesPrefix"].GetString();
				m_MEConfig.MePath.MEOutputDirs[energy] = spectrumInfo_Js["outputHighDir"].GetString();
			}
		}
		else {
			m_MEConfig.MePath.SpectrumEnergys = { "80kvp" };
			m_MEConfig.MePath.SpectrumPaths["80kvp"] = "resource/80kvp.txt";
			m_MEConfig.MePath.outputFilesNamePrefix["80kvp"] = "sgm_80kvp";
			m_MEConfig.MePath.MEOutputDirs["80kvp"] = "resource/sinogram/sgm_80kvp";
		}

		m_MEConfig.insertNoise = MEProcess_Js["insertNoise"].GetBool();
		if (MEProcess_Js.HasMember("Dose")){
			m_MEConfig.Dose = MEProcess_Js["Dose"].GetFloat();
		}
		else {
			m_MEConfig.Dose = 1e5;
		}
		
	}
	std::cout << "参数读取完成" << std::endl;
}


// acquire the list of output file names, replace substring in input file names, add prefix and postfix
std::vector<std::string> mutiEnergyProjection::GetOutputFileNames(const std::vector<std::string>& inputFileNames, std::pair<std::string, std::string>& replace, const std::string& prefix)
{
	std::vector<std::string> outputFiles;

	for (size_t fileIdx = 0; fileIdx < inputFileNames.size(); fileIdx++)
	{
		std::string outputFile = inputFileNames[fileIdx];
		auto pos = outputFile.find(replace.first);
		if (pos == std::string::npos){
			fprintf(stderr, "Did not find substring \"%s\" to be replaced!\n", replace.first.c_str());
			exit(2);
		}
		outputFile.replace(pos, replace.first.size(), replace.second);
		outputFiles.push_back(prefix + outputFile);
	}
	return outputFiles;
}


// acquire the list of file names in the dir that matches filter
std::vector<std::string> mutiEnergyProjection::GetInputFileNames(const std::string& dir, const std::string& filter)
{
	namespace fs = std::filesystem;

	std::vector<std::string> filenames;

	if (!fs::is_directory(dir))
	{
		return filenames;
	}

	fs::directory_iterator end_iter;
	std::regex e(filter);

	for (auto&& fe : fs::directory_iterator(dir))
	{
		std::string file = fe.path().filename().string();

		if (std::regex_match(file, e))
		{
			filenames.push_back(file);
		}
	}
	return filenames;
}


void mutiEnergyProjection::readData(const char* filepath, float* data_cpu, float* data_gpu, int length) {
	FILE* fp = fopen(filepath, "rb");
	if (fp == NULL){
		fprintf(stderr, "Cannot open file %s!\n", filepath);
		exit(3);
	}
	fread(data_cpu, sizeof(float), length, fp);
	cudaMemcpy(data_gpu, data_cpu, sizeof(float) * length, cudaMemcpyHostToDevice);
	fclose(fp);
}

void mutiEnergyProjection::writeData(const char* filepath, float* data_gpu, float* data_cpu) {
#pragma warning (disable : 4996)
	FILE* fp = NULL;
	fp = fopen(filepath, "wb");

	if (fp == NULL) {
		fprintf(stderr, "Cannot save to file %s!\n", filepath);
		exit(4);
	}
	//cudaMemcpy(data_cpu, data, sizeof(float) * config.sgmWidth * config.sgmHeight, cudaMemcpyDeviceToHost);
	if (m_process_type == ProcessType::ForwardPorjedction) {
		cudaMemcpy(data_cpu, data_gpu, sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views, cudaMemcpyDeviceToHost);
		fwrite(data_cpu, sizeof(float), m_FPConfig.detEltCount * m_FPConfig.views, fp);
	}
	else{
		cudaMemcpy(data_cpu, data_gpu, sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight, cudaMemcpyDeviceToHost);
		fwrite(data_cpu, sizeof(float), m_MEConfig.sgmWidth * m_MEConfig.sgmHeight, fp);
	}
	fclose(fp);
}


void mutiEnergyProjection::MallocData() {
	if (m_process_type == ProcessType::ForwardPorjedction){
		for (auto material : m_material_list){
			//img_materials_cpu[material] = new float[m_FPConfig.imgDim * m_FPConfig.imgDim];
			img_materials_cpu[material] = new float[512 * 512];
			sgm_materials_cpu[material] = new float[m_FPConfig.detEltCount * m_FPConfig.views];

			cudaMalloc((void**)&img_materials[material], sizeof(float) * m_FPConfig.imgDim * m_FPConfig.imgDim);
			cudaMalloc((void**)&sgm_materials[material], sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views);
		}
	}
	else if (m_process_type == ProcessType::MEProcess){
		for (auto material : m_material_list) {
			sgm_materials_cpu[material] = new float[m_MEConfig.sgmWidth * m_MEConfig.sgmHeight];
			cudaMalloc((void**)&sgm_materials[material], sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight);
		}
		cudaMalloc((void**)&sinogram, sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight);
		sinogram_cpu = new float[m_MEConfig.sgmWidth * m_MEConfig.sgmHeight];
	}
	else{
		for (auto material : m_material_list) {
			img_materials_cpu[material] = new float[m_FPConfig.imgDim * m_FPConfig.imgDim];
			sgm_materials_cpu[material] = new float[m_FPConfig.detEltCount * m_FPConfig.views];

			cudaMalloc((void**)&img_materials[material], sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views);
			cudaMalloc((void**)&sgm_materials[material], sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views);
		}
		cudaMalloc((void**)&sinogram, sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight);
		sinogram_cpu = new float[m_MEConfig.sgmWidth * m_MEConfig.sgmHeight];
	}

}


bool mutiEnergyProjection::do_forward_projection() {
	if (m_light_process == nullptr){
		printf("Forward Projection Error, Create lightSource first.\n");
		return false;
	}
	for (auto material : m_material_list){
		m_light_process->ForwardProjectionBilinear(img_materials[material], sgm_materials[material]);
	}
}

bool mutiEnergyProjection::do_forward_projection(float *img, float *sgm) {
	if (m_light_process == nullptr) {
		printf("Forward Projection Error, Create lightSource first.\n");
		return false;
	}
	m_light_process->ForwardProjectionBilinear(img, sgm);
}

bool mutiEnergyProjection::do_energy_process(std::string energy) {
	m_material_list, sgm_materials;
	if (m_energy_process == nullptr) {
		printf("Muti-Energy Process Error, Create mutiEnergyProcess first.\n");
		return false;
	}
	/*for (auto material : m_material_list){
		if (!sgm_materials.count(material) || sgm_materials[material] == nullptr) {
			printf("Memory Error, Malloc memory first.\n");
			return false;
		}
	}*/
	//read
	m_energy_process->SpectralPhotonCounting(energy, m_material_list, sgm_materials, sinogram);
	//write
}

bool mutiEnergyProjection::process() {
	if (m_process_type == ProcessType::ForwardPorjedction){
		std::unordered_map<MaterialType, std::vector<std::string>>image_material_names;
		std::unordered_map<MaterialType, std::vector<std::string>>sgm_material_names;
		auto PathData = m_FPConfig.FpPath;
		for (auto item : PathData.Materials) {
			auto image_dir = PathData.FPInputImageMaterialDir[item];
			auto sgm_dir = PathData.FPOutputSgmMaterialDir[item];
			auto filter = PathData.MaterialFilter[item];

			auto input_img_names = GetInputFileNames(image_dir, filter);
			image_material_names[item] = input_img_names;
			auto output_sgm_names = GetOutputFileNames(input_img_names, PathData.replace[item], PathData.prefix[item]);
			sgm_material_names[item] = output_sgm_names;
		}
		for (auto item : PathData.Materials) {
			auto input_dir = PathData.FPInputImageMaterialDir[item];
			auto output_dir = PathData.FPOutputSgmMaterialDir[item];
			for (int i = 0; i < image_material_names[item].size(); i++) {
				readData((input_dir + '/' + image_material_names[item][i]).c_str(), img_materials_cpu[item],
					img_materials[item], m_FPConfig.imgDim * m_FPConfig.imgDim);
				do_forward_projection(img_materials[item], sgm_materials[item]);
				writeData((output_dir + '/' + sgm_material_names[item][i]).c_str(), sgm_materials[item], sgm_materials_cpu[item]);
			}
		}
	}
	else if (m_process_type == ProcessType::MEProcess){
		std::unordered_map<MaterialType, std::vector<std::string>>material_names;
		for (auto item : m_material_list){
			material_names[item] = GetInputFileNames(m_MEConfig.MePath.MEInputSgmMaterialsDir[item].c_str(),
				m_MEConfig.MePath.MEMaterialFilter[item].c_str());
		}
		for (auto energy : m_MEConfig.MePath.SpectrumEnergys){
			printf("Start process energy: %s\n", energy);
			std::vector<std::string>outputFileNames = GetOutputFileNames(material_names[m_MEConfig.MePath.materialToReplace],
				m_MEConfig.MePath.outputFilesReplace, m_MEConfig.MePath.outputFilesNamePrefix[energy]);

			auto dirs = m_MEConfig.MePath.MEInputSgmMaterialsDir;
			for (size_t i = 0; i < outputFileNames.size(); i++) {
				for (auto item : m_material_list) {
					readData((dirs[item] + '/' + material_names[item][i]).c_str(), sgm_materials_cpu[item], sgm_materials[item], m_MEConfig.sgmWidth * m_MEConfig.sgmHeight);
				}
				do_energy_process(energy);
				writeData((m_MEConfig.MePath.MEOutputDirs[energy] + '/' + outputFileNames[i]).c_str(), sinogram, sinogram_cpu);
			}
		}
	}
	else if (m_process_type == ProcessType::ForwardAndMEProcess)
	{
		std::unordered_map<MaterialType, std::vector<std::string>>material_names;

		for (auto item : m_material_list) {
			material_names[item] = GetInputFileNames(m_FPConfig.FpPath.FPInputImageMaterialDir[item].c_str(),
				m_FPConfig.FpPath.MaterialFilter[item].c_str());
		}

		for (auto energy : m_MEConfig.MePath.SpectrumEnergys) {
			printf("\nStart process energy: %s\n", energy);
			std::vector<std::string>outputFileNames = GetOutputFileNames(material_names[m_MEConfig.MePath.materialToReplace],
				m_MEConfig.MePath.outputFilesReplace, m_MEConfig.MePath.outputFilesNamePrefix[energy]);

			auto dirs = m_FPConfig.FpPath.FPInputImageMaterialDir;
			for (size_t i = 0; i < outputFileNames.size(); i++) {
				for (auto item : m_material_list) {
					readData((dirs[item] + '/' + material_names[item][i]).c_str(), img_materials_cpu[item], img_materials[item], m_FPConfig.imgDim * m_FPConfig.imgDim);
				}
				do_forward_projection();
				do_energy_process(energy);
				writeData((m_MEConfig.MePath.MEOutputDirs[energy] + '/' + outputFileNames[i]).c_str(), sinogram, sinogram_cpu);
			}
		}
	}
	return true;
}