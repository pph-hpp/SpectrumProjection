#include "mutiEnergyProjection.h"
#include "SpectralProjection/MEProcess.cuh"
#include "omp.h"

#include <iostream>
#include <fstream>


mutiEnergyProjection::mutiEnergyProjection(const char* config_path, const cudaDeviceProp &prop)
{
	ReadConfigFile(config_path);
	this->prop = prop;
	if (this->m_use_stream) {
		CreateStreams();
		MallocStaticData();
	}
	else{
		MallocData();
	}
	
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

	if (m_use_stream){
		ClearStreams();
	}
}

void mutiEnergyProjection::CreateStreams() {
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);

	int baseStreams = 4;
	int maxStreams = prop.multiProcessorCount * 3;

	m_num_streams = baseStreams;
	if (freeMem > (totalMem / 2)) {
		m_num_streams = min(maxStreams, baseStreams * 2);
	}

	std::cout << "Creating " << m_num_streams << " CUDA streams." << std::endl;

	m_streams = new cudaStream_t[m_num_streams];
	for (int i = 0; i < m_num_streams; ++i) {
		cudaStreamCreate(&m_streams[i]);
	}

}

void mutiEnergyProjection::ClearStreams() {
	if (!m_streams){
		std::cout << "ClearStreams Error: no streams\n";
		return;
	}
	for (int i = 0; i < m_num_streams; ++i) {
		cudaStreamDestroy(m_streams[i]);
	}
	delete[] m_streams;
	m_streams = nullptr;
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

void mutiEnergyProjection::ReadConfigFile(const char* filename) {
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
	std::cout << "Parameter reading completed" << std::endl;
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


void mutiEnergyProjection::readData(const char* filepath, float* data_cpu, float* data_gpu, int length, cudaStream_t stream) {
	{
		std::lock_guard<std::mutex> lock(fileMutex); // 确保线程安全
		std::ifstream file(filepath, std::ios::binary); // 以二进制模式打开文件

		if (!file) {
			std::cerr << "Could not open file: " << filepath << std::endl;
			exit(3);
		}

		// 读取数据
		file.read(reinterpret_cast<char*>(data_cpu), length * sizeof(float));

		if (!file) {
			std::cerr << "Error reading file: " << filepath << std::endl;
		}

		file.close(); // 关闭文件
		if (stream){
			CHECK_CUDA_ERROR(cudaMemcpyAsync(data_gpu, data_cpu, sizeof(float) * length, cudaMemcpyHostToDevice, stream));
		}
		else{
			CHECK_CUDA_ERROR(cudaMemcpy(data_gpu, data_cpu, sizeof(float) * length, cudaMemcpyHostToDevice));
		}
	}
}

void mutiEnergyProjection::writeData(const char* filepath, float* data_gpu, float* data_cpu, int length, cudaStream_t stream)
{
	{
		std::lock_guard<std::mutex> lock(fileMutex); // 确保线程安全
		if (m_use_stream) {
			CHECK_CUDA_ERROR(cudaMemcpyAsync(data_cpu, data_gpu, sizeof(float) * length, cudaMemcpyDeviceToHost, stream));
		}
		else {
			cudaMemcpy(data_cpu, data_gpu, sizeof(float) * length, cudaMemcpyDeviceToHost);
		}
		std::ofstream file(filepath, std::ios::binary); // 以二进制模式打开文件

		if (!file) {
			std::cerr << "Could not open file: " << filepath << std::endl;
			exit(3);
		}

		// 写入数据
		file.write(reinterpret_cast<char*>(data_cpu), length * sizeof(float));

		if (!file) {
			std::cerr << "Error writing to file: " << filepath << std::endl;
		}

		file.close(); // 关闭文件
	}
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

void mutiEnergyProjection::MallocStaticData() {
	if (m_process_type == ProcessType::ForwardPorjedction) {
		for (auto material : m_material_list) {
			//img_materials_cpu[material] = new float[m_FPConfig.imgDim * m_FPConfig.imgDim];
			img_materials_cpu[material] = new float[m_FPConfig.imgDim * m_FPConfig.imgDim * m_num_streams];
			sgm_materials_cpu[material] = new float[m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams];
			// 锁定内存
			if (!MemoryAgent::LockMemory(sgm_materials_cpu[material], m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams)) {
				exit(-1);
			}

			cudaMalloc((void**)&img_materials[material], sizeof(float) * m_FPConfig.imgDim * m_FPConfig.imgDim * m_num_streams);
			cudaMalloc((void**)&sgm_materials[material], sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams);
		}
	}
	else if (m_process_type == ProcessType::MEProcess) {
		for (auto material : m_material_list) {
			sgm_materials_cpu[material] = new float[m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams];
			cudaMalloc((void**)&sgm_materials[material], sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams);
		}
		cudaMalloc((void**)&sinogram, sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams);
		sinogram_cpu = new float[m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams];
		if (MemoryAgent::LockMemory(sinogram_cpu, m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams)) {
			exit(-1);
		}
	}
	else {
		for (auto material : m_material_list) {
			img_materials_cpu[material] = new float[m_FPConfig.imgDim * m_FPConfig.imgDim * m_num_streams];
			sgm_materials_cpu[material] = new float[m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams];
			if (MemoryAgent::LockMemory(sgm_materials_cpu[material], m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams)) {
				exit(-1);
			}
			cudaMalloc((void**)&img_materials[material], sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams);
			cudaMalloc((void**)&sgm_materials[material], sizeof(float) * m_FPConfig.detEltCount * m_FPConfig.views * m_num_streams);
		}
		cudaMalloc((void**)&sinogram, sizeof(float) * m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams);
		sinogram_cpu = new float[m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams];
		if (MemoryAgent::LockMemory(sinogram_cpu, m_MEConfig.sgmWidth * m_MEConfig.sgmHeight * m_num_streams)) {
			exit(-1);
		}
	}
}


bool mutiEnergyProjection::do_forward_projection(int img_offset, int sgm_offset, cudaStream_t stream) {
	if (m_light_process == nullptr){
		printf("Forward Projection Error, Create lightSource first.\n");
		return false;
	}
	std::cout << "do_forward_projection:" << std::endl;
	for (auto material : m_material_list){
		m_light_process->ForwardProjectionBilinear(img_materials[material] + img_offset,
			sgm_materials[material] + sgm_offset, stream);
	}
}

bool mutiEnergyProjection::do_forward_projection(float *img, float *sgm, cudaStream_t stream) {
	if (m_light_process == nullptr) {
		printf("Forward Projection Error, Create lightSource first.\n");
		return false;
	}
	m_light_process->ForwardProjectionBilinear(img, sgm, stream);
}

bool mutiEnergyProjection::do_energy_process(int e_idx, int offset, cudaStream_t stream) {
	if (m_energy_process == nullptr) {
		printf("Muti-Energy Process Error, Create mutiEnergyProcess first.\n");
		return false;
	}

	m_energy_process->SpectralPhotonCounting(e_idx, m_material_list, sgm_materials, sinogram, offset, stream);
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
		auto imgSize = m_FPConfig.imgDim * m_FPConfig.imgDim;
		auto sgmSize = m_FPConfig.detEltCount * m_FPConfig.views;
		for (auto item : PathData.Materials) {
			auto input_dir = PathData.FPInputImageMaterialDir[item];
			auto output_dir = PathData.FPOutputSgmMaterialDir[item];
			if (m_use_stream) {
				omp_set_num_threads(m_num_streams);
				#pragma omp parallel
				{
				#pragma omp for
					// 处理文件
					for (int i = 0; i < image_material_names[item].size(); i ++) {
						int streamIndex = i % m_num_streams; // 轮流使用流
						auto img_offset = imgSize * streamIndex;
						auto sgm_offset = sgmSize * streamIndex;

						readData((input_dir + '/' + image_material_names[item][i]).c_str(), img_materials_cpu[item] + img_offset,
							img_materials[item] + img_offset, imgSize, m_streams[streamIndex]);
						do_forward_projection(img_materials[item] + img_offset, sgm_materials[item] + sgm_offset, m_streams[streamIndex]);
						writeData((output_dir + '/' + sgm_material_names[item][i]).c_str(), sgm_materials[item] + sgm_offset,
							sgm_materials_cpu[item] + sgm_offset, sgmSize, m_streams[streamIndex]);
					}
				}
			}
			else {
				for (auto item : PathData.Materials) {
					auto input_dir = PathData.FPInputImageMaterialDir[item];
					auto output_dir = PathData.FPOutputSgmMaterialDir[item];
					for (int i = 0; i < image_material_names[item].size(); i++) {
						readData((input_dir + '/' + image_material_names[item][i]).c_str(), img_materials_cpu[item],
							img_materials[item], imgSize, nullptr);
						do_forward_projection(img_materials[item], sgm_materials[item], nullptr);
						writeData((output_dir + '/' + sgm_material_names[item][i]).c_str(), sgm_materials[item],
							sgm_materials_cpu[item], sgmSize, nullptr);
					}
				}
			}
		}
	}
	else if (m_process_type == ProcessType::MEProcess){
		std::unordered_map<MaterialType, std::vector<std::string>>material_names;
		for (auto item : m_material_list){
			material_names[item] = GetInputFileNames(m_MEConfig.MePath.MEInputSgmMaterialsDir[item].c_str(),
				m_MEConfig.MePath.MEMaterialFilter[item].c_str());
		}
		auto sgmSize = m_MEConfig.sgmWidth * m_MEConfig.sgmHeight;
		for (int spec_idx = 0; spec_idx < m_MEConfig.MePath.SpectrumEnergys.size(); spec_idx++) {
			std::string energy = m_MEConfig.MePath.SpectrumEnergys[spec_idx];
			printf("Start process energy: %s\n", energy.c_str());
			std::vector<std::string>outputFileNames = GetOutputFileNames(material_names[m_MEConfig.MePath.materialToReplace],
				m_MEConfig.MePath.outputFilesReplace, m_MEConfig.MePath.outputFilesNamePrefix[energy]);

			auto dirs = m_MEConfig.MePath.MEInputSgmMaterialsDir;
			if (m_use_stream){
				omp_set_num_threads(m_num_streams);
				#pragma omp parallel
				{
				#pragma omp for
					for (int i = 0; i < outputFileNames.size(); i ++) {
						printf("i = %d, threadIndex = %d\n", i, omp_get_thread_num());
						int streamIndex = i % m_num_streams; // 轮流使用流
						auto sgm_offset = sgmSize * streamIndex;
						for (auto item : m_material_list) {
							std::string name = dirs[item] + '/' + material_names[item][i];
							readData(name.c_str(), sgm_materials_cpu[item] + sgm_offset,
								sgm_materials[item] + sgm_offset, sgmSize, m_streams[streamIndex]);
						}
						timer.startCUDA();
						do_energy_process(spec_idx, sgm_offset, m_streams[streamIndex]);
						timer.stopCUDA();
						std::cout << "\nCUDA Execution Time: " << timer.getElapsedTimeCUDA() << " milliseconds" << std::endl;
						writeData((m_MEConfig.MePath.MEOutputDirs[energy] + '/' + outputFileNames[i]).c_str(),
							sinogram + sgm_offset, sinogram_cpu + sgm_offset, sgmSize, m_streams[streamIndex]);
					}
				}
			}
			else{
				for (size_t i = 0; i < outputFileNames.size(); i++) {
					for (auto item : m_material_list) {
						std::string name = dirs[item] + '/' + material_names[item][i];
						readData((dirs[item] + '/' + material_names[item][i]).c_str(), sgm_materials_cpu[item], sgm_materials[item], sgmSize, nullptr);
					}
					timer.startCUDA();
					do_energy_process(spec_idx, 0, nullptr);
					timer.stopCUDA();
					std::cout << "\nCUDA Execution Time: " << timer.getElapsedTimeCUDA() << " milliseconds" << std::endl;
					writeData((m_MEConfig.MePath.MEOutputDirs[energy] + '/' + outputFileNames[i]).c_str(), sinogram, sinogram_cpu, sgmSize, nullptr);
				}
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
		auto imgSize = m_FPConfig.imgDim * m_FPConfig.imgDim;
		auto sgmSize = m_MEConfig.sgmWidth * m_MEConfig.sgmHeight;
		for (int spec_idx = 0; spec_idx < m_MEConfig.MePath.SpectrumEnergys.size(); spec_idx++) {
			std::string energy = m_MEConfig.MePath.SpectrumEnergys[spec_idx];
			printf("\nStart process energy: %s\n", energy.c_str());
			std::vector<std::string>outputFileNames = GetOutputFileNames(material_names[m_MEConfig.MePath.materialToReplace],
				m_MEConfig.MePath.outputFilesReplace, m_MEConfig.MePath.outputFilesNamePrefix[energy]);

			auto dirs = m_FPConfig.FpPath.FPInputImageMaterialDir;
			auto outputSize = outputFileNames.size();

			if (m_use_stream){
				omp_set_num_threads(m_num_streams);
				#pragma omp parallel
				{
				#pragma omp for
					//int thread_count = omp_get_num_threads();

					for (int i = 0; i < outputSize; i ++) {
						int streamIndex = i % m_num_streams; // 轮流使用流
						auto img_offset = imgSize * streamIndex;
						auto sgm_offset = sgmSize * streamIndex;

						for (auto item : m_material_list) {
							readData((dirs[item] + '/' + material_names[item][i]).c_str(),
								img_materials_cpu[item] + img_offset, img_materials[item] + img_offset, imgSize, m_streams[streamIndex]);
						}
						timer.startCUDA();
						do_forward_projection(img_offset, sgm_offset, m_streams[streamIndex]); // 确保核函数使用当前流
						do_energy_process(spec_idx, sgm_offset, m_streams[streamIndex]); // 确保核函数使用当前流
						timer.stopCUDA();
						std::cout << "\nCUDA Execution Time: " << timer.getElapsedTimeCUDA() << " milliseconds" << std::endl;
						writeData((m_MEConfig.MePath.MEOutputDirs[energy] + '/' + outputFileNames[i]).c_str(), sinogram + sgm_offset, sinogram_cpu + sgm_offset, sgmSize, m_streams[streamIndex]);
					}
				}
			}
			else{
				for (size_t i = 0; i < outputFileNames.size(); i++) {
					for (auto item : m_material_list) {
						readData((dirs[item] + '/' + material_names[item][i]).c_str(), img_materials_cpu[item], img_materials[item], imgSize, nullptr);
					}
					timer.startCUDA();
					do_forward_projection(0, 0, nullptr);
					do_energy_process(spec_idx, 0, nullptr);
					timer.stopCUDA();
					std::cout << "\nCUDA Execution Time: " << timer.getElapsedTimeCUDA() << " milliseconds" << std::endl;
					writeData((m_MEConfig.MePath.MEOutputDirs[energy] + '/' + outputFileNames[i]).c_str(), sinogram, sinogram_cpu, sgmSize, nullptr);
				}
			}
		}
	}
	return true;
}