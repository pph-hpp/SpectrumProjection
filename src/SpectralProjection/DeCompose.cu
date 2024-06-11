#include "DeCompose.cuh"
#include <io.h>

__global__ void materials(float* sgmLowEnergy, float* sgmHighEnergy, float* sgmBone, float* sgmWater,
	float* paramFitBone, float* paramFitWater, int width, int height, int order) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < height && col < width)
	{
		int idx = row * width + col;
		float x1 = sgmLowEnergy[idx];
		float x2 = sgmHighEnergy[idx];
		sgmBone[idx] = 0;
		sgmWater[idx] = 0;

		int index = 0;
		/*for (int i = 0; i <= order; i++)		// 指定拟合阶次，速度较慢
		{
			for (int j = 0; j <= i; j++) {
				sgmBone[idx] += paramFitBone[index] * pow(x1, (i - j)) * pow(x2, j);
				sgmWater[idx] += paramFitWater[index] * pow(x1, (i - j)) * pow(x2, j);
				index++;
			}
		}*/

		sgmBone[idx] = paramFitBone[0] + paramFitBone[1] * x1 + paramFitBone[2] * x2 + paramFitBone[3] * x1 * x1 + paramFitBone[4] * x1 * x2 + paramFitBone[5] * x2 * x2;
		sgmWater[idx] = paramFitWater[0] + paramFitWater[1] * x1 + paramFitWater[2] * x2 + paramFitWater[3] * x1 * x1 + paramFitWater[4] * x1 * x2 + paramFitWater[5] * x2 * x2;
	}

}


std::vector<std::string> DeCompose::GetInputFileNames(const std::string& dir, const std::string& filter)
{

	std::vector<std::string> filenames;

	std::string buffer = dir + "/" + filter;
	_finddata_t fileInfo;   //存放文件信息的结构体
	intptr_t hFile;
	hFile = _findfirst(buffer.c_str(), &fileInfo); //找第一个文件

	if (hFile == -1L) {
		//没找到指定类型的文件
		std::cout << "No " << filter << " files in current directory!" << std::endl;
	}
	else {
		std::string fullFilePath;
		do {
			fullFilePath.clear();
			fullFilePath = dir + "\\" + fileInfo.name;
			filenames.push_back(fileInfo.name);

		} while (_findnext(hFile, &fileInfo) == 0);  //如果找到下个文件的名字成功的话就返回0,否则返回-1  
		_findclose(hFile);
	}
	return filenames;
}

std::vector<std::string> DeCompose::GetOutputFileNames(const std::vector<std::string>& inputFileNames, const std::vector<std::string>& replace, const std::string& prefix)
{
	std::vector<std::string> outputFiles;

	for (size_t fileIdx = 0; fileIdx < inputFileNames.size(); fileIdx++)
	{
		std::string outputFile = inputFileNames[fileIdx];
		auto pos = outputFile.find(replace[0]);
		if (pos == std::string::npos)
		{
			fprintf(stderr, "Did not find substring \"%s\" to be replaced!\n", replace[0].c_str());
			exit(2);
		}
		outputFile.replace(pos, replace[0].size(), replace[1]);
		outputFiles.push_back(prefix + outputFile);
	}

	return outputFiles;
}

DeCompose::DeCompose(int w, int h, int o) : width(w), height(h), order(o) {
	cudaMalloc((void**)&sgmHighEnergy, sizeof(float) * w * h);
	cudaMalloc((void**)&sgmLowEnergy, sizeof(float) * w * h);
	cudaMalloc((void**)&sgmBone, sizeof(float) * w * h);
	cudaMalloc((void**)&sgmWater, sizeof(float) * w * h);
	paramNum = ((o + 1) * (o + 2)) / 2;
	cudaMalloc((void**)&paramFitBone, sizeof(float) * paramNum);
	cudaMalloc((void**)&paramFitWater, sizeof(float) * paramNum);
	sgmHighEnergyCpu = new float[w * h];
	sgmLowEnergyCpu = new float[w * h];
	sgmBoneCpu = new float[w * h];
	sgmWaterCpu = new float[w * h];

}


DeCompose::~DeCompose()
{
	if (sgmHighEnergy != nullptr) {
		cudaFree(sgmHighEnergy);
		sgmHighEnergy = nullptr;
	}
	if (sgmLowEnergy != nullptr) {
		cudaFree(sgmLowEnergy);
		sgmLowEnergy = nullptr;
	}
	if (sgmBone != nullptr) {
		cudaFree(sgmBone);
		sgmBone = nullptr;
	}
	if (sgmWater != nullptr) {
		cudaFree(sgmWater);
		sgmWater = nullptr;
	}
	if (paramFitBone != nullptr) {
		cudaFree(paramFitBone);
		paramFitBone = nullptr;
	}
	if (paramFitWater != nullptr) {
		cudaFree(paramFitWater);
		paramFitWater = nullptr;
	}
	if (sgmHighEnergyCpu != nullptr)
	{
		delete[] sgmHighEnergyCpu;
	}
	if (sgmLowEnergyCpu != nullptr)
	{
		delete[] sgmLowEnergyCpu;
	}
	if (sgmBoneCpu != nullptr)
	{
		delete[] sgmBoneCpu;
	}
	if (sgmWaterCpu != nullptr)
	{
		delete[] sgmWaterCpu;
	}

}

void DeCompose::readSgm(const char* fileNameLow, const char* fileNameHigh) {
	FILE* fpLow = fopen(fileNameLow, "rb");
	FILE* fpHigh = fopen(fileNameHigh, "rb");
	if (fpLow == nullptr || fpHigh == nullptr)
	{
		perror("Error opening file");
		exit(3);
	}
	fread(sgmLowEnergyCpu, sizeof(float), width * height, fpLow);
	fread(sgmHighEnergyCpu, sizeof(float), width * height, fpHigh);

	cudaMemcpy(sgmLowEnergy, sgmLowEnergyCpu, sizeof(float) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(sgmHighEnergy, sgmHighEnergyCpu, sizeof(float) * width * height, cudaMemcpyHostToDevice);

	fclose(fpLow);
	fclose(fpHigh);


}
void DeCompose::saveSgm(const char* fileNameBone, const char* fileNameWater) {
	FILE* fpBone = fopen(fileNameBone, "wb");
	FILE* fpWater = fopen(fileNameWater, "wb");
	if (fpBone == nullptr || fpWater == nullptr)
	{
		perror("Error opening file");
		exit(0);
	}
	cudaMemcpy(sgmBoneCpu, sgmBone, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(sgmWaterCpu, sgmWater, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	fwrite(sgmBoneCpu, sizeof(float), width * height, fpBone);
	fwrite(sgmWaterCpu, sizeof(float), width * height, fpWater);

	fclose(fpBone);
	fclose(fpWater);

}

void DeCompose::material_decompose() {
	dim3 grid((width + 15) / 16, (height + 15) / 16);
	dim3 block(16, 16);

	materials << <grid, block >> > (sgmLowEnergy, sgmHighEnergy, sgmBone, sgmWater,
		paramFitBone, paramFitWater, width, height, order);
	cudaDeviceSynchronize();
}

void DeCompose::readParam(std::vector<float>& paramBone, std::vector<float>& paramWater) {

	if (paramBone.size() != paramNum || paramWater.size() != paramNum)
	{
		printf("input error");
		exit(0);
	}

	cudaMemcpy(paramFitBone, paramBone.data(), sizeof(float) * paramNum, cudaMemcpyHostToDevice);
	cudaMemcpy(paramFitWater, paramWater.data(), sizeof(float) * paramNum, cudaMemcpyHostToDevice);
}

void DeCompose::readParam(std::vector<std::vector<float>>& params) {
	std::vector<float> paramBone(paramNum);
	std::vector<float> paramWater(paramNum);

	for (int i = 0; i < paramNum; i++)
	{
		paramWater[i] = params[0][i];
		paramBone[i] = params[1][i];
	}

	cudaMemcpy(paramFitBone, paramBone.data(), sizeof(float) * paramNum, cudaMemcpyHostToDevice);
	cudaMemcpy(paramFitWater, paramWater.data(), sizeof(float) * paramNum, cudaMemcpyHostToDevice);
}

void DeCompose::readConfigFile(const char* filename) {
	namespace js = rapidjson;

	// load the config file
	std::cout << "参数读取" << std::endl;
	std::ifstream ifs(filename);
	if (!ifs)
	{
		printf("Cannot open config file '%s'!\n", filename);
		exit(-2);
	}
	rapidjson::IStreamWrapper isw(ifs);
	rapidjson::Document doc;
	doc.ParseStream<js::kParseCommentsFlag | js::kParseTrailingCommasFlag>(isw);

	this->lowFilter = doc["lowFilter"].GetString();
	this->highFilter = doc["highFilter"].GetString();

	this->sgmHighDir = doc["inputHighDir"].GetString();
	this->sgmLowDir = doc["inputLowDir"].GetString();

	this->sgmBoneDir = doc["outputBoneDir"].GetString();
	this->sgmWaterDir = doc["outputWaterDir"].GetString();

	/*this->sgmWidth = doc["sgmWidth"].GetInt();
	this->sgmHeight = doc["sgmHeight"].GetInt();
	this->energyNum = doc["energyNum"].GetInt();*/

	const js::Value& replaceJs = doc["OutputFileReplaceBone"];
	for (js::SizeType i = 0; i < replaceJs.Size(); i++)
	{
		this->replaceBone.push_back(replaceJs[i].GetString());
	}

	const js::Value& replaceJsw = doc["OutputFileReplaceWater"];
	for (js::SizeType i = 0; i < replaceJsw.Size(); i++)
	{
		this->replaceWater.push_back(replaceJsw[i].GetString());
	}

	if (doc.HasMember("waterPath"))
	{
		this->waterPath = doc["waterPath"].GetString();
	}
	else {
		this->waterPath = "water.txt";
	}

	if (doc.HasMember("bonePath"))
	{
		this->bonePath = doc["bonePath"].GetString();
	}
	else {
		this->bonePath = "bone.txt";
	}

	if (doc.HasMember("highEnergyPath"))
	{
		this->highEnergyPath = doc["highEnergyPath"].GetString();
	}
	else {
		this->highEnergyPath = "120kvp.txt";
	}

	if (doc.HasMember("lowEnergyPath"))
	{
		this->lowEnergyPath = doc["lowEnergyPath"].GetString();
	}
	else {
		this->lowEnergyPath = "80kvp.txt";
	}

	std::cout << "参数读取完成" << std::endl;
}

void DeCompose::Init(const char* filename, int o) {
	readConfigFile(filename);
	cudaMalloc((void**)&sgmHighEnergy, sizeof(float) * sgmWidth * sgmHeight);
	cudaMalloc((void**)&sgmLowEnergy, sizeof(float) * sgmWidth * sgmHeight);
	cudaMalloc((void**)&sgmBone, sizeof(float) * sgmWidth * sgmHeight);
	cudaMalloc((void**)&sgmWater, sizeof(float) * sgmWidth * sgmHeight);
	paramNum = ((o + 1) * (o + 2)) / 2;
	cudaMalloc((void**)&paramFitBone, sizeof(float) * paramNum);
	cudaMalloc((void**)&paramFitWater, sizeof(float) * paramNum);
	sgmHighEnergyCpu = new float[sgmWidth * sgmHeight];
	sgmLowEnergyCpu = new float[sgmWidth * sgmHeight];
	sgmBoneCpu = new float[sgmWidth * sgmHeight];
	sgmWaterCpu = new float[sgmWidth * sgmHeight];

}


void DeCompose::readEnergyData(std::string filename, float* data, float p) {
	std::ifstream inputFile;
	float* param_cpu = new float[this->energyNum];
	//Opens data file for reading.
	inputFile.open(filename);

	//Creates vector, initially with 0 points.
	/*vector<Point> data(0);*/
	int temp_x;
	double temp_y;
	int i = 0;

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
	cudaMemcpy(data, param_cpu, sizeof(float) * energyNum, cudaMemcpyHostToDevice);
}

//void DeCompose::Init(const mutiEnergyProcess* proc, int o) {
//
//	this->sgmWidth = proc->config.sgmWidth;
//	this->sgmHeight = proc->config.sgmHeight;
//	this->energyNum = proc->config.energyNum;
//
//	this->sgmLowDir = proc->config.outputHighDir;
//	this->sgmHighDir = proc->config.outputLowDir;
//	this->sgmBoneDir = proc->config.outputBoneDir;
//	this->sgmWaterDir = proc->config.outputWaterDir;
//
//	this->waterPath = proc->config.waterPath;
//	this->bonePath = proc->config.bonePath;
//	this->highEnergyPath = proc->config.highEnergyPath;
//	this->lowEnergyPath = proc->config.lowEnergyPath;
//
//	this->param->start(bonePath, waterPath, lowEnergyPath, highEnergyPath);
//
//	std::cout << "参数读取完成" << std::endl;
//	sgmHighEnergy = proc->sinogramHigh;
//	sgmLowEnergy = proc->sinogramLow;
//
//	cudaMalloc((void**)&sgmBone, sizeof(float) * sgmWidth * sgmHeight);
//	cudaMalloc((void**)&sgmWater, sizeof(float) * sgmWidth * sgmHeight);
//
//	paramNum = ((o + 1) * (o + 2)) / 2;
//
//	cudaMalloc((void**)&paramFitBone, sizeof(float) * paramNum);
//	cudaMalloc((void**)&paramFitWater, sizeof(float) * paramNum);
//	sgmBoneCpu = new float[sgmWidth * sgmHeight];
//	sgmWaterCpu = new float[sgmWidth * sgmHeight];
//
//	readParam(param->params);
//	
//}


void DeCompose::use() {

	std::vector<std::string>lowFileNames = GetInputFileNames(this->sgmLowDir, this->lowFilter);
	std::vector<std::string>highFileNames = GetInputFileNames(this->sgmHighDir, this->highFilter);
	std::vector<std::string>outputBoneFileNames = GetOutputFileNames(lowFileNames, this->replaceBone, "");
	std::vector<std::string>outputWaterFileNames = GetOutputFileNames(lowFileNames, this->replaceWater, "");
	std::string sgmBoneFileName;
	std::string sgmWaterFileName;

	for (size_t i = 0; i < lowFileNames.size(); i++)
	{
		std::cout << i << "\n";

		//this->ReadImageFile((inputBoneDir + boneFileNames[i]).c_str(), (inputWaterDir + waterFileNames[i]).c_str());
		this->readSgm((sgmLowDir + '/' + lowFileNames[i]).c_str(), (sgmHighDir + '/' + highFileNames[i]).c_str());

		/*decompose.readParam(paramBone, paramWater);*/
		this->material_decompose();
		sgmBoneFileName = sgmBoneDir + '/' + outputBoneFileNames[i];
		sgmWaterFileName = sgmWaterDir + '/' + outputWaterFileNames[i];
		this->saveSgm(sgmBoneFileName.c_str(), sgmWaterFileName.c_str());

		printf("%d\n", i);

	}
}