#include "param.h"


Param::Param(int l_w, int l_b, float i_w, float i_b, int energyNum) : total_len_water(l_w), total_len_bone(l_b), inter_water(i_w), inter_bone(i_b), energyNum(energyNum) {
	spec_high = new float[energyNum];
	spec_low = new float[energyNum];

	ubone = new float[energyNum];
	uwater = new float[energyNum];
}

Param::~Param()
{
	if (spec_high != nullptr)
	{
		delete[] spec_high;
		spec_high = nullptr;
	}
	if (spec_low != nullptr)
	{
		delete[] spec_low;
		spec_low = nullptr;
	}
	if (ubone != nullptr)
	{
		delete[] ubone;
		ubone = nullptr;
	}
	if (uwater != nullptr)
	{
		delete[] uwater;
		uwater = nullptr;
	}

}


std::vector<std::vector<float>> Param::paramFit(int len_water, int len_bone, float inter_w, float inter_b) {
	int num_w = len_water / inter_w;
	int num_b = len_bone / inter_b;
	double l_bone = 0;
	double l_water = 0;

	double Sp_low = 0;
	double Sp_high = 0;

	double allSpecLow = 0;
	double allSpecHigh = 0;
	for (int i = 0; i < energyNum; i++)
	{
		allSpecHigh += spec_high[i];
		allSpecLow += spec_low[i];
	}

	int idx = 0;

	Eigen::MatrixXd Length(2, (num_w + 1) * (num_b + 1));
	Eigen::MatrixXd Postlog(6, (num_w + 1) * (num_b + 1));


	for (int iWater = 0; iWater <= num_w; iWater++)
	{
		l_water = iWater * inter_w;
		for (int jBone = 0; jBone <= num_b; jBone++) {
			idx = iWater * (num_b+1) + jBone;
			l_bone = jBone * inter_b;
			l_water = iWater * inter_w;
			Sp_low = 0;
			Sp_high = 0;

			for (int i = 0; i < energyNum; i++)
			{
				Sp_low += 1e5 * 1.0 * spec_low[i] * (exp(-ubone[i] * l_bone - uwater[i] * l_water)) / allSpecLow;
				Sp_high += 1e5 * 1.0 * spec_high[i] * (exp(-ubone[i] * l_bone - uwater[i] * l_water)) / allSpecHigh;
			}

			Sp_low = log(1e5 / Sp_low);
			Sp_high = log(1e5 / Sp_high);

			Length(0, idx) = l_water;
			Length(1, idx) = l_bone;
			Postlog.col(idx) << 1, Sp_low, Sp_high, Sp_low* Sp_low, Sp_low* Sp_high, Sp_high* Sp_high;

		}
	}

	Eigen::MatrixXd paramM(2, 6);
	printf("%s", "compute inverse");

	paramM = Length * Postlog.transpose() * (Postlog * Postlog.transpose()).inverse();

	/*std::vector<float>paramBone;
	std::vector<float>paramWater;*/
	std::vector<std::vector<float>>params(2, std::vector<float>(6));

	for (int i = 0; i < 6; i++)
	{
		/*paramBone.push_back(paramM(1, i));
		paramWater.push_back(paramM(0, i));*/

		params[0][i] = paramM(0, i);
		params[1][i] = paramM(1, i);
	}

	//for (int i = 0; i < 6; i++)
	//{
	//	printf("%f\t%f\n", params[0][i], params[1][i]);
	//}

	return params;

}


void Param::readWater(std::string filename) {
	std::cout << "readWater: " << filename << std::endl;
	std::ifstream inputFile;
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
		uwater[i] = float(temp_y) * pWater;
		i += 1;
	}

	if (!inputFile.eof())
		if (inputFile.fail()) std::cout << "Type mismatch during parsing." << std::endl;
		else std::cout << "Unknow problem during parsing." << std::endl;

	//Close data file.
	inputFile.close();
}

void Param::readBone(std::string filename) {
	std::cout << "readBone: "<< filename << std::endl;
	std::ifstream inputFile;
	inputFile.open(filename);

	int temp_x;
	double temp_y;
	int i = 0;

	while (inputFile.good()) {
		inputFile >> temp_x >> temp_y;
		ubone[i] = float(temp_y) * pBone;
		i += 1;
	}

	if (!inputFile.eof())
		if (inputFile.fail()) std::cout << "Type mismatch during parsing." << std::endl;
		else std::cout << "Unknow problem during parsing." << std::endl;

	inputFile.close();
}

void Param::readSpec(std::string filenameLow, std::string filenameHigh) {
	std::ifstream inputFileLow;
	inputFileLow.open(filenameLow);

	int temp_x;
	float temp_y;
	int i = 0;

	//Read contents of file till EOF.
	while (inputFileLow.good()) {
		inputFileLow >> temp_x >> temp_y;
		spec_low[i] = temp_y;
		i += 1;
	}

	if (!inputFileLow.eof())
		if (inputFileLow.fail()) std::cout << "Type mismatch during parsing." << std::endl;
		else std::cout << "Unknow problem during parsing." << std::endl;

	inputFileLow.close();


	std::ifstream inputFileHigh;
	inputFileHigh.open(filenameHigh);

	i = 0;

	//Read contents of file till EOF.
	while (inputFileHigh.good()) {
		inputFileHigh >> temp_x >> temp_y;
		spec_high[i] = temp_y;
		i += 1;
	}

	if (!inputFileHigh.eof())
		if (inputFileHigh.fail()) std::cout << "Type mismatch during parsing." << std::endl;
		else std::cout << "Unknow problem during parsing." << std::endl;

	inputFileHigh.close();

}

std::vector<std::vector<float>> Param::start(std::string file_bone, std::string file_water, std::string file_lowSpec, std::string file_highSpec) {
	readBone(file_bone);
	readWater(file_water);
	readSpec(file_lowSpec, file_highSpec);

	std::vector<std::vector<float>>params(2, std::vector<float>(6));

	params = paramFit(total_len_water, total_len_bone, inter_water, inter_bone);
	return params;
}


