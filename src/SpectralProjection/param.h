#pragma once
#include <string>
#include <vector>
#include <stdio.h>
#include <cmath>
#include<random>
#include<iostream>
#include <fstream>
#include <Eigen\Dense>
class DeCompose;

class Param
{
public:
	friend class DeCompose;
	Param(int l_w, int l_b, float i_w, float i_b, int energyNum);
	~Param();

	std::vector<std::vector<float>> paramFit(int len_water, int len_bone, float inter_w, float inter_b);

	void readWater(std::string filename);
	void readBone(std::string filename);
	void readSpec(std::string filenameLow, std::string filenameHigh);

	std::vector<std::vector<float>> start(std::string file_bone,std::string file_water,std::string file_lowSpec, std::string file_highSpec);
private:

	int energyNum;

	float* spec_high = nullptr;
	float* spec_low = nullptr;

	float* ubone = nullptr;
	float* uwater = nullptr;

	float pBone = 1;
	float pWater = 1;

	int total_len_water;
	int total_len_bone;
	float inter_water;
	float inter_bone;

};


