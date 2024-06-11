#include <string>
#include <vector>

class singleSgm {
public:
	int sgm_num = 0;
	std::string inputDir;
	std::string outputDir;
	int sgmWidth;
	int sgmHeight;
	float mu;
	float Dose = 1e5;

	float* sgm = nullptr;
	float* sinogram = nullptr;
	float* sgmCpu = nullptr;
	float* sinogramCpu = nullptr;

	singleSgm(int num, int width, int height, float u) :sgm_num(num), sgmWidth(width), sgmHeight(height), mu(u) {}
	~singleSgm();

	void readSgm(const char* filename);
	void sgmToSinogram();
	void saveSinogram(const char* filename, float* sinogram);
	void InitSgm();

};