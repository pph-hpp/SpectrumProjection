#include "stdafx.h"
#include <string>
#include <iostream>
#include "mutiEnergyProjection.h"


int main()
{
	
	cudaDeviceProp prop = MemoryAgent::getDeviceProperties();
	MemoryAgent::SetWorkingSetSize(50, 200);
	MemoryAgent::GetWorkingSetSize();

	mutiEnergyProjection proj("src/config.jsonc", prop);
	std::cout << "Creation completed" << std::endl;
	proj.process();
	std::cout << "Execution completed" << std::endl;
	/*lightSource light;
	std::string configPath = "config.jsonc";

	light.ReadConfigFile(configPath.c_str());

	light.InitParam();
	light.config.coneBeam = false;

	fs::path inDir(lightSource::config.inputDir);
	fs::path outDir(lightSource::config.outputDir);

	for (size_t i = 0; i < lightSource::config.inputFiles.size(); i++)
	{
		printf("    \nForward projection %s ...", lightSource::config.inputFiles[i].c_str());

		light.ReadImageFile((inDir / lightSource::config.inputFiles[i]).string().c_str());

		light.ForwardProjectionBilinearAndSave((outDir / lightSource::config.outputFiles[i]).string().c_str());

		printf("\n->\tSaved to file %s\n", lightSource::config.outputFiles[i].c_str());
	}*/
	return 0;
}