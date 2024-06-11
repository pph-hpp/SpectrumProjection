#include <vector>
#include <filesystem>
#include <io.h>
#include <string>
#include <iostream>

class FileOperate {
	static std::vector<std::string> GetInputFileNames(const std::string& dir, const std::string& filter);
	static std::vector<std::string> GetOutputFileNames(const std::vector<std::string>& inputFileNames, const std::vector<std::string>& replace, const std::string& prefix);

};

