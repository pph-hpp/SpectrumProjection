#include "function.h"

std::vector<std::string> FileOperate::GetInputFileNames(const std::string& dir, const std::string& filter)
{
	/*namespace fs = std::filesystem;*/

	std::vector<std::string> filenames;
	/*
	if (!std::filesystem::is_directory(dir))
	{
		return filenames;
	}
	std::filesystem::directory_iterator end_iter;
	std::regex e(filter);

	for (auto&& fe : std::filesystem::directory_iterator(dir))
	{
		std::string file = fe.path().filename().string();

		if (std::regex_match(file, e))
		{
			filenames.push_back(file);
		}
	}
	return filenames;*/

	std::string buffer = dir + "/" + filter;
	_finddata_t fileInfo;   //����ļ���Ϣ�Ľṹ��
	intptr_t hFile;
	hFile = _findfirst(buffer.c_str(), &fileInfo); //�ҵ�һ���ļ�

	if (hFile == -1L) {
		//û�ҵ�ָ�����͵��ļ�
		std::cout << "No " << filter << " files in current directory!" << std::endl;
	}
	else {
		std::string fullFilePath;
		do {
			fullFilePath.clear();
			fullFilePath = dir + "\\" + fileInfo.name;
			filenames.push_back(fileInfo.name);

		} while (_findnext(hFile, &fileInfo) == 0);  //����ҵ��¸��ļ������ֳɹ��Ļ��ͷ���0,���򷵻�-1  
		_findclose(hFile);
	}
	return filenames;
}

std::vector<std::string> FileOperate::GetOutputFileNames(const std::vector<std::string>& inputFileNames, const std::vector<std::string>& replace, const std::string& prefix)
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