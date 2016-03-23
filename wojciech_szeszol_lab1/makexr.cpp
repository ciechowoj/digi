

#include <iostream>
#include <vector>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfStringAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfArray.h>

using namespace std;
using namespace Imf;

void save_image_exr(int width, int height, const float* data, const string& path) {
	Header header (width, height);
	header.channels().insert ("R", Channel (Imf::FLOAT));
	header.channels().insert ("G", Channel (Imf::FLOAT));
	header.channels().insert ("B", Channel (Imf::FLOAT));

	OutputFile file (path.c_str(), header);

	FrameBuffer framebuffer;		
		
	std::vector<float> data_copy(data, data + width * height * 3);

	auto R = Slice(Imf::FLOAT, (char*)(data_copy.data() + 0), sizeof(float) * 3, sizeof(float) * width * 3);
	auto G = Slice(Imf::FLOAT, (char*)(data_copy.data() + 1), sizeof(float) * 3, sizeof(float) * width * 3);
	auto B = Slice(Imf::FLOAT, (char*)(data_copy.data() + 2), sizeof(float) * 3, sizeof(float) * width * 3);

	framebuffer.insert("R", R);
	framebuffer.insert("G", G);
	framebuffer.insert("B", B);

	file.setFrameBuffer(framebuffer);
	file.writePixels(height);
}

int main()
{
	string path;
	int width, height;
	vector<float> data;

	cin >> path >> width >> height;
	data.resize(width * height * 3);

	for (int i = 0; i < width * height * 3; ++i) {
		cin >> data[i];
	}

	save_image_exr(width, height, data.data(), path);

	return 0;
}
