// Record a fixed MJPG clip from a stream so A/B runs can replay identical frames.
// Standalone helper (not part of the CMake build); build with the same OpenCV as rfdetr_camera:
//   C=/home/models/stack/opencv4/lib
//   g++ -O2 -std=c++17 core/apps/record_clip.cpp -o build-camera/record_clip \
//       -I/home/models/stack/opencv4/include/opencv4 -L$C -lopencv_videoio -lopencv_imgcodecs \
//       -lopencv_core -Wl,-rpath,$C -Wl,--no-as-needed $C/libstdc++.so.6
// Usage: RECORD_SOURCE=<url-or-file> record_clip <output.avi> <frames> <keep_every_n> <output_fps>
// The source comes from the environment so credentials never appear in the process list.
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const char* source = std::getenv("RECORD_SOURCE");
    if (argc != 5 || !source || !*source) {
        std::cerr << "usage: RECORD_SOURCE=<source> record_clip <output.avi> <frames> <keep_every_n> <fps>\n";
        return 2;
    }
    const std::string output = argv[1];
    const int frames = std::atoi(argv[2]), every = std::atoi(argv[3]);
    const double fps = std::atof(argv[4]);
    if (frames < 1 || every < 1 || fps <= 0) { std::cerr << "frames, keep_every_n and fps must be positive\n"; return 2; }

    cv::VideoCapture capture;
    const std::string text(source);
    const bool network = text.rfind("rtsp", 0) == 0 || text.rfind("http", 0) == 0;
    const bool opened = network
        ? capture.open(text, cv::CAP_FFMPEG, {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv::CAP_PROP_READ_TIMEOUT_MSEC, 10000})
        : capture.open(text);
    if (!opened) { std::cerr << "cannot open source\n"; return 1; }

    cv::VideoWriter writer;
    int kept = 0;
    long read = 0, failures = 0;
    while (kept < frames) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            if (++failures > 20) break;
            continue;
        }
        failures = 0;
        if (read++ % every) continue;
        if (!writer.isOpened() &&
            !writer.open(output, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size())) {
            std::cerr << "cannot open output writer\n";
            return 1;
        }
        writer.write(frame);
        ++kept;
    }
    writer.release();
    std::cout << "frames_written=" << kept << " frames_read=" << read << '\n';
    return kept == frames ? 0 : 1;
}
