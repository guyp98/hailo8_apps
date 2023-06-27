#ifndef MULTISTREAMDISPLAY_HPP
#define MULTISTREAMDISPLAY_HPP

#include <opencv2/opencv.hpp>
#include <cmath>

class MultiStreamDisplay {
private:
    int numStreams_;
    int gridRows_;
    int gridCols_;
    int cellSize_;
    cv::Mat gridFrame_;
    std::vector<double> lastTimestamps_;

public:
    MultiStreamDisplay(int numStreams, int cellSize = 400);
    void displayFrames(cv::Mat frame, int streamIndex);
};

#endif  // MULTISTREAMDISPLAY_HPP
