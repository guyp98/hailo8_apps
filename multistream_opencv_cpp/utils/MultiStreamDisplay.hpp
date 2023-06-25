#ifndef MULTISTREAMDISPLAY_HPP
#define MULTISTREAMDISPLAY_HPP

#include <opencv2/opencv.hpp>

class MultiStreamDisplay {
private:
    int numStreams_;
    int gridRows_;
    int gridCols_;
    int cellSize_;
    cv::Mat gridFrame_;

public:
    MultiStreamDisplay(int numStreams, int cellSize = 400);
    void displayFrames(std::map<int, cv::Mat>& frames);
};

#endif  // MULTISTREAMDISPLAY_HPP
