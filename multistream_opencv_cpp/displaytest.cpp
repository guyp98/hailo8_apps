#include "utils/DemuxStreams.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

int main() {
    std::vector<cv::VideoCapture> captures;
    captures.push_back(cv::VideoCapture("../car_drive.mp4"));  // First video stream
    captures.push_back(cv::VideoCapture("../car_drive.mp4"));  // Second video stream
    captures.push_back(cv::VideoCapture("../car_drive.mp4"));  // Third video stream
    captures.push_back(cv::VideoCapture("../car_drive.mp4"));  // Fourth video stream
    int numStreams = captures.size();

    std::vector<std::shared_ptr<SynchronizedQueue>> frameQueues;
    for (int i = 0; i < numStreams; i++) {
        auto queue = std::make_shared<SynchronizedQueue>(i);
        frameQueues.push_back(queue);
    }

    

    std::vector<cv::Mat> frames(numStreams);

    for (int i = 0; i < numStreams; i++) {
            captures[i] >> frames[i];
            frameQueues[i]->push(frames[i]);
        }

    // Create a thread for running the demuxStreams.readAndDisplayStreams() function
    std::thread displayThread([numStreams,frameQueues]() {
        DemuxStreams demuxStreams(numStreams, frameQueues);
        while(true)
            demuxStreams.readAndDisplayStreams();
    });

    while (true) {
        bool endReached = false;
        for (int i = 0; i < numStreams; i++) {
            captures[i] >> frames[i];
            if (frames[i].empty()) {
                endReached = true;
                break;
            }
            frameQueues[i]->push(frames[i].clone());
        }

        if (endReached)
            break;

        // demuxStreams.readAndDisplayStreams();
        
    }

    // Release the video capture resources
    for (cv::VideoCapture& capture : captures) {
        capture.release();
    }

    // Wait for the display thread to finish
    displayThread.join();

    return 0;
}
