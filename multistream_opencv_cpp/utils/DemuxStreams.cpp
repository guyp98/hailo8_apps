#include "DemuxStreams.hpp"

DemuxStreams::DemuxStreams(const int numStreams, const std::vector<std::shared_ptr<SynchronizedQueue>>& syncQueue)
    : display(numStreams), frameQueues(syncQueue) {
    frameQueues = syncQueue;
    lastFrames = std::map<int, cv::Mat>();
    // for (int i = 0; i < numStreams; i++)
    //     lastFrames.insert(std::make_pair(i, cv::Mat::zeros(?,? , CV_8UC3)));//ToDO: fill in the blanks
}

void DemuxStreams::readAndDisplayStreams() {
    std::map<int, cv::Mat> frames;
    for (std::shared_ptr<SynchronizedQueue> queue : frameQueues) {
        int queueIndex = queue->getStreamIndex();
        cv::Mat frame;
        if (queue->empty()){
            frame = lastFrames[queueIndex];
            std::cout<< "used the last frame stream: " << std::to_string(queueIndex) << std::endl;
        }
        else
            frame = queue->pop();
        frames.insert(std::make_pair(queueIndex, frame));

        lastFrames[queueIndex] = frame;
    }
    display.displayFrames(frames);
}



