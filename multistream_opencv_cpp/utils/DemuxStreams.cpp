#include "DemuxStreams.hpp"

DemuxStreams::DemuxStreams(const int numStreams, const std::vector<std::shared_ptr<SynchronizedQueue>>& syncQueue)
    : display(numStreams), frameQueues(syncQueue) {
    frameQueues = syncQueue;
}

void DemuxStreams::readAndDisplayStreams() {
    std::map<int, cv::Mat> frames;
    for (std::shared_ptr<SynchronizedQueue> queue : frameQueues) {
        cv::Mat frame = queue->pop();
        int queueIndex = queue->getStreamIndex();
        frames.insert(std::make_pair(queueIndex, frame));
    }
    display.displayFrames(frames);
}



