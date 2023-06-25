#include "SynchronizedQueue.hpp"

SynchronizedQueue::SynchronizedQueue(int streamIndex) : streamIndex(streamIndex) {}

void SynchronizedQueue::push(cv::Mat frame) {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "Pushing frame to queue " << streamIndex << std::endl;
    frameQueue.push(frame);
    // cv.notify_one();  // Notify waiting threads
}

cv::Mat SynchronizedQueue::pop() {
    std::lock_guard<std::mutex> lock(mtx);
    if (frameQueue.empty()) {
        std::cout << "Queue " << streamIndex << " is empty" << std::endl;
        return cv::Mat(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    }
    std::cout << "Popping frame from queue " << streamIndex << std::endl;
    // cv.wait(lock, [this] { return !frameQueue.empty(); });  // Wait until queue is not empty
    cv::Mat frame = frameQueue.front();
    frameQueue.pop();
    return frame;
}

int SynchronizedQueue::getStreamIndex() const {
    return streamIndex;
}
