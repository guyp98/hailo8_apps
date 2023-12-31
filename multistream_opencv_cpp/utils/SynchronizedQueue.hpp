#ifndef SYNCHRONIZEDQUEUE_HPP
#define SYNCHRONIZEDQUEUE_HPP

#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <mutex>
#include <queue>



class SynchronizedQueue {
private:
    int streamIndex;
    std::queue<cv::Mat> frameQueue;
    std::mutex mtx;
    std::condition_variable con_v;

public:
    SynchronizedQueue(int streamIndex);
    void push(cv::Mat frame);
    cv::Mat pop();
    int getStreamIndex() const;
    bool empty() ;
};

#endif  // SYNCHRONIZEDQUEUE_HP