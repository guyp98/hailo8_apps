#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>

enum class EOI {
    TRUE,
    FALSE
}; 

enum class StreamType {
    CAMERA,
    VIDEO
};

struct FrameData {
    cv::Mat frame;
    int streamIndex;
    EOI eoi;
};

std::queue<FrameData> frameQueue;
std::mutex mtx;
std::condition_variable cv_lock;

void streamThread(int streamIndex, StreamType streamType, const std::string& path) {
    cv::VideoCapture capture;
    if (streamType == StreamType::CAMERA) {
        capture.open(streamIndex);
    } else {
        capture.open(path);
    }

    if (!capture.isOpened()) {
        if (streamType == StreamType::CAMERA) {
            std::cout << "Camera " << streamIndex << " cannot be opened." << std::endl;
        } else {
            std::cout << "Video " << path << " cannot be opened." << std::endl;
        }
        return;
    }

    while (true) {
        cv::Mat frame;
        capture >> frame;

        if (frame.empty()) {
            if (streamType == StreamType::CAMERA) {
                std::cout << "Camera " << streamIndex << " disconnected." << std::endl;
            } else {
                std::cout << "Video " << path << " end." << std::endl;
            }
            int EOI = 1;
            frameQueue.push({frame, streamIndex, EOI::TRUE});
            break;
        }
        std::lock_guard<std::mutex> lock(mtx);
        frameQueue.push({frame, streamIndex, EOI::FALSE});
        cv_lock.notify_one();
    }

    capture.release();
    // cv::destroyWindow(std::to_string(streamIndex));
}

void displayThread(int numStreams) {
    
    int live_streams = numStreams;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_lock.wait(lock, [] { return !frameQueue.empty(); });

        FrameData frameData = frameQueue.front();
        frameQueue.pop();

        lock.unlock();
        if(frameData.eoi == EOI::TRUE) {
            cv::destroyWindow(std::to_string(frameData.streamIndex));
            live_streams--;
            if (live_streams <= 0) {
                break;
            }
            
        }
        else{
            std::string windowName = std::to_string(frameData.streamIndex);
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::resizeWindow(windowName, 640, 480);
            cv::imshow(windowName, frameData.frame);
        }
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
}

int main() {
    int numStreams = 3;  // Set the number of cameras you want to use
    StreamType streamType = StreamType::VIDEO;  // Set the stream type
    std::string videoPath = "../car_drive.mp4";  // Set the video path

    std::vector<std::thread> threads;

    std::thread displayThreadObj(displayThread, numStreams);

    for (int i = 0; i < numStreams; ++i) {
        threads.emplace_back(streamThread, i, streamType, videoPath);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); 
    }

    

    for (auto& thread : threads) {
        thread.join();
    }

    displayThreadObj.join();

    return 0;
}
