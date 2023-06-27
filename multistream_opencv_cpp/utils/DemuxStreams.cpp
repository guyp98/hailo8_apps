#include "DemuxStreams.hpp"

DemuxStreams::DemuxStreams(const int numStreams, const std::vector<std::shared_ptr<SynchronizedQueue>>& syncQueue)
    : display(numStreams), frameQueues(syncQueue) {
    frameQueues = syncQueue;
}

void DemuxStreams::readAndDisplayStreams() {
    std::map<int, cv::Mat> frames;
    while(true){
        for (std::shared_ptr<SynchronizedQueue> queue : frameQueues) {
            int queueIndex = queue->getStreamIndex();
            cv::Mat frame;
            if(queue->empty()){
                std::cout << "Queue " << queueIndex << " is empty" << std::endl;
                continue;
            }
            frame = queue->pop();
             auto start = std::chrono::high_resolution_clock::now();
        
        
        
            display.displayFrames(frame, queueIndex);
                
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "display On Host Runtime: " << duration.count() << " milliseconds" << std::endl;
        }
    }
    
}



