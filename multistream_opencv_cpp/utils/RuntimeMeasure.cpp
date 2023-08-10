#include "RuntimeMeasure.hpp"
#include <iostream>

RuntimeMeasure* RuntimeMeasure::getInstance() {
    static RuntimeMeasure instance;
    return &instance;
}

void RuntimeMeasure::startTimer(int processId) {
    std::lock_guard<std::mutex> lock(mtx);
    startTimes[processId] = std::chrono::high_resolution_clock::now();
}

double RuntimeMeasure::endTimer(int processId) {
    std::lock_guard<std::mutex> lock(mtx);
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTimes[processId]).count() / 1000.0;
    return elapsedTime;
}
