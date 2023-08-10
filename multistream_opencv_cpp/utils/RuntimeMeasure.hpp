#ifndef RUNTIME_MEASURE_HPP
#define RUNTIME_MEASURE_HPP

#include <chrono>
#include <thread>
#include <mutex>
#include <map>

class RuntimeMeasure {
private:
    std::mutex mtx;
    std::map<int, std::chrono::time_point<std::chrono::high_resolution_clock>> startTimes;

    RuntimeMeasure() {}

public:
    static RuntimeMeasure* getInstance();

    void startTimer(int processId);
    double endTimer(int processId);
};

#endif // RUNTIME_MEASURE_HPP
