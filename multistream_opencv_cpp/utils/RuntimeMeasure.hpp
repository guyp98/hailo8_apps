//A singleton class to measure the runtime of different processes of threads

#ifndef RUNTIME_MEASURE_HPP
#define RUNTIME_MEASURE_HPP

#include <chrono>
#include <thread>
#include <mutex>
#include <map>


#define TIMER_1 1
#define TIMER_2 2
#define TIMER_3 3
#define TIMER_4 4
#define TIMER_5 5

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
