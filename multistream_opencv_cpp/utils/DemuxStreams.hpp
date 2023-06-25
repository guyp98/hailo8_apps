#ifndef DEMUXSTREAMS_HPP
#define DEMUXSTREAMS_HPP

#include "MultiStreamDisplay.hpp"
#include "SynchronizedQueue.hpp"
#include <memory>
#include <vector>

class DemuxStreams {
private:
    MultiStreamDisplay display;
    std::vector<std::shared_ptr<SynchronizedQueue>> frameQueues;

public:
    DemuxStreams(const int numStreams, const std::vector<std::shared_ptr<SynchronizedQueue>>& syncQueue);
    void readAndDisplayStreams();
};

#endif  // DEMUXSTREAMS_HPP
