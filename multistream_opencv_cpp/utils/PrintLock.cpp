#include "PrintLock.hpp"

std::mutex PrintLock::m_mutex;

PrintLock::PrintLock() {}

PrintLock& PrintLock::getInstance() {
    static PrintLock instance;
    return instance;
}

void PrintLock::print(const std::string& message) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cout << message << std::flush;
}