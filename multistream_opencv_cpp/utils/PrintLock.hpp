#ifndef PRINTLOCK_HPP
#define PRINTLOCK_HPP

#include <mutex>
#include <iostream>

class PrintLock {
public:
    static PrintLock& getInstance();
    void print(const std::string& message);

private:
    PrintLock();
    PrintLock(const PrintLock&) = delete;
    PrintLock& operator=(const PrintLock&) = delete;
    static std::mutex m_mutex;
};

#endif // PRINTLOCK_HPP