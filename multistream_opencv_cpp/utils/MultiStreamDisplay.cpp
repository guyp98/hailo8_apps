#include "MultiStreamDisplay.hpp"

MultiStreamDisplay::MultiStreamDisplay(int numStreams, int cellSize)
    : numStreams_(numStreams), cellSize_(cellSize) {
    gridRows_ = std::sqrt(numStreams_);
    gridCols_ = gridRows_;
    // Initialize the grid frame
    gridFrame_ = cv::Mat(gridRows_ * cellSize_, gridCols_ * cellSize_, CV_8UC3, cv::Scalar(0, 0, 0));

    // Resize the window
    cv::namedWindow("Multiple Streams", cv::WINDOW_NORMAL);
    cv::resizeWindow("Multiple Streams", gridCols_ * cellSize_, gridRows_ * cellSize_);
}

void MultiStreamDisplay::displayFrames(std::map<int, cv::Mat>& frames) {
    // Check if the number of input frames matches the expected number of streams
    if (frames.size() != numStreams_) {
        std::cout << "Number of frames does not match the number of streams." << std::endl;
        return;
    }
    std::cout << "Displaying frames" << std::endl;
    for (int streamIndex = 0; streamIndex < numStreams_; streamIndex++) {
        cv::Mat frame = frames[streamIndex].clone();  // Clone the input frame to avoid modifying the original
        std::cout << "Before resize" << std::endl;
        // Resize the frame to match the desired cell size
        cv::resize(frame, frame, cv::Size(cellSize_, cellSize_));
        std::cout << "after resize" << std::endl;
        
        int row = streamIndex / gridCols_;
        int col = streamIndex % gridCols_;

        cv::Rect roi(col * cellSize_, row * cellSize_, frame.cols, frame.rows);
        std::cout << "before copy" << std::endl;
        frame.copyTo(gridFrame_(roi));
        std::cout << "after copy" << std::endl;

        // Add watermark with stream number to the frame
        std::string watermarkText = "Stream " + std::to_string(streamIndex);
        cv::putText(gridFrame_, watermarkText, cv::Point(col * cellSize_ + 10, row * cellSize_ + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }
    std::cout << "before imshow" << std::endl;
    cv::imshow("Multiple Streams", gridFrame_);
    std::cout << "after imshow" << std::endl;
}
