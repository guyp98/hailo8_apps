#include "MultiStreamDisplay.hpp"

MultiStreamDisplay::MultiStreamDisplay(int numStreams, int cellSize)
    : numStreams_(numStreams), cellSize_(cellSize) {
    gridRows_ = std::ceil(std::sqrt(numStreams));
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
    for (int streamIndex = 0; streamIndex < numStreams_; streamIndex++) {
        cv::Mat frame = frames[streamIndex];  // Clone the input frame to avoid modifying the original
        
        cv::resize(frame, frame, cv::Size(cellSize_, cellSize_));
        
        int row = streamIndex / gridCols_;
        int col = streamIndex % gridCols_;

        cv::Rect roi(col * cellSize_, row * cellSize_, frame.cols, frame.rows);
        frame.copyTo(gridFrame_(roi));


        // Add watermark with stream number to the frame
        std::string watermarkText = "Stream " + std::to_string(streamIndex);
        cv::putText(gridFrame_, watermarkText, cv::Point(col * cellSize_ + 10, row * cellSize_ + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }
    cv::imshow("Multiple Streams", gridFrame_);
    cv::waitKey(1);
}
