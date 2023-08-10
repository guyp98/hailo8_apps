#include "MultiStreamDisplay.hpp"

#include <opencv2/videoio.hpp>

MultiStreamDisplay::MultiStreamDisplay(int numStreams, int cellSize)
    : numStreams_(numStreams), cellSize_(cellSize) {
    gridRows_ = std::ceil(std::sqrt(numStreams));
    gridCols_ = gridRows_;
    // Initialize the grid frame
    gridFrame_ = cv::Mat(gridRows_ * cellSize_, gridCols_ * cellSize_, CV_8UC3, cv::Scalar(0, 0, 0));

    // Resize the window
    // cv::namedWindow("Multiple Streams", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Multiple Streams", gridCols_ * cellSize_, gridRows_ * cellSize_);

    // Initialize the video writer
    videoWriter_.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(gridCols_ * cellSize_, gridRows_ * cellSize_));
    // videoWriter_.open("appsrc is-live=true block=true format=time ! videoconvert ! x264enc ! rtph265pay ! udpsink host=10.0.0.1 port=5000", 0, 30.0, cv::Size(cellSize_, cellSize_), true);
}

void MultiStreamDisplay::displayFrames(cv::Mat frame, int streamIndex) {    
    
    cv::resize(frame, frame, cv::Size(cellSize_, cellSize_));
    
    int row = streamIndex / gridCols_;
    int col = streamIndex % gridCols_;

    cv::Rect roi(col * cellSize_, row * cellSize_, frame.cols, frame.rows);
    frame.copyTo(gridFrame_(roi));

    // Add watermark with stream number to the frame
    std::string watermarkText = "Stream " + std::to_string(streamIndex+1);
    cv::putText(gridFrame_, watermarkText, cv::Point(col * cellSize_ + 10, row * cellSize_ + 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    // // cv::imshow("Multiple Streams", gridFrame_);
    // // cv::waitKey(1);
    // Write the frame to the video file
    
    videoWriter_.write(gridFrame_);
    
}

MultiStreamDisplay::~MultiStreamDisplay() {
    // Release the video writer
    videoWriter_.release();
}
