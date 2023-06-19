# Hailo Windows Apps

This project is built on Visual Studio C++ compiler on Windows.

## Build Environment

1. Install CMake version 3.23.0 and up.
2. Install the latest `hailort.msi` from the Hailo website.
3. Install the latest `C++ opencv.msi` for Windows.
4. Make sure that OpenCV and Hailo Runtime are added to the Windows PATH.
5. Clone the project.
6. Select the desired app (For example, let's choose "yolact" for demonstration purposes).
7. From the "yolact\" directory, run the following commands:
    ```
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
    .\Release\yolact_app.exe
    ```
8. To view available options, use the command:
    ```
    .\Release\yolact_app.exe -h
    ```

## Common Pitfalls

* If the project path is too long, you may encounter an error. It is recommended to keep your path short.
* Running the project from a path with spaces in its name can sometimes cause unexpected errors.
