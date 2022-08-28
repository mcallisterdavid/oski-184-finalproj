# Project "Teapot in a Stadium" 🫖
_Final Project for CS184_

![image](https://user-images.githubusercontent.com/26942890/187052354-97901d64-fa37-45bd-aedf-dc88e36489f5.png)

#### 👇 Watch the Project Video
[![Stadium AR Overview Video](https://img.youtube.com/vi/xeA3LtNYQFo/default.jpg)](https://youtu.be/xeA3LtNYQFo)

#### Project Goal
The goal of our final project was to add AR visuals to stadium performances and football games while learning about computer vision. The vision was for anyone in the surrounding stands to be able to experience rich, motion-tracked 3D visuals to accompany a live event. We very quickly figured out that we had a lot to learn, including implementing GPU compute, developing a landmark detection approach, and making sure it was performant enough to run real-time on mobile hardware.

#### Technical Approach

We began by trying to use Hough Line Transform to identify lines in our frames. This approach bins each detected edge in the image (found via a Sobel filter or otherwise) into each line it could be part of. We were able to represent lines in Hesse normal form and discretized the angle in one degree increments, which meant each pixel could be part of 180 different lines. For each pixel, we had to perform read and write memory operations, which quickly became computationally infeasible. This was too expensive and noisy when we went from tabletop demos to the field demo. Furthermore, we had to run it on the CPU since the algorithm would lead to lots of memory access races when run in parallel. There was no way we would be able to perform these operations on each frame, so we had to rethink our approach.

![image](https://user-images.githubusercontent.com/26942890/187052111-769b5e6e-be03-4104-98de-2596bbc73e03.png)
_Hough transform on tabletop example._

![image](https://user-images.githubusercontent.com/26942890/187052213-247d35e3-6949-4452-a381-cd6be523c7a5.png)
_Hough transform on stadium frame capture (too much entropy in input image for proper detection)_

We noticed that the field had lots of rich color information that we could use for segmentation, so we developed a series of GPU compute functions to pull this out of each image. These were either kernel functions applied to each pixel or each vertical stripe of pixels. The massive parallelization of the GPU allowed us to run a series of these functions over each pixel for every single frame while preserving 60 FPS real-time performance. 

![image](https://user-images.githubusercontent.com/26942890/187052245-bd8bff1c-9757-4a12-97cf-81d95b51b807.png)
_Our green color detection kernel function._

![image](https://user-images.githubusercontent.com/26942890/187052258-1f279fa8-ec8b-452c-9fd9-0f52f5b9e9f1.png)
_Our green-white edge detection filter_

In one such kernel function, we used logistic regression to classify each pixel as green, white, or gold according to its RGB values. Then, we analyzed these filtered pixels’ proximity to each other to identify field lines and make guesses about their locations on the field. 

![image](https://user-images.githubusercontent.com/26942890/187052290-da430e9b-ef62-4d77-a131-e928d82226f2.png)

After some denoising, we got some reasonable outputs. We raycasted from the Cal Logo in different directions to intersect points on the field. This mapped a set of 2D pixel coordinates to their 3D world coordinates. Going from three of these mappings to a transformation matrix is called the perspective-three-point problem, which we solved using a public algorithm. 

The goal of such an algorithm is to calculate the extrinsic matrix, establishing the relationship between the camera and the target object in 3D space. This encodes rotation and translation of the camera from a defined origin point. Then, we can multiply this matrix with an intrinsic matrix, representing the internal parameters of the camera to get the projection matrix. These internal features include known parameters like focal length and can encode simple distortions, though we plan to ignore those and assume a pinhole camera model. 

![image](https://user-images.githubusercontent.com/26942890/187052309-2d4b2ec2-f522-43fd-b721-220b542f4e7f.png)

The perspective-three-point algorithm we chose involves building a quartic polynomial from those mappings and finding its roots. Then, it confirms which root is the correct solution by verifying it with a fourth point. We’ve debugged the process up to the quartic equation solver, so full position-invariant tracking does not work yet. We approximated it by fixing the phone’s starting location and carrying out some more basic trigonometry calculations using our identified points. This solution updates the tracking based on the points each frame, so it’s very subject to noise between captured frames. 

We think we’re very close to a solution that will work from any location in the student section. Our final solution will be much less jittery as well, since we can derive a camera localization from a subset of high quality frame analyses then update the tracking using the gyroscope and accelerometer data. We can further improve this solution by using maximum likelihood estimation to compute the extrinsic parameters that most likely explain data recorded over many frames. 

Finally, we plan to use Apple’s SceneKit to produce more convincing 3D visuals. Currently, they’re simple un-textured meshes with single point light sources. We might be able to estimate the location of the Sun using time of day and compass data then shade to match the scene’s real world light.

![image](https://user-images.githubusercontent.com/26942890/187052335-8f391a41-d5ce-49cb-aace-234f95b86ee7.png)
_Cal Logo feature points superimposed (red triangles)_

![image](https://user-images.githubusercontent.com/26942890/187052346-c9e4a6be-cd8e-4842-9f24-4ce0a71004e1.png)
_Other field feature points superimposed_

![image](https://user-images.githubusercontent.com/26942890/187052354-97901d64-fa37-45bd-aedf-dc88e36489f5.png)
_Teddy bear model tracked onto football field_

![image](https://user-images.githubusercontent.com/26942890/187052371-3fb5644e-7796-45e9-b829-370add189e92.png)
_Utah Teapot tracked onto football field_

##### Watch the video for more visuals!

### References
* Project 3-Point Algorithm: https://www.youtube.com/watch?v=N1aCvzFll6Q
* Project 3-Point Algorithm Paper: http://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-23-p3p.pptx.pdf
* Direct Linear Transform: https://www.cs.toronto.edu/~urtasun/courses/CV/lecture09.pdf
* Camera Calibration: https://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/LeviFranklinCS231AProjectReport.pdf
* Swift Linear Regression Solution: https://aquarchitect.github.io/swift-algorithm-club/Linear%20Regression/
