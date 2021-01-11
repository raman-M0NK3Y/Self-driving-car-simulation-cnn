# Self-driving-car-simulation-cnn

In this project I train a self driving car using a Convolutional neural network. I used the open source self driving simulator provided by [Udacity](https://github.com/udacity/self-driving-car-sim)  to collect data and then train a CNN model to learn the driving behaviour and test it back on the simulator. The CNN model used has been proposed by [Nvidia](https://arxiv.org/pdf/1604.07316v1.pdf), they use this model to train a real car with real data and got promising results.

The idea and inspiration for the project came from youtuber [Siraj Raval](https://www.youtube.com/watch?v=EaY5QiZwSP4).

Once data is collected using the simulator you get two files, a log file and a folder with a lot of images, located in mySimData folder. There are three cameras on the car so you get the center, left and right camera image. Not all the images have been uploaded due to Githubs file size limit. In the driving log file you have all the data that includes paths of the center, left, and right images, the steering angle, throttle, brake, and speed. I used the center images and steering angle to train the model, that will then generalize how to drive.

**Video** : https://www.dailymotion.com/video/x7ykd7x?queue-enable=false

### Packages/Dependencies
packages.txt file

### Balancing and visualizing the data

The data was balanced and visualized to ensure there are equal amounts of information because if there are a lot more angles for the left curve compared to the right curve for instance, then your model will generalize to mostly go on the left hand side. So you want to make sure training process is not biased. There will be a lot more data for '0' than compared to left or right as evident on the driving log, since you are driving straight most of the time on the track.

![Image of bal data](https://dsm01pap002files.storage.live.com/y4mVFIlf_-BVTMBbFSstwwBSsMqrXrWB9Vm3tdYg88ZWIzNQ2JJFuWcFZBsYLOpSyNqjsEzW-MkmhZWA0uuHTa8em0IoKf_44eeJYYfyKL2B33Rh68UiiMumJMqOSnRqmwgsmJRvCg3UJ_-v6l0hqY0TeaRB6Rhox-Ax1QEc589wZzVhBLwg03vNwBdJZT-59rt?width=640&height=480&cropmode=none)
![img of redun bal data](https://dsm01pap002files.storage.live.com/y4m48wSWL402gKCLUz2x9tUKd-ZCHrSctsB4KFSq2CqzICloV-w5wwcqhR24_1OcFDxY7bkBMZNpNUgiBFuDYptl6pkCP9lbg3bFuocEQwDxeWPLp4Dt_LbOVDntyPonvUpBDVL_zG9rv1kV3MMSkuR3_rhye8oKipOnutLPajtq9OWDmi-Ft0w9XZ0ml3AR2Gq?width=640&height=480&cropmode=none)

### Data/Image Augmentation

Data is augmented using a variety of techniques to add more variety to help model generalize more efficiently.

**original**

![orginal](https://dsm01pap002files.storage.live.com/y4m4ONOpak3mzPHYwOBpS-GaJX8G8NCBcpBYrZrC2mqY25HIJX9rGP5b_y-R4DxQn04kwjOXuZcT6qB1rUZ224h5Vu_DvYIXUeiIGlyK9ptfVMDYP2Rq1GP5OuFPCL3JxLGD7AFjGuhWoGx3PREPcDCD1d4kt7ULkGzJPRF7dhlEVGs_fAODdt6A0Atw01zxHTC?width=320&height=160&cropmode=none)

**panning**

![panning](https://dsm01pap002files.storage.live.com/y4mQRU5IikIBFhGb6iQ1zoiMZ3ddBSfGm6jnB23kJJPuakHkjZHN9eQwrMRjlcqvlQgkbzVsVxEcheX_be_2DsBdZlJFv-qvuA67ioZJJPpgyZQKY8m28_zgWQxG0YDUWLHkrOkvE6OJqCHun8jllNK5O6qjif6y7atbP0y6fxkF-3H9goXS-Bvs5DIlghh3CgS?width=640&height=480&cropmode=none)

**zoom**

![zoom](https://dsm01pap002files.storage.live.com/y4mhf8pO_uO5Hd5z4pRz_dxZNAxVQ1WmBLC---5Hi0YZpy7X3F-gDVWT3X5Lz0gN3qq-SH7eY0aNTjokBB-SneNPPKnmeCceRuMlj5okCdjl1T-S9INtgmD7BMr7dyOcpGGUuFbBZOY4FjJ_eRY9gBHRpWl0rqvxisDOFz1R-VQvPdX-Am5O3YKLUiM0OKcfokV?width=640&height=480&cropmode=none)

**brightness**

![bright](https://dsm01pap002files.storage.live.com/y4miZEgjnyfjigPFWrokfJXPVcUnhi-EjTs4u3X_XYGFaj9eVU_YU0Vzf0qjGL3nZ_D_n5H_yC3YRMp1quEednpOnE7VlW8ED34xk-Rdefmf6MjKTT1RvdFEk4mKAXSPCFyWFqmU-bF6TfMgU376qYbFdeju33f5WjzUjAOXxIIpL2xhv5APcZLz83NXBXGOFkK?width=640&height=480&cropmode=none)

**flip**

![flip](https://dsm01pap002files.storage.live.com/y4mAAKMkLVOfUSHV_fiRJhorI7V7zEdVL9XcUJ35lVIL-lsd4puLhA98HjmprdxzgIiExkcZ4hdPKGacs6hjZ4swLl4Il08pIhn_U039zeE2F1knmqPPEVKvZnxU0RZL_dErwUtKyr2p9CuWkIEdFOLN2AdfsSgalrRCGjORAlMGDvUWahpTL888Oj0czUWqnzP?width=640&height=480&cropmode=none)

### Data Preprocessing/Generator

Images are cropped to remove any unnecessary data (sky, trees, front part of car). Images are also resized, color space is changed to YUV and normalized as proposed by Nvidia. This helps to better define lane lines and general path. Images are sent to training model in batches to help stabilize learning process.

![crop](https://dsm01pap002files.storage.live.com/y4miADtza2yDtti5GlJbnYlvnll3AnZC4l6O0fSspk_aN1sqgMn8GFu9DWmQ-W6OwAZkhevZ60WQTUFy3C9YLE415uL3Qwu7-d3eIFjH2dyonG47C-zFXKmqP5_b-lgkN-LfiUtR9cg7ac9sUlmmufVz5tteF5mJCilLxuvDdrp5oFFS6dBktqp0mZOLs2LcpOf?width=640&height=480&cropmode=none)

![yuv](https://dsm01pap002files.storage.live.com/y4mdXre-kODGzl3ArwNeUEmgJnQ-QOCqqxj9Rife-hFJ_8vGFcHY6rRI9VMoneYHbJx-KMHt5T_oeRKw_c46axvxIweJC6rqCPqw85iFVljDjfRe_EFNgI9DCFV7abDVgINiFjxCItASrQ1nJ37yvqiOKF76Xe7qzA2tq4uWZhnsZ9Mv-q4p16r8fHZFAME0QQ-?width=640&height=480&cropmode=none)

### Model

The architecture of the model is based on the ![Nvidia Model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) which they have tested on a real car. It is a CNN which works well tackling supervised regression problems such as this. End up with total parameters of 250 thousand which is what Nvidia has shown.

![model](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

### Simulation Result

80% of data for training, 20% for validation. Trained the model on 45,000 images per epoch (10 times), and everytime it trains will generate batch of 150 images and validate 200 steps. Ran on GPU -gtx 1660.

**Video** : https://www.dailymotion.com/video/x7ykd7xqueue-enable=false

Now from the video you can see that the model is able to generalize enough to drive around the track succesfully without going out of bounds. Now the car is touching the lanes quite often when making turns but the car does not go out of bounds and is still able to go around the entire track. Considering that the model has only been trained on a few thousand images for couple of minutes compared to the 72 hours of training data Nvidia collected for their model it performs quite well. Also, to improve the model can make a lot of small changes to different parameters such as getting more data, training for longer, adding more flexibility in terms of augmentation, etc. 

### References

-udacity simulator: https://github.com/udacity/self-driving-car-sim

-nvidia model: https://developer.nvidia.com/blog/deep-learning-self-driving-cars/ (full paper linked in blog post)

-idea for project: https://www.youtube.com/watch?v=EaY5QiZwSP4

