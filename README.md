# Self-driving-car-simulation-cnn

In this project I train a self driving car using a Convolutional neural network. I used the open source self driving simulator provided by [Udacity](https://github.com/udacity/self-driving-car-sim)  to collect data and then train a CNN model to learn the driving behaviour and test it back on the simulator. The CNN model used has been proposed by [Nvidia](https://arxiv.org/pdf/1604.07316v1.pdf), they use this model to train a real car with real data and got promising results.

The idea and inspiration for the project came from youtuber [Siraj Raval](https://www.youtube.com/watch?v=EaY5QiZwSP4).

Once data is collected using the simulator you get two files, a log file and a folder with a lot of images, located in mySimData folder. There are three cameras on the car so you get the center, left and right camera image. Not all the images have been uploaded due to Githubs file size limit. In the driving log file you have all the data that includes paths of the center, left, and right images, the steering angle, throttle, brake, and speed. The center images and steering angle are used to train the model, that will then generalize how to drive.

### Packages/Dependencies
packages.txt file

### Balancing and visualizing the data

The data was balanced and visualized to ensure there are equal amounts of information because if there are a lot more angles for the left curve compared to the right curve for instance, then your model will generalize to mostly go on the left hand side. So you want to make sure training process is not biased. There will be a lot more data for '0' than compared to left or right as evident on the driving log, since you are driving straight most of the time on the track.

![Image of bal data](https://dsm01pap002files.storage.live.com/y4mVFIlf_-BVTMBbFSstwwBSsMqrXrWB9Vm3tdYg88ZWIzNQ2JJFuWcFZBsYLOpSyNqjsEzW-MkmhZWA0uuHTa8em0IoKf_44eeJYYfyKL2B33Rh68UiiMumJMqOSnRqmwgsmJRvCg3UJ_-v6l0hqY0TeaRB6Rhox-Ax1QEc589wZzVhBLwg03vNwBdJZT-59rt?width=256&height=192&cropmode=none)
