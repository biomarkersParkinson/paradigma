# Coordinate System
As a prerequisite to reliably estimating gait and arm swing measures, it is important to align the coordinate system of the IMU sensor with the coordinate system of the trained classifiers. For the tremor and heart rate pipelines, differences between coordinate systems do not affect outcomes. 

## Coordinate system used
The coordinate system of the IMU sensor used for training the classifiers can be observed below. The direction of acceleration is indicated by arrows, and the direction of gyroscope rotation can be determined using [Ampère's right-hand grip rule](https://en.wikipedia.org/wiki/Right-hand_rule#Amp%C3%A8re's_right-hand_grip_rule) applied to the accelerometer axes. 

<p align="center">
  <img src="https://raw.githubusercontent.com/biomarkersParkinson/paradigma/main/docs/source/_static/img/directions_axes.png" alt="Coordinate system"/>
</p>

### Accelerometer
The three accelerometer axes are set such that the x-axis is aligned with the arm pointing toward the hand, the y-axis is perpendicular to the arm pointing upward from the top of the sensor, and the z-axis points away from the arm and body. If the arrow representing a specific axis is pointing downward to the ground, parallel to and in the direction of the arrow representing gravity (1G), the acceleration of this specific axis is equal to -1g if the sensor is stable (i.e., no acceleration due to movement). 

### Gyroscope
If the sensor is rotating in the direction of the arrow (deducted using the [Ampère's right-hand grip rule](https://en.wikipedia.org/wiki/Right-hand_rule#Amp%C3%A8re's_right-hand_grip_rule) applied to the accelerometer axes), the gyroscope data will be positive.

## Lateral differences
Wearing the watch on the left wrist or right wrist influences the relation between movements and the coordinate system. In fact, the x-axis of the accelerometer, and the y-axis and z-axis of the gyroscope, are inverted. For this purpose, we have added `invert_watch_side` to the toolbox (which can be imported using `from paradigma.util import invert_watch_side`). First, ensure the coordinate system aligns with the coordinate system shown above. Do this for all participants wearing the watch on one specific side, for example the left wrist. Then, apply `invert_watch_side` to ensure individuals wearing the watch on the right wrist have the correct coordinate system accounting for differences in sensor orientation.