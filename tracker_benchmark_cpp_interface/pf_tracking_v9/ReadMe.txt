Target pedestrian tracking based on particle filter
Version 9.0

What's new? This version is changed based on version 8.0.

This version:
In this version, the ratio of random walk particles is self-adaptive.
Fisrt, detect feature points of frame at time t-1 in the region of tracking result;
Second, track feature points at time t and calculate the displacement of every tracked 
points. The average displacement is used to judge whether the target is in fast motion.
If so, the ratio of random walk is decreased so that feature point track is in major.
Otherwise, the ratio of random walk is set to the initialization.