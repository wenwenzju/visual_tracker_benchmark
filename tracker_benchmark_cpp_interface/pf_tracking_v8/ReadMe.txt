Target pedestrian tracking based on particle filter
Version 8.0

What's new? This version is changed based on version 7.0.

This version:
Before v8.0, particles predicted by feature point track are calculated as follows: 
First, detect feature points in the region in which lies these particles of time t-1;
Second, track these detected feature points in the image of time t;
Third, calculate the average velocity of the tracked feature points and predict every 
particles using this velocity and add noise.
In this version, particles predicted by feature point track are calculated as before v4.0, 
e.g. as follows:
First, detect feature points in the rectange of time t-1 detected result;
Second, track these detected feature points in the image of time t;
Third, particles in time t are generated obeying gaussian distribution.