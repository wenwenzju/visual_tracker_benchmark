Target pedestrian tracking based on particle filter
Version 12.0

What's new? This version is changed based on version 10.0.

This version:
This a universal version, e.g. an visual object tracker, not just for pedestrian tracking.

1.Add height in particle's state. For pedestrian tracking, particle's height is proportional 
to particle's width, so 5 states(x,y,width,vx,vy) are needed. But in this version, for general 
visual object tracking, 6 states(x,y,width,height,vx,vy) are needed.
2.Disuse pedestrian detector in observation model.