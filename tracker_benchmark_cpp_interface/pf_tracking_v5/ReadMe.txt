Target pedestrian tracking based on particle filter
Version 5.0

What's new? This version is changed based on version 4.0
Templates used before are consisted of two portion. One is the template at time t0,
while the other is the weighted average of t0 template and t-1 template. Candidates at 
time t are matched with the two templates and the minimum distance is returned.

In this new version, templates used are from time t-k-1 to time t-1. There are two 
optional ways to compute distance between candidates and templates. One computes the 
weighted--the newer template the bigger weight-- average of the k+1 templates, together with t0 template and matches
candidates with the weighted average template and t0 template and returns the minimum 
distance. The other computes the weighted average of distances between candidates and each 
k+1 templates, again the newer template the bigger weight.

Version 5.0 is a generalized version of version 4.0. Let k equal to 0 we get a same version as 4.0.