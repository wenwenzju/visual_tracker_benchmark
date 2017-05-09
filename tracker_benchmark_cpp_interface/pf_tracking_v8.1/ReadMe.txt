Target pedestrian tracking based on particle filter
Version 8.1

What's new? This version is changed based on version 8.0.

This version:
In this version, feature extracting is different.
First, particles are divided into cells;
Second, channel histograms are extracted in every cell.
Third, histograms are vectorized in the order of l0,l1,...,ln,u0,u1,...,un,v0,v1,...,vn,hog0,hog1,...,hogn.