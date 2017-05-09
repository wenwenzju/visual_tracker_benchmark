Target pedestrian tracking based on particle filter
Version 7.0

What's new? This version is changed based on version 6.0.

This version:
By the purpose of elminating the lack of random walk in fast modtion, first, particles are 
sorted based on their weights; Second, the first n particles are used to calculate the weighted average.

Experiment results show that using weighted average calculated by first n particles is not good. Then treat 
those particles which weight is too big or too small as noise using n-sigma principle. Elminate the noise and 
calculate weighted average of the rest particles.