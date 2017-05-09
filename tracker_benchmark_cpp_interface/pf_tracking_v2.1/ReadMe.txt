Target pedestrian tracking based on particle filter
Version 2.1

No self-adaptive weight in observation model.
Add self-adaptive sigma of x, y, and w. Sigma of x, y, and w is adapted based on size of targets.
The gaussian distribution of Sdalf's mapknl is changed to uniform distribution.
Add head map and head features in sdalf.

What's new?
Based on version 2.0, the distribution of particles has been changed. Before is normal distribution, 
wherever particle is. Now when particle is near the border, the distribution is changed to exponential 
distribution.