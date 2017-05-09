Target pedestrian tracking based on particle filter
Version 4.0

What's new? This version is a big change
No self-adaptive weight in observation model. Self-adative sigma of particles.
No using of new template model, that is the linear combination of template of each particle.

The gaussian distribution of Sdalf's mapknl is changed to uniform distribution.
Add head map and head features in sdalf.

Particle is updated from particle respectively, not from the weighted mean.
Particles is consist of two portion. One is propagated using feature point tracking, the other 
is propagated based on random walk.
Add velocity at direction x and y to particle's state.
Some tricks:
1.particle's width is constrained to be bigger than a small value like 10 and and 
particle's height is constrained to be smaller to image's height;
2.particles propagated using feature point tracking are added noise as well. Noise here obeys normal distribution;
3.when particles propagated using random walk are near the border, the noise added is more likely
  to make the particle's velocity direct to the center of the image. Noise here obeys uniform distribution;
4.when particles propagated using random walk are near the center, the noise added is more likely 
  to make the particle's velocity equal to the estimated target velocity. Noise here obeys uniform distribution;