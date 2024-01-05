# Location prediction using density estimation

This is a small toy project to predict the paths of moving objects, e.g. marine vessels. Its main purpose is as a CV-padder, so it is relatively simple, but if it contains any code which you find useful, feel free to use it with reference. This section of the README acts as a write-up and rationale. The [Package Structure](./README.md#Package-Structure) section outlines the package structure and use. If you're busy and would rather just dive in with an example, check out the [Track and Visualise Notebook](TrackAndVisualise.ipynb).

## Theoretical considerations
### Trajectories or not

Strictly speaking, this project does not predict trajectories, but it predicts future locations at a set time, i.e. the end-point of the trajectory up to that moment. It could be altered to predict trajectories by outputting a vector of locations, or it could be altered to be conditional on time, so that it can make predictions of the end-point at any time step (hence the trajectory). However, for many purposes this implementation is sufficient; if we are tracking an object using observations at fixed time intervals then information of what is happening *between* those intervals is extraneous.

### Isn't this just object tracking?

No. Object tracking is about consistent identification of an object over a set of frames.

Here I assume that a method for object-identification already exits, but that the object is rarely observed.

### Does this relate to Kalman filtering?
While Kalman filtering is used in similar situations, it is not quite the use-case I'm considering (although I wouldn't discount it entirely). Given a model, Kalman filtering reconciles the noisy dynamics of the model with the noisy observations of its outcomes. Here I'm assuming noiseless observations, and using flexible normalising flow-based models.

### Stochastic processes need stochastic predictions
For a stochastic process, even given the exact same input conditions the outcome is not guaranteed. This is sometimes considered to be because of latent unobserved variables, but if you ask a quantum physicist, it is a fundamental property of performing observations.

In cases where certainty is important, it 

## The simulator
We need a simulator to generate fake data. The simulator is not a critical part of the application - we just need some data from somewhere. I've written a simulator with a toy model of a moving sea vessel. See the below image for 100 simulated trajectories from this simulator with the default settings.

## Density estimators
Given a set of observations are drawn from some underlying probability distribution

Normalising flows allow us to learn, evaluate, and sample from probability densities. In applications
where certainty important, but not achievable, then being able to quantify certainty is very valuable.

### Conditioning
It is not difficult to extend the model to condition upon informative data such as the ship bearing or velocity. However, given that I initialise all ships under the same conditions in the simulator I've written, this is not productive for this project in its current state. Some examples of interesting real-world conditioners:

1. GPS location
2. Depth-gradient
3. Bearing of nearest port
4. Vessel id
5. Weather
6. Currents


# Package Structure
There are four main files

### simulator.py
This includes a simulator for mock trajectories of, say, a marine vessel. The trajectories are modelled as concatenated arc segments, where the angle of the arc-segment and the speed along which an object traverses it are stochastic.

### training.py
This includes the main training code, as well as a datloader utility function. It relies on flow.py, in which I have simply imported an existing architecture for normalising flows from the sbi package, with some very minor adaptation.

### tracking.py

### visualisation.py
This is currently a bit more *under construction*, but contains some useful visualisation utility functions. 
