# Implementation Details

The equations given by Thygesen et al. (2009) are relatively
straight-forward to implement. Most complications stem from the fact
that we cannot store the probability maps for every time-step due to
memory constrains (in particular on the GPU).

The figure below illustrates how the caching works on the high-level.

![compute flow](assets/compute_flow.png)
