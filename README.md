# CMM21-Projects-Overview
This is a brief introduction to the 6 project I have done in the CMM course, given by ETH CRL group, in 21 spring semester.
The most interesting contents of each project is listed below:


* Boids simulation
* Soft body simulation and control
* Rigid body dynamics
* Trajectory Optimization

## Boids simulation
In this task, I simulate the Boids(Bird-oid objects) system with a self written GUI interface. The keypoints in this task are: 
* The realization of the basic kinematics intergration feature, with the consideration of the different intergration methods(Sympletic Euler, Forward Euler, and Explicit Midpoint).
* The realization of different flocking features, including cohesion, alignment, seperation and leader following.

Here is the demo.
![](images/a3-intergrationmethods.gif)
*Figure 1: Intergration methods simulations, using circular motion as an example*

![](images/a3-flocking.gif)
*Figure 2: Flocking features simulations, with a leading bird and considering the influence of the position and velocity of the flocking birds' "Center of Mass". Simulated under basic PI controller*

## Soft body simulation and control
In this task, a elastic bar is simulated considering both forward and inverse problem. By setting the total energy as the target function, and applying different optimizing methods(gradient descent, newtons method), the problem can be solved.

![](images/a4-fem.gif)
*Figure 3: The forward fem simulation*

![](images/a4-manip.gif)
*Figure 4: The inverse manipulation problem. After adding the regularizer term (obj.Question_12), it will process very slowly to the optimal condition.*

It is noticed that at the end of the inverse problem demo, the end of the bar is in some wired position, this can be attributed to the fact of the nonconvexiety of the problem and the property of the optimization method. By adding a regularizer, this problem can be addressed, in spite of slowly processing. The final outcome looks like this.

![](images/a4-final.png)
*Figure 5: The final state of inverse problem*

## Rigid body dynamics
### Basic features 

This task attains the feature of rigid body dynamics simulation. Under these three conditions, the motion of a rigid cube/some rigid cubes with some certain mass and MOI is simulated.
* Projectile
* Single spring
* Double springs and double cubes 

As you can see in the demo below, the simulation of the dynamics and kinematics process is very well.
![](images/a5-demo1.gif)
*Figure 6: Dynamics of the cube under different situations: 1. Projectile, 2. Single Spring, 3. Double Springs*

### Contact simulation
After finishing the basic dynamics simulation, the basics contact simulation is also achieved. By using impluse into integration, setting a criteria of contacting, and setting a restitution coefficient epsilon, the dynamics with contact is simulated. It is noted that the friction is not considered in this task.

![](images/a5-demo2.gif)
*Figure 7: Dynamics simulation with contact under different epsilon of restitution coefficient*

## Trajectory Optimization
In this task, a trajectory optimization problem is attained. Given a specific intial position and velocity, a satellite can precisely program its driving policy and land at the target with or without the influence of an added gravity field. The established policy also works under arbitary initial position and velocity. Apart from that, a target circular motion problem is also achieved. After the optimization, the satellite can converge to the orbit around another planet precisely just by the gravity which it is bearing.

![](images/a6-direction.gif)
*Figure 8: The trajectory optimization problem with and without an added gravity field disturbance. The purple point is the target point and the green ball is the added disturbance planet.

![](images/a6-circular.gif)
*Figure 9: The trajectory optimization problem to let a satellite fly around a planet just by the planet's gravity 