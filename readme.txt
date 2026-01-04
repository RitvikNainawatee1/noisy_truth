Core Components of UKF Class

The UKF Class is a core component. The UKF Class is, in the ukf_v2.py file.

The UKF Class implements the core prediction and update steps of the UKF filter.

The UKF filter has these prediction and update steps.

These steps are important for the UKF filter to work correctly. Here are the key methods that this filter uses.

The first method is init. This method does a lot of things. It starts the filter with some information. This information includes the state and covariance and noise parameters. It also includes the process model and the measurement model. The process model is also called iterating_func. The measurement model is also called measure_func. This method also calculates the weights for the Sigma Points. These weights are called W_m and W_c.

There are some parameters that we can adjust to get the results we want. These parameters are α, β and k. These parameters control how we choose the sigma points and how weight we give to each one. We use these parameters to calculate something called λ. The formula for λ is λ = α²(N_x + k) − N_x.

The method is predict. This method takes the state and covariance. Moves them forward in time. It is like predicting what will happen next.

The filter has these methods to help it work properly.

The init method is very important.

It does a lot of work to get the filter started.

The predict method is also very important.

It helps the filter make predictions, about what will happen

The filter uses these methods to make predictions. The Key Methods are used to make the filter work. The filter uses the init method and the predict method to do its job. The filter is very useful. It helps us make predictions. The Key Methods init. Predict are very important. They help the filter work. This is about creating a bunch of sigma points, which's 2N_x + 1 of them. We take these sigma points. Put them through a function called f also known as ukf_process. Then we combine them again using a sum, which is called the Unscented Transform. This helps us find the state of the system, which is x⁻ and the covariance, which is P⁻.

The update function, update(z, R) is used when we get a new measurement, which is z. The sigma points are put through a function that's not linear which is the measurement function to figure out what the measurement is going to be called z⁻.

Then the Kalman Gain, which is K and the residual, which is y and the system uncertainty, which is S are all calculated.

This is done to correct the state, which's x and the covariance, which is P.

The sigma points and the measurement function and the Kalman Gain and the residual and the system uncertainty and the state and the covariance are all important, for this.

Simulation Script, the file is called ukf_testing.py.

This file is used to set up a simulation. It defines the process and the measurement models.

It also gives us a way to interact with the filter parameters so we can tune them. Here is how the system works.

The true process is what really happens in the system that we are trying to understand. This is the true_process function. It takes into account the state of the system and the random noise that affects it.

We also have the function. This is the model that we use to predict what the system will do. It is based on the physics of the system. It does not include the random noise.

The measurement_model function tells us what we are measuring. In this case we are only measuring the position. This is the state variable.

For example the measurement_model function is defined as h(x).

We have a main functions:

true_process(x, Q_true): This is the physical system.

Ukf_process(x): This is the model of the system.

Measurement_model(x): This defines what we are measuring.

These functions are important because they help us understand the system. The true_process function shows us what really happens. The ukf_process function shows us what we think will happen. The measurement_model function shows us what we can measure.

The true_process function includes the process noise. The ukf_process function does not include the process noise. The process noise is accounted for separately.

We use the true_process function and the ukf_process function to understand the system. We use the measurement_model function to measure the system.

The system is defined by these functions:

True_process(x, Q_true)

Ukf_process(x)

Measurement_model(x)

These functions are important. They help us understand the system and make predictions, about what it will do. run_sim(...): Executes the simulation loop, calling ukf.predict() and ukf.update() at each time step.

The Dynamic Model- The system simulates the motion of an object where the acceleration is due to air drag, which is proportional to the square of the velocity (for high speeds):
p¨ = −d · v |v|.

The state update equations used by the UKF (ukf_process) are derived using the Euler method over a time step Δt:

x_{k+1} = f(x_k) =

[ p_{k+1}
v_{k+1}
d_{k+1} ]

=

[ p_k + v_k · Δt
v_k − d_k · v_k |v_k| · Δt
d_k ]

The process noise Q, which is set using the sliders Q_p, Q_v, Q_d is really important. This is because the process noise Q accounts for things that're not certain in the process model. It also accounts for disturbances that are not explicitly modeled such as wind or inaccuracies, in the drag law.

To get started with the simulation there are a things you need to do.

You need to have Python installed on your computer.

You also need to have some libraries installed.

You can install these libraries using Bash.

You will need to install numpy.

You will also need to install scipy and matplotlib.

You can do this by running the following command:

pip install numpy scipy matplotlib

Running the Simulation
Execute the testing script:

python ukf_testing.py

When you run this a plot will open. This plot shows you the True, Estimated and Measured values for position, velocity and drag coefficient over time. You can see how these values change as time goes on.

You can use the sliders to change some settings and see how they affect the filter. The filter is like a tool that helps us figure out what is really going on. You can adjust the noise parameters. There are a things you can change:

Q position

Q velocity

Q drag. These are like controls that tell us how sure we are about what's happening. If you set these to values it means we are not very sure about what is going on with the process model.

Then there is R meas. This is, like a setting that tells us how much we trust the measurements that are coming in. If you set this to a value it means we do not trust the measurements very much.

P0: The scale of the Initial Covariance matrix P_0. Higher values represent greater initial uncertainty in the state estimate.

Click "generate graphs" to rerun the simulation with the new parameter settings.