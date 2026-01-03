import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from ukf_v2 import UKF

dt = 0.05
steps = 300

true_drag = 0.15

def true_process(x, Q_true):
    p, v, d = x

    p = p + v * dt + np.random.normal(0, np.sqrt(Q_true[0]))
    v = v - d * v * abs(v) * dt + np.random.normal(0, np.sqrt(Q_true[1]))
    d = d + np.random.normal(0, np.sqrt(Q_true[2]))

    return np.array([p, v, d])

def ukf_process(x):
    p, v, d = x
    p = p + v * dt
    v = v - d * v * abs(v) * dt
    return np.array([p, v, d])

def measurement_model(x):
    return np.array([x[0]])

def run_sim(Qp, Qv, Qd, R_scale, P0_scale):
    x_true = np.array([10.0, -3.0, true_drag])

    Q_true = np.array([Qp, Qv, Qd])

    ukf = UKF(
        initial_state=np.array([8.0, -1.0, 0.05]),
        n_dim_z=1,
        measure_func=measurement_model,
        iterating_func=ukf_process,
        initial_covar=np.eye(3) * P0_scale,
        process_noise=np.diag([Qp, Qv, Qd]),
        alpha=0.6,
        beta=2.0,
        k=0.0
    )

    R = np.array([[R_scale]])

    true_hist, est_hist, meas_hist = [], [], []

    for i in range(steps):
        x_true = true_process(x_true, Q_true)
        z = measurement_model(x_true) + np.random.normal(0, np.sqrt(R_scale))

        ukf.predict()
        ukf.update(z, R)

        true_hist.append(x_true.copy())
        est_hist.append(ukf.get_state())
        meas_hist.append(z)

    return (
        np.array(true_hist),
        np.array(est_hist),
        np.array(meas_hist)
    )

fig, axs = plt.subplots(3, 1, figsize=(10, 9))
plt.subplots_adjust(bottom=0.45)

ax_Qp = plt.axes([0.1, 0.35, 0.8, 0.03])
ax_Qv = plt.axes([0.1, 0.30, 0.8, 0.03])
ax_Qd = plt.axes([0.1, 0.25, 0.8, 0.03])
ax_R  = plt.axes([0.1, 0.20, 0.8, 0.03])
ax_P0 = plt.axes([0.1, 0.15, 0.8, 0.03])

s_Qp = Slider(ax_Qp, "Q position", 1e-6, 1e-1, valinit=1e-3)
s_Qv = Slider(ax_Qv, "Q velocity", 1e-6, 1e-1, valinit=1e-3)
s_Qd = Slider(ax_Qd, "Q drag",     1e-6, 1e-3, valinit=1e-4)
s_R  = Slider(ax_R,  "R meas",     1e-4, 1.0, valinit=0.05)
s_P0 = Slider(ax_P0, "P0",         0.01, 10.0, valinit=1.0)

ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
button = Button(ax_button, "generate graphs")

def update(event):
    true, est, meas = run_sim(
        s_Qp.val,
        s_Qv.val,
        s_Qd.val,
        s_R.val,
        s_P0.val
    )

    t = np.arange(len(true))

    for ax in axs:
        ax.cla()

    axs[0].plot(t, true[:,0], label="True")
    axs[0].plot(t, est[:,0], "--", label="UKF")
    axs[0].plot(t, meas[:,0], ".", alpha=0.4, label="Measured")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(t, true[:,1], label="True")
    axs[1].plot(t, est[:,1], "--", label="UKF")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(t, true[:,2], label="True")
    axs[2].plot(t, est[:,2], "--", label="UKF")
    axs[2].set_ylabel("Drag Coefficient")
    axs[2].set_xlabel("Time step")

    fig.canvas.draw_idle()

button.on_clicked(update)
plt.show()
