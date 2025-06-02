import matplotlib.pyplot as plt

data = {
    #  'cartpole_delay': {
    #      r'$\mathrm{TD}-\mathrm{MPC2}$': {
    #         0: (863.9, 0.00),
    #          3: (802.3, 43.2),
    #          6: ( 797.51,  56.84),
    #          10: (659.42, 164.50),
    #      },
    #      '$\mathrm{LTC}_\mathrm{det}^\mathrm{ZP}$': {
    #         0: (822.04, 0.90),
    #         3: (811.71, 3.83),
    #         6: (790.87, 73.87),
    #         10: (730.36, 97.89),
    #      },
    # },
    'walker_delay': {
        r'$\mathrm{TD}-\mathrm{MPC2}$': {
            0: (863.9, 0.00),
            1: (779.37, 11.41),
            2: (567.43, 95.23),
            3: (231.78, 38.29),
        },
        '$\mathrm{LTC}_\mathrm{det}^\mathrm{ZP}$': {
            0: (822.04 , 0.90),
            1: (763.19, 69.46),
            2: (761.46, 74.66),
            3: (686.53, 22.26),
        }
    },
}

# ---- 1) define the data ----

titles = {
    # 'cartpole_perturb':         'Cartpole-Swingup (perturbed)',
    'cartpole_delay': 'Cartpole-Swingup (stochastic observation delay)',
    'walker_delay': 'Walker-Run (stochastic observation delay)',
}

# Tell matplotlib to use LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 24})  # Set global font size

# ---- 2) plotting loop ----
for env, models in data.items():
    plt.figure(figsize=(12,8))
    for model_name, vals in models.items():
        # sort by sigma
        sigmas = sorted(vals.keys())
        means  = [ vals[s][0] for s in sigmas ]
        stds   = [ vals[s][1] for s in sigmas ]
        # plot mean line
        line, = plt.plot(sigmas, means, marker='o', label=model_name)
        # shaded ±1 std band
        plt.fill_between(
            sigmas,
            [m - sd for m, sd in zip(means, stds)],
            [m + sd for m, sd in zip(means, stds)],
            alpha=0.3,
            color=line.get_color(),
        )

    plt.title(titles.get(env, env))
    plt.xlabel(r'Maximum value for discrete random observation delay in environment timesteps')
    plt.ylabel(r'Mean \& std. episodic return')
    plt.xticks([0.0, 1, 2, 3])
    plt.legend()
    plt.tight_layout()

# ---- 3) show all plots ----
plt.show()