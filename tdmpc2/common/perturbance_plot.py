import matplotlib.pyplot as plt

# ---- 1) define the data ----
data = {
    'cartpole_perturb_flicker': {
        r'$\mathrm{TD}-\mathrm{MPC2}$': {
            0.0: (863.9, 0.00),
            0.1: (378.82, 45.43),
            0.2: (353.63, 76.27),
            0.3: (365.71, 47.37),
        },
        r'$\mathrm{CfC}_\mathrm{det}^\mathrm{ZP}$': {
            0.0: (856.0, 0.0),
            0.1: (798.89, 50.64),
            0.2: (742.19, 52.02),
            0.3: (550.69, 76.55),
        },
        r'$\mathrm{CfC}_\mathrm{det}^\mathrm{OP}$': {
            0.0: (880.0, 0.0),
            0.1: (878.0, 0.0),
            0.2: (756.0, 70.0),
            0.3: (645.49, 92.07)
        },
        r'$\mathrm{CfC}_\mathrm{stoch}^\mathrm{OP}$': {
            0.0: (856.0, 7.4),
            0.1: (825.0, 25.45),
            0.2: (693.08, 54.02),
            0.3: (600.22, 79.86),
        },
        # 'LTC_OP_det': {
        #     0.0: (860.0, 9.04),
        #     0.1: (824.35, 16.02),
        #     0.2: (744.14, 36.91),
        #     0.3: (471.47, 104.39)
        # }
    },
    'acrobot_perturb': {
         r'$\mathrm{TD}-\mathrm{MPC2}$': {
            0.0: (329.0, 0.0),
            0.1: (90.50, 104.67),
            0.2: (57.74, 55.10),
            0.3: (74.80, 41.12),
        },
        r'$\mathrm{CfC}_\mathrm{det}^\mathrm{ZP}$': {
            0.0: (829.0, 0.0),
            0.1: (559.78, 28.61),
            0.2: (448.90, 75.03),
            0.3: (322.62, 113.33),
        },
        r'$\mathrm{CfC}_\mathrm{det}^\mathrm{OP}$': {
            0.0: (544.7, 6.73),
            0.1: (564.55, 35.14),
            0.2: (495.90, 48.82),
            0.3: (374.15, 47.69)
        },
        r'$\mathrm{CfC}_\mathrm{stoch}^\mathrm{OP}$': {
            0.0: (263.26, 61.96),
            0.1: (272.56, 93.26),
            0.2: (285.83, 56.31),
            0.3: (239.59, 161.22),
        }
    },
    'walker_run_perturb': {
          r'$\mathrm{TD}-\mathrm{MPC2}$': {
            0.0: (830.9, 0.0),
            0.1: (581.92, 37.27),
            0.2: (495.75, 38.88),
            0.3: (387.69, 38.61),
        },
        r'$\mathrm{CfC}_\mathrm{det}^\mathrm{ZP}$': {
            0.0: (785.98, 16.14),
            0.1: (764.30, 15.06),
            0.2: (734.35, 22.53),
            0.3: (647.65, 38.56),
        },
        r'$\mathrm{CfC}_\mathrm{det}^\mathrm{OP}$': {
            0.0: (632.7, 16.14),
            0.1: (632.7, 0.0),
            0.2: (569.53, 36.58),
            0.3: (519.73, 26.55),
        },
        r'$\mathrm{CfC}_\mathrm{stoch}^\mathrm{OP}$': {
            0.0: (724.32, 16.03),
            0.1: (697.32, 17.28),
            0.2: (672.14, 12.43),
            0.3: (630.33, 27.61),
        },
        # 'LTC_OP_stoch': {
        #     0.1: (642.81, 17.38),
        #     0.2: (597.67, 32.84),
        #     0.3: (485.32, 64.43),
        # },
    },
}

titles = {
    # 'cartpole_perturb':         'Cartpole-Swingup (perturbed)',
    'cartpole_perturb_flicker': r'$\mathrm{cartpole}_{\mathrm{perturbed-flickering}}^\sigma$',
    'acrobot_perturb':          r'$\mathrm{acrobot-swingup}_\mathrm{perturbed}^\sigma$',
    'walker_run_perturb':       r'$\mathrm{walker-run}_\mathrm{perturbed}^\sigma$',
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
    plt.xlabel(r'$\sigma$ (noise level)')
    plt.ylabel(r'Mean \& std. episodic return')
    plt.xticks([0.0, 0.1, 0.2, 0.3])
    plt.legend()
    plt.tight_layout()

# ---- 3) show all plots ----
plt.show()