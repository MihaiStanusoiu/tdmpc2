import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import umap
from matplotlib import rc, cm
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def plot_state_space_heatmap(data, title, x_label, y_label, x_ticks, y_ticks, save_path):
	fig, ax = plt.subplots()
	sns.heatmap(data, ax=ax, cmap='coolwarm', cbar_kws={'label': 'Value'})
	plt.title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_xticks(x_ticks)
	ax.set_yticks(y_ticks)
	# plt.savefig(save_path)
	# plt.close()
	plt.show()

def plot_ep_rollout(states, wm_states, title, save_path):
	# plot trajectory of each state and associated  wm_state with the same color (wm_states are dashed)
	fig = plt.figure(figsize=(12, 6))
	num_states = states.shape[1]
	time = np.arange(states.shape[0])

	color_map = cm.get_cmap('viridis')
	np.linspace(0, 1, num_states)
	colors = [color_map(i) for i in np.linspace(0, 1, num_states)]
	state_labels = [r'$\chi$', r'$\cos{\alpha}$', r'$\sin{\alpha}$', r'$\dot \chi$', r'$\dot \alpha$']

	for i in range(num_states):
		plt.plot(time, states[:, i], label=state_labels[i], color=colors[i])
		plt.plot(time, wm_states[:, i], '--', color=colors[i], label=f'Predicted ' + state_labels[i])

	plt.xlabel('Time Step')
	plt.ylabel('State Value')
	plt.title('True vs Predicted States Over Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()
	plt.savefig(save_path)

	return fig

def plot_imag_trajectories(samples, title, save_path):
	# samples: shape (N, H, D)
	H, N, D = samples.shape
	time = np.arange(H)
	color_map = cm.get_cmap('viridis')
	np.linspace(0, 1, D)
	colors = [color_map(i) for i in np.linspace(0, 1, D)]

	fig = plt.figure(figsize=(14, 6))

	for d in range(D):
		for n in range(N):
			plt.plot(time, samples[:, n, d].cpu().numpy(), color=colors[d], alpha=0.2)
		plt.plot(time, samples[:, :, d].mean(axis=1).cpu().numpy(), color=colors[d], label=f'Mean dim {d}', linewidth=2)

	plt.xlabel("Prediction Horizon")
	plt.ylabel("State Value")
	plt.title("MPPI Sampled Trajectories")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def plot_umap(states: np.ndarray, wm_states: np.ndarray, title, save_path):
	# Check the task type and call the appropriate function
	if 'acrobot' in title:
		return _plot_acrobot_umap(states, wm_states, title, save_path)
	elif 'cartpole' in title:
		return _plot_cartpole_umap(states, wm_states, title, save_path)
	else:
		raise ValueError("Unknown task type. Please provide a valid task name.")

def _plot_acrobot_umap(states: np.ndarray, wm_states: np.ndarray, title, save_path):
	# Apply UMAP to reduce to 2D
	umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(wm_states)
	umap_proj_actual = umap.UMAP(n_components=2, random_state=42).fit_transform(states)

	# Plot the 2D representations
	log_cart_velocity = np.log(np.abs(states[:, 4]) + 1e-6)
	log_pole_velocity = np.log(np.abs(states[:, 5]) + 1e-6)

	# Plot 1: UMAP projection colored by log(cart velocity)
	fig = plt.figure(figsize=(12, 10))

	plt.subplot(2, 2, 1)
	scatter1 = plt.scatter(umap_proj[:, 0], umap_proj[:, 1], c=log_cart_velocity, cmap="viridis", alpha=0.7)
	plt.colorbar(scatter1, label="Log Velocity of Joint 1")
	plt.title("UMAP Projection of Hidden States (Colored by Log Velocity of Joint 1)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	# Plot 2: UMAP projection colored by pole angular velocity
	plt.subplot(2, 2, 2)
	scatter2 = plt.scatter(umap_proj[:, 0], umap_proj[:, 1], c=log_pole_velocity, cmap="plasma", alpha=0.7)
	plt.colorbar(scatter2, label="Log Velocity of Joint 2")
	plt.title("UMAP Projection of Hidden States (Colored by Log Velocity of Joint 2)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	plt.subplot(2, 2, 3)
	scatter1 = plt.scatter(umap_proj_actual[:, 0], umap_proj_actual[:, 1], c=log_cart_velocity, cmap="viridis", alpha=0.7)
	plt.colorbar(scatter1, label="Log Velocity of Joint 1")
	plt.title("UMAP Projection of Actual States (Colored by Log Velocity of Joint 1)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	# Plot 2: UMAP projection colored by pole angular velocity
	plt.subplot(2, 2, 4)
	scatter2 = plt.scatter(umap_proj_actual[:, 0], umap_proj_actual[:, 1], c=log_pole_velocity, cmap="plasma", alpha=0.7)
	plt.colorbar(scatter2, label="Log Velocity of Joint 2")
	plt.title("UMAP Projection of Actual States (Colored by Log Velocity of Joint 2)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	plt.tight_layout()
	plt.show()
	plt.savefig(save_path)


	return fig, umap_proj

def _plot_cartpole_umap(states: np.ndarray, wm_states: np.ndarray, title, save_path):
	# Apply UMAP to reduce to 2D
	umap_proj = umap.UMAP(n_components=2, random_state=42).fit_transform(wm_states)
	umap_proj_actual = umap.UMAP(n_components=2, random_state=42).fit_transform(states)

	# Plot the 2D representations
	log_cart_velocity = np.log(np.abs(states[:, 3]) + 1e-6)
	log_pole_velocity = np.log(np.abs(states[:, 4]) + 1e-6)

	# Plot 1: UMAP projection colored by log(cart velocity)
	fig = plt.figure(figsize=(12, 10))

	plt.subplot(2, 2, 1)
	scatter1 = plt.scatter(umap_proj[:, 0], umap_proj[:, 1], c=log_cart_velocity, cmap="viridis", alpha=0.7)
	plt.colorbar(scatter1, label="Log Cart Velocity")
	plt.title("UMAP Projection of Hidden States (Colored by Log Cart Velocity)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	# Plot 2: UMAP projection colored by pole angular velocity
	plt.subplot(2, 2, 2)
	scatter2 = plt.scatter(umap_proj[:, 0], umap_proj[:, 1], c=log_pole_velocity, cmap="plasma", alpha=0.7)
	plt.colorbar(scatter2, label="Log Pole Angular Velocity")
	plt.title("UMAP Projection of Hidden States (Colored by Log Pole Angular Velocity)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	plt.subplot(2, 2, 3)
	scatter1 = plt.scatter(umap_proj_actual[:, 0], umap_proj_actual[:, 1], c=log_cart_velocity, cmap="viridis", alpha=0.7)
	plt.colorbar(scatter1, label="Log Cart Velocity")
	plt.title("UMAP Projection of Actual States (Colored by Log Cart Velocity)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	# Plot 2: UMAP projection colored by pole angular velocity
	plt.subplot(2, 2, 4)
	scatter2 = plt.scatter(umap_proj_actual[:, 0], umap_proj_actual[:, 1], c=log_pole_velocity, cmap="plasma", alpha=0.7)
	plt.colorbar(scatter2, label="Log Pole Angular Velocity")
	plt.title("UMAP Projection of Actual States (Colored by Log Pole Angular Velocity)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	plt.tight_layout()
	plt.show()
	plt.savefig(save_path)


	return fig, umap_proj


def plot_state_wm_state_correlation(states : np.ndarray, wm_states: np.ndarray, title, save_path):
	fig, ax = plt.subplots()
	# for each state variable, get the best linear combination of wm_states variables and their squares
	combinations = np.hstack([wm_states, wm_states**2])
	ridge = Ridge(alpha=1.0)  # Regularization strength
	ridge.fit(combinations, states)  # Fit the model
	predicted_states_best_fit = ridge.predict(combinations)
	# W, _, _, _ = np.linalg.lstsq(combinations, states, rcond=None)
	# predicted_states_best_fit = combinations @ W

	# Compute correlation coefficients
	correlations = np.array([
	    np.corrcoef(states[:, i], predicted_states_best_fit[:, i])[0, 1]
	    for i in range(len(states[0]))
	])

	# Plot the correlation values
	state_labels = [r'$\chi$', r'$\cos{\alpha}$', r'$\sin{\alpha}$', r'$\dot \chi$', r'$\dot \alpha$']
	plt.rcParams.update({'font.size': 14})
	fig, axes = plt.subplots(2, 3, figsize=(12, 8))

	for i, ax in enumerate(axes.flat[:5]):
		ax.scatter(states[:, i], predicted_states_best_fit[:, i], alpha=0.6, label=f"{state_labels[i]} vs best fit")
		ax.plot(states[:, i], states[:, i], color='red', linestyle='dashed', label="Ideal Fit (y=x)")
		ax.set_xlabel(f"True {state_labels[i]}")
		ax.set_ylabel(f"Best Linear Combination")
		ax.legend()
		ax.set_title(f"Correlation coef. for {state_labels[i]}: {correlations[i]:.2f}")

	fig.delaxes(axes[5])

	plt.tight_layout()
	plt.show()
	# save
	plt.savefig(save_path)

	# return plot
	return fig, states, predicted_states_best_fit