import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import umap
from matplotlib import rc
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

def plot_ep_rollout(states, wm_states, rewards, title, save_path):
	combinations = np.hstack([wm_states, wm_states**2])
	W, _, _, _ = np.linalg.lstsq(combinations, states, rcond=None)
	predicted_states_best_fit = combinations @ W

	# plot states trajectory vs best fit, all in one plot
	fig, ax = plt.subplots()
	ax.plot(states[:, 0], label='True State')
	ax.plot(predicted_states_best_fit[:, 0], label='Best Fit')
	ax.set_title(title)
	ax.set_xlabel('Time')
	ax.set_ylabel('State')
	ax.legend()
	plt.show()
	plt.savefig(save_path)

	# return plot
	return fig, states, predicted_states_best_fit

def plot_umap(states: np.ndarray, wm_states: np.ndarray, title, save_path):
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
	plt.colorbar(scatter2, label="Pole Angular Velocity")
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
	plt.colorbar(scatter2, label="Pole Angular Velocity")
	plt.title("UMAP Projection of Actual States (Colored by Log Pole Angular Velocity)")
	plt.xlabel("UMAP 1")
	plt.ylabel("UMAP 2")

	plt.tight_layout()
	plt.show()

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
	fig, axes = plt.subplots(2, 3, figsize=(12, 8))

	for i, ax in enumerate(axes.flat[:5]):
		ax.scatter(states[:, i], predicted_states_best_fit[:, i], alpha=0.6, label=f"{state_labels[i]} vs best fit")
		ax.plot(states[:, i], states[:, i], color='red', linestyle='dashed', label="Ideal Fit (y=x)")
		ax.set_xlabel(f"True {state_labels[i]}")
		ax.set_ylabel(f"Best Linear Combination")
		ax.legend()
		ax.set_title(f"Correlation coef. for {state_labels[i]}: {correlations[i]:.2f}")

	plt.tight_layout()
	plt.show()
	# save
	plt.savefig(save_path)

	# return plot
	return fig, states, predicted_states_best_fit