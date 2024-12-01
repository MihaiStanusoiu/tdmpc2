from matplotlib import pyplot as plt
import seaborn as sns

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