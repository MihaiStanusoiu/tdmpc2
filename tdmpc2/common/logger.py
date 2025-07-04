import dataclasses
import os
import datetime
import re
import uuid

import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf

from common import TASK_SET
from torch.utils.tensorboard import SummaryWriter

CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
    "evaluate_ep": "green",
    "evaluate_task": "green",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
        )

    observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """
    Return a wandb-safe group name for logging.
    Optionally returns group name as list.
    """

    if cfg.wandb_group != '???':
        lst = [cfg.wandb_group]
    else:
        lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, cfg, wandb, fps=30):
        self.cfg = cfg
        self._save_dir = make_dir(cfg.work_dir / 'eval_video')
        self._wandb = wandb
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self._save_dir and self._wandb and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            self.frames.append(env.render(mode='rgb_array'))

    def save(self, step, key='videos/eval_video'):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames)
            return self._wandb.log(
                {key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
            )


class TensorBoardLogger:
    def __init__(self, cfg):
        self._log_dir = os.path.join(cfg.work_dir, 'tensorboard')
        self._writer = SummaryWriter(log_dir=self._log_dir)
        self._step = 0

    def log(self, d, category):
        for key, value in d.items():
            self._writer.add_scalar(f'{category}/{key}', value, self._step)
        self._step += 1

    def close(self):
        self._writer.close()

class Logger:
    """Primary logging object. Logs either locally, using wandb, or TensorBoard."""

    def __init__(self, cfg):
        self._log_dir = make_dir(cfg.work_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = cfg.save_csv
        self._save_agent = cfg.save_agent
        self._save_buffer = cfg.save_buffer
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._checkpoint = cfg.checkpoint
        self._eval = []
        self._use_tensorboard = cfg.get("use_tensorboard", False)
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")
        if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
        else:
            os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
            import wandb
        wandb.login(key="0b961bb8b95bbca48519a84eeca715dc4187268f")
        wandb.init(
			project=self.project,
			entity=self.entity,
            id=str(cfg.id) if cfg.id != '???' else None,
			name=str(cfg.exp_name),
            resume='allow' if cfg.checkpoint != '???' else None,
            group=self._group,
            # tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
            dir=self._log_dir,
            config=dataclasses.asdict(cfg),
        )

        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb
        self._video = (
            VideoRecorder(cfg, self._wandb)
            if self._wandb and cfg.save_video
            else None
        )

        if self._use_tensorboard:
            self._tensorboard_logger = TensorBoardLogger(cfg)
        else:
            self._tensorboard_logger = None
            
    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return self._model_dir

    def load_agent(self, version='latest'):
        try:
            identifier = 'model'
            artifact = self._wandb.use_artifact(
                self._group + '-' + str(self._seed) + '-' + str(identifier) + f':{version}', type='model'
            )
        except:
            identifier = str(self._checkpoint)
            artifact = self._wandb.use_artifact(
                self._group + '-' + str(self._seed) + '-' + str(identifier) + ':v0', type='model'
            )
        fp = f'{str(identifier)}.pt'
        artifact_dir = artifact.download()
        return os.path.join(artifact_dir, fp)

    def load_buffer(self, version='latest'):
        identifier = 'buffer'
        artifact = self._wandb.use_artifact(
            self._group + '-' + str(self._seed) + '-' + str(identifier) + f':{version}', type='dataset'
        )
        fp = 'buffer'
        artifact_dir = artifact.download()
        return artifact_dir
        # return os.path.join(artifact_dir, fp)

    def save_agent(self, agent=None, buffer=None, metrics={}, identifier='model', buffer_identifier='buffer'):
        if self._save_agent and agent:
            fp = self._model_dir / f'{str(identifier)}.pt'
            agent.save(fp, metrics=metrics)

            if self._wandb:
                try:
                    artifact = self._wandb.use_artifact(
                        self._group + '-' + str(self._seed) + '-' + str(identifier) + ":latest",
                    )
                    draft_artifact = artifact.new_draft()
                    draft_artifact.add_file(fp)
                    self._wandb.log_artifact(draft_artifact)
                except:
                    artifact = self._wandb.Artifact(
                        self._group + '-' + str(self._seed) + '-' + str(identifier),
                        type='model',
                    )
                    artifact.add_file(fp)
                    self._wandb.log_artifact(artifact)
        if self._save_buffer and buffer:
            bfp = os.path.join(self._model_dir, 'buffer')
            ok = buffer.dumps(bfp)

            if self._wandb and ok:
                try:
                    artifact = self._wandb.use_artifact(
                        self._group + '-' + str(self._seed) + '-' + str(buffer_identifier) + ":latest",
                    )
                    draft_artifact = artifact.new_draft()
                    draft_artifact.add_file(bfp)
                    self._wandb.log_artifact(draft_artifact)
                except:
                    artifact = self._wandb.Artifact(
                        self._group + '-' + str(self._seed) + '-' + str(buffer_identifier),
                        type='dataset',
                    )
                    artifact.add_dir(bfp)
                    self._wandb.log_artifact(artifact)

    def finish(self, agent=None, buffer=None, metrics={}):
        try:
            self.save_agent(agent, buffer, metrics)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()
        if self._tensorboard_logger:
            self._tensorboard_logger.close()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def pprint_multitask(self, d, cfg):
        """Pretty-print evaluation metrics for multi-task training."""
        print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
        dmcontrol_reward = []
        metaworld_reward = []
        metaworld_success = []
        for k, v in d.items():
            if '+' not in k:
                continue
            task = k.split('+')[1]
            if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
                dmcontrol_reward.append(v)
                print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
            elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
                if k.startswith('episode_reward'):
                    metaworld_reward.append(v)
                elif k.startswith('episode_success'):
                    metaworld_success.append(v)
                    print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
        dmcontrol_reward = np.nanmean(dmcontrol_reward)
        d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
        print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
        if cfg.task == 'mt80':
            metaworld_reward = np.nanmean(metaworld_reward)
            metaworld_success = np.nanmean(metaworld_success)
            d['episode_reward+avg_metaworld'] = metaworld_reward
            d['episode_success+avg_metaworld'] = metaworld_success
            print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
            print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

    def log_state_wm_prediction(self, fig, state_labels, states, predicted_states_best_fit, key="state_prediction_correlation"):
        # log received fig
        self._wandb.log({key: fig})

        # # Log Data to W&B
        # table = self._wandb.Table(columns=["State", "True Value", "Best Fit Value"])
        #
        # for i in range(states.shape[0]):
        #     for j in range(states.shape[1]):
        #         table.add_data(state_labels[j], states[i, j], predicted_states_best_fit[i, j])
        #
        # self._wandb.log({"state_predictions": table})
        #
        # # Optional: Log scatter plots directly
        # for i, label in enumerate(state_labels):
        #     self._wandb.log({
        #         f"{key}/{label}": self._wandb.plot.scatter(
        #             table, x="True Value", y="Best Fit Value",
        #             title=f"Scatter Plot for {label}"
        #         )
        #     })

    def log_fig(self, fig, key):
        self._wandb.log({key: fig})

    def log_video(self, step, title):
        self._wandb.log({"rollout_video": self._wandb.Video(title, fps=30, format="mp4")}, step=step)

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb:
            if category in {"train", "eval"}:
                xkey = "step"
            elif category == "pretrain":
                xkey = "iteration"
            elif category == "evaluate_ep":
                xkey = "episode"
            elif category == "evaluate_task":
                xkey = "task"
            _d = dict()
            for k, v in d.items():
                _d[category + "/" + k] = v
            self._wandb.log(_d, step=d[xkey])
        if category == "eval" and self._save_csv:
            keys = ["step", "episode_reward"]
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / "eval.csv", header=keys, index=None
            )
        if self._tensorboard_logger:
            self._tensorboard_logger.log(d, category)
        self._print(d, category)