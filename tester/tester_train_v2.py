import collections
import math
import os
from collections import deque
from functools import partial as bind

import elements
import embodied
import numpy as np


def _scalar(x, default=0.0):
  if x is None:
    return default
  arr = np.asarray(x)
  if arr.size == 0:
    return default
  return float(arr.reshape(-1)[0])


class RunningNorm:

  def __init__(self, eps=1e-6, warmup=100):
    self.eps = eps
    self.warmup = warmup
    self.count = 0
    self.mean = 0.0
    self.m2 = 0.0

  def update(self, x):
    x = float(x)
    self.count += 1
    delta = x - self.mean
    self.mean += delta / self.count
    delta2 = x - self.mean
    self.m2 += delta * delta2

  @property
  def std(self):
    if self.count < 2:
      return 1.0
    return math.sqrt(max(self.m2 / (self.count - 1), self.eps))

  def normalize(self, x, clip=5.0, signed=True):
    if self.count < self.warmup:
      return 0.0
    z = (float(x) - self.mean) / max(self.std, self.eps)
    if signed:
      return float(np.clip(z, -clip, clip))
    return float(np.clip(z, 0.0, clip))


class ImageHashNovelty:

  def __init__(self, stride=8):
    self.stride = int(stride)
    self.counts = collections.defaultdict(int)

  def _hash(self, image):
    image = np.asarray(image)
    if image.ndim != 3:
      return None
    small = image[::self.stride, ::self.stride]
    if small.shape[-1] == 3:
      small = small.mean(-1)
    small = small.astype(np.uint8, copy=False)
    return small.tobytes()

  def reward(self, image):
    key = self._hash(image)
    if key is None:
      return 0.0
    count = self.counts[key]
    self.counts[key] += 1
    return float(1.0 / math.sqrt(count + 1.0))


class RepeatPenalty:

  def __init__(self, window=4, reward_eps=1e-3):
    self.window = int(window)
    self.reward_eps = float(reward_eps)
    self.buffers = collections.defaultdict(lambda: deque(maxlen=self.window))

  def penalty(self, worker, action, game_reward):
    if action is None:
      return 0.0
    try:
      action = int(np.asarray(action).reshape(-1)[0])
    except Exception:
      return 0.0
    buf = self.buffers[worker]
    buf.append(action)
    if len(buf) < self.window:
      return 0.0
    if len(set(buf)) == 1 and abs(float(game_reward)) <= self.reward_eps:
      return 1.0
    return 0.0


class DualState:
  """
  Constrained RL style dual variables.
  - lambda_task: clean gameplay competence constraint
  - lambda_rep : repeat-behavior constraint
  """

  def __init__(
      self,
      baseline_score,
      alpha=0.7,
      rho_rep=0.08,
      eta_task=0.02,
      eta_rep=0.02,
      init_lambda_task=1.0,
      init_lambda_rep=0.1,
      max_lambda_task=5.0,
      max_lambda_rep=5.0,
  ):
    self.baseline_score = float(baseline_score)
    self.alpha = float(alpha)
    self.rho_rep = float(rho_rep)
    self.eta_task = float(eta_task)
    self.eta_rep = float(eta_rep)

    self.lambda_task = float(init_lambda_task)
    self.lambda_rep = float(init_lambda_rep)

    self.max_lambda_task = float(max_lambda_task)
    self.max_lambda_rep = float(max_lambda_rep)

  def update(self, clean_score_mean=None, repeat_mean=None):
    if self.baseline_score > 0 and clean_score_mean is not None:
      gap_task = self.alpha * self.baseline_score - float(clean_score_mean)
      self.lambda_task = float(np.clip(
          self.lambda_task + self.eta_task * gap_task,
          0.0,
          self.max_lambda_task,
      ))

    if repeat_mean is not None:
      gap_rep = float(repeat_mean) - self.rho_rep
      self.lambda_rep = float(np.clip(
          self.lambda_rep + self.eta_rep * gap_rep,
          0.0,
          self.max_lambda_rep,
      ))


class AdaptiveExplorationState:
  """
  Adaptive controller for bug / coverage emphasis.
  - beta_cov: coverage contribution
  - w_bug   : bug reward contribution
  """

  def __init__(
      self,
      baseline_score,
      floor_ratio=0.7,
      init_beta_cov=0.25,
      min_beta_cov=0.02,
      max_beta_cov=0.5,
      lr_cov_up=0.01,
      lr_cov_down=0.03,
      init_w_bug=1.0,
      min_w_bug=0.3,
      max_w_bug=3.0,
      lr_bug_up=0.05,
      lr_bug_down=0.05,
      target_bug_gain=0.5,
  ):
    self.baseline_score = float(baseline_score)
    self.floor_ratio = float(floor_ratio)

    self.beta_cov = float(init_beta_cov)
    self.min_beta_cov = float(min_beta_cov)
    self.max_beta_cov = float(max_beta_cov)
    self.lr_cov_up = float(lr_cov_up)
    self.lr_cov_down = float(lr_cov_down)

    self.w_bug = float(init_w_bug)
    self.min_w_bug = float(min_w_bug)
    self.max_w_bug = float(max_w_bug)
    self.lr_bug_up = float(lr_bug_up)
    self.lr_bug_down = float(lr_bug_down)

    self.target_bug_gain = float(target_bug_gain)
    self.prev_bug_mean = None
    self.last_bug_gain = 0.0

  def competence_ok(self, clean_score_mean):
    if self.baseline_score <= 0:
      return True
    if clean_score_mean is None:
      return True
    return float(clean_score_mean) >= self.floor_ratio * self.baseline_score

  def update(self, clean_score_mean=None, bug_score_mean=None):
    ok = self.competence_ok(clean_score_mean)

    # Coverage weight: only strong when clean competence is okay
    if ok:
      self.beta_cov = float(np.clip(
          self.beta_cov + self.lr_cov_up,
          self.min_beta_cov, self.max_beta_cov))
    else:
      self.beta_cov = float(np.clip(
          self.beta_cov - self.lr_cov_down,
          self.min_beta_cov, self.max_beta_cov))

    # Bug weight: increase if bug discovery stagnates, decrease if competence is low
    bug_gain = 0.0
    if bug_score_mean is not None:
      if self.prev_bug_mean is not None:
        bug_gain = float(bug_score_mean) - float(self.prev_bug_mean)
      self.prev_bug_mean = float(bug_score_mean)

    self.last_bug_gain = bug_gain

    if ok:
      if bug_gain < self.target_bug_gain:
        self.w_bug = float(np.clip(
            self.w_bug + self.lr_bug_up,
            self.min_w_bug, self.max_w_bug))
      else:
        self.w_bug = float(np.clip(
            self.w_bug - 0.5 * self.lr_bug_down,
            self.min_w_bug, self.max_w_bug))
    else:
      self.w_bug = float(np.clip(
          self.w_bug - self.lr_bug_down,
          self.min_w_bug, self.max_w_bug))


def tester_train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  assert args.from_checkpoint, (
      'tester_train은 clean gameplay checkpoint를 초기값으로 쓰는 걸 권장함. '
      '--run.from_checkpoint를 지정해줘.')

  # --------------------------------------------------
  # Hyperparameters via env vars
  # --------------------------------------------------
  novelty_stride = int(os.getenv('TESTER_NOVELTY_STRIDE', '8'))
  repeat_window = int(os.getenv('TESTER_REPEAT_WINDOW', '4'))
  repeat_reward_eps = float(os.getenv('TESTER_REPEAT_REWARD_EPS', '0.001'))

  ref_score_key = os.getenv('TESTER_REF_SCORE_KEY', 'log/ref_bug_score')
  use_clean_only_baseline = bool(int(os.getenv('TESTER_CLEAN_ONLY_BASELINE', '1')))
  ref_checkpoint = os.getenv('TESTER_REF_CHECKPOINT', args.from_checkpoint)

  # --------------------------------------------------
  # Constrained objective params
  # --------------------------------------------------
  baseline_score = float(os.getenv('TESTER_BASELINE_SCORE', '10.0'))
  floor_ratio = float(os.getenv('TESTER_TASK_FLOOR_RATIO', '0.7'))
  clean_window = int(os.getenv('TESTER_CLEAN_EP_WINDOW', '20'))
  repeat_window_stats = int(os.getenv('TESTER_REPEAT_EP_WINDOW', '20'))
  bug_window = int(os.getenv('TESTER_BUG_EP_WINDOW', '20'))

  repeat_budget = float(os.getenv('TESTER_REPEAT_BUDGET', '0.08'))

  # dual variables
  dual_lr_task = float(os.getenv('TESTER_DUAL_LR_TASK', '0.02'))
  dual_lr_rep = float(os.getenv('TESTER_DUAL_LR_REPEAT', '0.02'))
  init_lambda_task = float(os.getenv('TESTER_INIT_LAMBDA_TASK', '1.0'))
  init_lambda_rep = float(os.getenv('TESTER_INIT_LAMBDA_REPEAT', '0.1'))
  max_lambda_task = float(os.getenv('TESTER_MAX_LAMBDA_TASK', '5.0'))
  max_lambda_rep = float(os.getenv('TESTER_MAX_LAMBDA_REPEAT', '5.0'))

  # adaptive exploration
  init_beta_cov = float(os.getenv('TESTER_INIT_BETA_COV', '0.25'))
  min_beta_cov = float(os.getenv('TESTER_MIN_BETA_COV', '0.02'))
  max_beta_cov = float(os.getenv('TESTER_MAX_BETA_COV', '0.5'))
  lr_cov_up = float(os.getenv('TESTER_LR_COV_UP', '0.01'))
  lr_cov_down = float(os.getenv('TESTER_LR_COV_DOWN', '0.03'))

  init_w_bug = float(os.getenv('TESTER_INIT_W_BUG', '1.0'))
  min_w_bug = float(os.getenv('TESTER_MIN_W_BUG', '0.3'))
  max_w_bug = float(os.getenv('TESTER_MAX_W_BUG', '3.0'))
  lr_bug_up = float(os.getenv('TESTER_LR_BUG_UP', '0.05'))
  lr_bug_down = float(os.getenv('TESTER_LR_BUG_DOWN', '0.05'))
  target_bug_gain = float(os.getenv('TESTER_TARGET_BUG_GAIN', '0.5'))

  # normalization
  task_clip = float(os.getenv('TESTER_TASK_Z_CLIP', '5.0'))
  bug_clip = float(os.getenv('TESTER_BUG_Z_CLIP', '5.0'))
  cov_clip = float(os.getenv('TESTER_COV_Z_CLIP', '5.0'))
  rep_clip = float(os.getenv('TESTER_REP_Z_CLIP', '5.0'))

  norm_warmup = int(os.getenv('TESTER_NORM_WARMUP', '100'))
  bug_warmup = int(os.getenv('TESTER_BUG_NORM_WARMUP', '100'))

  # --------------------------------------------------
  # Agents / replay / logger
  # --------------------------------------------------
  tester_agent = make_agent()
  ref_agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()

  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  # reward component normalization
  task_norm = RunningNorm(warmup=norm_warmup)
  bug_norm = RunningNorm(warmup=bug_warmup)
  cov_norm = RunningNorm(warmup=norm_warmup)
  rep_norm = RunningNorm(warmup=norm_warmup)

  novelty = ImageHashNovelty(stride=novelty_stride)
  repeat_penalty = RepeatPenalty(
      window=repeat_window, reward_eps=repeat_reward_eps)

  # episode-level tracking
  recent_clean_scores = deque(maxlen=clean_window)
  recent_repeat_means = deque(maxlen=repeat_window_stats)
  recent_fault_bug_scores = deque(maxlen=bug_window)
  episode_fault_seen = collections.defaultdict(int)

  dual = DualState(
      baseline_score=baseline_score,
      alpha=floor_ratio,
      rho_rep=repeat_budget,
      eta_task=dual_lr_task,
      eta_rep=dual_lr_rep,
      init_lambda_task=init_lambda_task,
      init_lambda_rep=init_lambda_rep,
      max_lambda_task=max_lambda_task,
      max_lambda_rep=max_lambda_rep,
  )

  adaptive = AdaptiveExplorationState(
      baseline_score=baseline_score,
      floor_ratio=floor_ratio,
      init_beta_cov=init_beta_cov,
      min_beta_cov=min_beta_cov,
      max_beta_cov=max_beta_cov,
      lr_cov_up=lr_cov_up,
      lr_cov_down=lr_cov_down,
      init_w_bug=init_w_bug,
      min_w_bug=min_w_bug,
      max_w_bug=max_w_bug,
      lr_bug_up=lr_bug_up,
      lr_bug_down=lr_bug_down,
      target_bug_gain=target_bug_gain,
  )

  print('Tester training logdir:', logdir)
  print('Reference checkpoint:', ref_checkpoint)
  print('Reference bug key:', ref_score_key)
  print('Constrained RL settings:')
  print(dict(
      baseline_score=baseline_score,
      floor_ratio=floor_ratio,
      repeat_budget=repeat_budget,
      dual_lr_task=dual_lr_task,
      dual_lr_rep=dual_lr_rep,
      init_lambda_task=init_lambda_task,
      init_lambda_rep=init_lambda_rep,
      init_beta_cov=init_beta_cov,
      init_w_bug=init_w_bug,
      target_bug_gain=target_bug_gain,
  ))

  # --------------------------------------------------
  # Logging
  # --------------------------------------------------
  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    if tran['is_first']:
      episode.reset()
      episode_fault_seen[worker] = 0

    game_reward = _scalar(tran.get('log/game_reward', tran['reward']))
    tester_reward = _scalar(tran['reward'])
    fault_episode = int(_scalar(tran.get('log/fault_episode', 0.0)) > 0.5)
    fault_applied = int(_scalar(tran.get('log/fault_applied', 0.0)) > 0.5)
    episode_fault_seen[worker] = max(
        episode_fault_seen[worker], fault_episode, fault_applied)

    episode.add('score', game_reward, agg='sum')
    episode.add('tester_score', tester_reward, agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', game_reward, agg='stack')

    for key, value in tran.items():
      if getattr(value, 'dtype', None) == np.uint8 and getattr(value, 'ndim', None) == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')

    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      tester_score = result.pop('tester_score')
      length = result.pop('length')
      ep_fault = int(episode_fault_seen[worker])

      logger.add({
          'score': score,
          'tester_score': tester_score,
          'length': length,
          'fault_episode': ep_fault,
      }, prefix='episode')

      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()

      rep_avg = float(result.get('log/raw_repeat_penalty/avg', 0.0))
      recent_repeat_means.append(rep_avg)

      if ep_fault == 0:
        recent_clean_scores.append(float(score))
      else:
        recent_fault_bug_scores.append(
            float(result.get('log/raw_bug_reward/avg', 0.0)))

      clean_score_mean = (
          float(np.mean(recent_clean_scores))
          if len(recent_clean_scores) > 0 else None
      )
      repeat_mean = (
          float(np.mean(recent_repeat_means))
          if len(recent_repeat_means) > 0 else None
      )
      fault_bug_mean = (
          float(np.mean(recent_fault_bug_scores))
          if len(recent_fault_bug_scores) > 0 else None
      )

      dual.update(clean_score_mean, repeat_mean)
      adaptive.update(clean_score_mean, fault_bug_mean)

      result['clean_score_mean_recent'] = (
          clean_score_mean if clean_score_mean is not None else 0.0
      )
      result['repeat_mean_recent'] = (
          repeat_mean if repeat_mean is not None else 0.0
      )
      result['fault_bug_mean_recent'] = (
          fault_bug_mean if fault_bug_mean is not None else 0.0
      )

      result['lambda_task'] = dual.lambda_task
      result['lambda_rep'] = dual.lambda_rep
      result['beta_cov'] = adaptive.beta_cov
      result['w_bug'] = adaptive.w_bug
      result['bug_gain_recent'] = adaptive.last_bug_gain
      result['competence_ok'] = (
          1.0 if adaptive.competence_ok(clean_score_mean) else 0.0
      )

      epstats.add(result)

  # --------------------------------------------------
  # Policy wrapper
  # --------------------------------------------------
  def init_dual_policy(batch_size):
    return {
        'tester': tester_agent.init_policy(batch_size),
        'ref': ref_agent.init_policy(batch_size),
    }

  def dual_policy(carry, obs, mode='train'):
    tester_carry, acts, tester_outs = tester_agent.policy(
        carry['tester'], obs, mode='train')
    ref_carry, _, ref_outs = ref_agent.policy(
        carry['ref'], obs, mode='eval')

    outs = dict(tester_outs)

    for key, value in ref_outs.items():
      if key == 'bug/score':
        outs['log/ref_bug_score'] = value
      elif key == 'bug/kl':
        outs['log/ref_bug_kl'] = value
      elif key == 'bug/reward_err':
        outs['log/ref_bug_reward_err'] = value
      elif key == 'bug/continue_err':
        outs['log/ref_bug_continue_err'] = value

    return {'tester': tester_carry, 'ref': ref_carry}, acts, outs

  # --------------------------------------------------
  # Reward shaping
  # scalar reward:
  #   lambda_task * normalized_task
  # + fault * (w_bug * normalized_bug + beta_cov * normalized_cov)
  # - lambda_rep * normalized_repeat
  # --------------------------------------------------
  @elements.timer.section('shape_reward')
  def shape_reward(tran, worker):
    game_reward = _scalar(tran['reward'])
    fault_episode = int(_scalar(tran.get('log/fault_episode', 0.0)) > 0.5)

    if tran['is_first']:
      raw_bug = 0.0
      raw_cov = 0.0
      raw_rep = 0.0

      norm_task = 0.0
      norm_bug = 0.0
      norm_cov = 0.0
      norm_rep = 0.0

      tester_reward = 0.0
      exploration_term = 0.0
    else:
      raw_bug = _scalar(tran.get(ref_score_key, 0.0))
      raw_cov = novelty.reward(tran['image'])
      raw_rep = repeat_penalty.penalty(
          worker, tran.get('action', None), game_reward)

      # running stats update
      task_norm.update(game_reward)
      cov_norm.update(raw_cov)
      rep_norm.update(raw_rep)

      if use_clean_only_baseline:
        if fault_episode == 0:
          bug_norm.update(raw_bug)
      else:
        bug_norm.update(raw_bug)

      # normalized signals
      norm_task = task_norm.normalize(game_reward, clip=task_clip, signed=True)
      norm_bug = bug_norm.normalize(raw_bug, clip=bug_clip, signed=False)
      norm_cov = cov_norm.normalize(raw_cov, clip=cov_clip, signed=False)
      norm_rep = rep_norm.normalize(raw_rep, clip=rep_clip, signed=False)

      exploration_term = 0.0
      if fault_episode:
        exploration_term = (
            adaptive.w_bug * norm_bug
            + adaptive.beta_cov * norm_cov
        )

      tester_reward = (
          dual.lambda_task * norm_task
          + exploration_term
          - dual.lambda_rep * norm_rep
      )

    tran['reward'] = np.float32(tester_reward)

    # raw logs
    tran['log/game_reward'] = np.float32(game_reward)
    tran['log/raw_bug_reward'] = np.float32(raw_bug)
    tran['log/raw_cov_reward'] = np.float32(raw_cov)
    tran['log/raw_repeat_penalty'] = np.float32(raw_rep)

    # normalized logs
    tran['log/task_reward_norm'] = np.float32(norm_task)
    tran['log/bug_reward_norm'] = np.float32(norm_bug)
    tran['log/cov_reward_norm'] = np.float32(norm_cov)
    tran['log/repeat_penalty_norm'] = np.float32(norm_rep)
    tran['log/exploration_term'] = np.float32(exploration_term)

    # controller logs
    tran['log/lambda_task'] = np.float32(dual.lambda_task)
    tran['log/lambda_rep'] = np.float32(dual.lambda_rep)
    tran['log/beta_cov'] = np.float32(adaptive.beta_cov)
    tran['log/w_bug'] = np.float32(adaptive.w_bug)
    tran['log/bug_gain_recent'] = np.float32(adaptive.last_bug_gain)
    tran['log/competence_ok'] = np.float32(
        1.0 if adaptive.competence_ok(
            float(np.mean(recent_clean_scores)) if recent_clean_scores else None
        ) else 0.0
    )

  # --------------------------------------------------
  # Replay add with filtering
  # --------------------------------------------------
  @elements.timer.section('replay_add_filtered')
  def replay_add_filtered(tran, worker):
    allowed = set(tester_agent.spaces.keys())
    filtered = {k: v for k, v in tran.items() if k in allowed}
    replay.add(filtered, worker)

  # --------------------------------------------------
  # Driver / replay / stream
  # --------------------------------------------------
  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(shape_reward)
  driver.on_step(replay_add_filtered)
  driver.on_step(logfn)

  stream_train = iter(tester_agent.stream(make_stream(replay, 'train')))
  stream_report = iter(tester_agent.stream(make_stream(replay, 'report')))
  carry_train = [tester_agent.init_train(args.batch_size)]
  carry_report = tester_agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = tester_agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')

  driver.on_step(trainfn)

  # --------------------------------------------------
  # Checkpoints
  # --------------------------------------------------
  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = tester_agent
  cp.replay = replay

  load_regex = args.from_checkpoint_regex if hasattr(args, 'from_checkpoint_regex') else None

  if load_regex is None:
    elements.checkpoint.load(
        args.from_checkpoint,
        dict(agent=tester_agent.load))
  else:
    elements.checkpoint.load(
        args.from_checkpoint,
        dict(agent=bind(tester_agent.load, regex=load_regex)))

  if load_regex is None:
    elements.checkpoint.load(
        ref_checkpoint,
        dict(agent=ref_agent.load))
  else:
    elements.checkpoint.load(
        ref_checkpoint,
        dict(agent=bind(ref_agent.load, regex=load_regex)))

  cp.load_or_save()

  print('Start tester fine-tuning loop')

  driver.reset(init_dual_policy)

  while step < args.steps:
    driver(dual_policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = tester_agent.report(
            carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
