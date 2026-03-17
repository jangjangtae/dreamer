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

  def normalize(self, x, clip=5.0):
    if self.count < self.warmup:
      return 0.0
    z = (float(x) - self.mean) / max(self.std, self.eps)
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


def tester_train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  assert args.from_checkpoint, (
      'tester_train은 clean gameplay checkpoint를 초기값으로 쓰는 걸 권장함. '
      '--run.from_checkpoint를 지정해줘.')

  # --------------------------------------------------
  # Hyperparameters via env vars
  # --------------------------------------------------
  lambda_task = float(os.getenv('TESTER_LAMBDA_TASK', '1.0'))
  lambda_bug = float(os.getenv('TESTER_LAMBDA_BUG', '0.1'))
  lambda_cov = float(os.getenv('TESTER_LAMBDA_COV', '0.01'))
  lambda_rep = float(os.getenv('TESTER_LAMBDA_REPEAT', '0.03'))

  novelty_stride = int(os.getenv('TESTER_NOVELTY_STRIDE', '8'))
  repeat_window = int(os.getenv('TESTER_REPEAT_WINDOW', '4'))
  repeat_reward_eps = float(os.getenv('TESTER_REPEAT_REWARD_EPS', '0.001'))

  bug_clip = float(os.getenv('TESTER_BUG_Z_CLIP', '5.0'))
  bug_warmup = int(os.getenv('TESTER_BUG_NORM_WARMUP', '100'))

  ref_score_key = os.getenv('TESTER_REF_SCORE_KEY', 'log/ref_bug_score')
  use_clean_only_baseline = bool(int(os.getenv('TESTER_CLEAN_ONLY_BASELINE', '1')))
  ref_checkpoint = os.getenv('TESTER_REF_CHECKPOINT', args.from_checkpoint)

  # --------------------------------------------------
  # Competence-aware trade-off params
  # --------------------------------------------------
  baseline_score = float(os.getenv('TESTER_BASELINE_SCORE', '0.0'))
  floor_ratio = float(os.getenv('TESTER_TASK_FLOOR_RATIO', '0.5'))
  clean_window = int(os.getenv('TESTER_CLEAN_EP_WINDOW', '20'))

  # clean episode에서는 bug/cov를 거의 끄자
  clean_bug_scale = float(os.getenv('TESTER_CLEAN_BUG_SCALE', '0.0'))
  clean_cov_scale = float(os.getenv('TESTER_CLEAN_COV_SCALE', '0.0'))

  # competence floor 아래로 내려가면 bug/cov를 줄인다
  below_floor_bug_scale = float(os.getenv('TESTER_BELOW_FLOOR_BUG_SCALE', '0.25'))
  below_floor_cov_scale = float(os.getenv('TESTER_BELOW_FLOOR_COV_SCALE', '0.25'))

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

  bug_norm = RunningNorm(warmup=bug_warmup)
  novelty = ImageHashNovelty(stride=novelty_stride)
  repeat_penalty = RepeatPenalty(
      window=repeat_window, reward_eps=repeat_reward_eps)

  # 최근 clean episode score 추적
  recent_clean_scores = deque(maxlen=clean_window)
  episode_fault_seen = collections.defaultdict(int)

  print('Tester training logdir:', logdir)
  print('Reference checkpoint:', ref_checkpoint)
  print(
      'Tester reward weights:',
      dict(task=lambda_task, bug=lambda_bug, cov=lambda_cov, repeat=lambda_rep))
  print('Reference bug key:', ref_score_key)
  print(
      'Competence floor:',
      dict(
          baseline_score=baseline_score,
          floor_ratio=floor_ratio,
          clean_window=clean_window,
          clean_bug_scale=clean_bug_scale,
          clean_cov_scale=clean_cov_scale,
          below_floor_bug_scale=below_floor_bug_scale,
          below_floor_cov_scale=below_floor_cov_scale,
      ))

  def competence_ok():
    if baseline_score <= 0:
      return True
    if len(recent_clean_scores) < max(5, clean_window // 2):
      return True
    return float(np.mean(recent_clean_scores)) >= baseline_score * floor_ratio

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

      if ep_fault == 0:
        recent_clean_scores.append(float(score))

      result['clean_score_mean_recent'] = (
          float(np.mean(recent_clean_scores)) if recent_clean_scores else 0.0)
      result['competence_ok'] = 1.0 if competence_ok() else 0.0

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
  # --------------------------------------------------
  @elements.timer.section('shape_reward')
  def shape_reward(tran, worker):
    game_reward = _scalar(tran['reward'])
    fault_episode = int(_scalar(tran.get('log/fault_episode', 0.0)) > 0.5)

    if tran['is_first']:
      bug_reward = 0.0
      cov_reward = 0.0
      rep_pen = 0.0
      tester_reward = game_reward
      effective_bug_lambda = 0.0
      effective_cov_lambda = 0.0
    else:
      raw_bug = _scalar(tran.get(ref_score_key, 0.0))

      if use_clean_only_baseline:
        if fault_episode == 0:
          bug_norm.update(raw_bug)
      else:
        bug_norm.update(raw_bug)

      bug_reward = bug_norm.normalize(raw_bug, clip=bug_clip)
      cov_reward = novelty.reward(tran['image'])
      rep_pen = repeat_penalty.penalty(
          worker, tran.get('action', None), game_reward)

      # 기본적으로 clean episode에서는 bug/cov를 끈다
      if fault_episode == 0:
        effective_bug_lambda = lambda_bug * clean_bug_scale
        effective_cov_lambda = lambda_cov * clean_cov_scale
      else:
        effective_bug_lambda = lambda_bug
        effective_cov_lambda = lambda_cov

      # gameplay competence가 floor 아래면 bug/cov를 더 줄인다
      if not competence_ok():
        effective_bug_lambda *= below_floor_bug_scale
        effective_cov_lambda *= below_floor_cov_scale

      tester_reward = (
          lambda_task * game_reward
          + effective_bug_lambda * bug_reward
          + effective_cov_lambda * cov_reward
          - lambda_rep * rep_pen
      )

    tran['reward'] = np.float32(tester_reward)

    tran['log/game_reward'] = np.float32(game_reward)
    tran['log/tester_reward'] = np.float32(tester_reward)
    tran['log/bug_reward'] = np.float32(bug_reward)
    tran['log/cov_reward'] = np.float32(cov_reward)
    tran['log/repeat_penalty'] = np.float32(rep_pen)
    tran['log/bug_norm_mean'] = np.float32(bug_norm.mean)
    tran['log/bug_norm_std'] = np.float32(bug_norm.std)
    tran['log/effective_bug_lambda'] = np.float32(effective_bug_lambda)
    tran['log/effective_cov_lambda'] = np.float32(effective_cov_lambda)
    tran['log/competence_ok'] = np.float32(1.0 if competence_ok() else 0.0)

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
