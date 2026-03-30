import json
import os
from pathlib import Path
from collections import deque, defaultdict

import crafter
import elements
import embodied
import numpy as np
import math


class Crafter(embodied.Env):

  def __init__(
      self,
      task,
      size=(64, 64),
      logs=False,
      logdir=None,
      seed=None,
      fault_enabled=False,
      action_drop_prob=0.0,
      fallback_action=0,
      fault_start_episode=0,
      fault_seed=None,
      trace_path=None,
  ):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()

    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

    # RNG
    self._fault_seed = fault_seed if fault_seed is not None else seed
    self._rng = np.random.default_rng(self._fault_seed)

    # Legacy fixed fault mode
    self._fault_enabled = bool(fault_enabled)
    self._action_drop_prob = float(action_drop_prob)
    self._fallback_action = int(fallback_action)
    self._fault_start_episode = int(fault_start_episode)

    self._fault_enabled = bool(int(os.getenv('CRAFTER_FAULT', '0')))
    self._action_drop_prob = float(os.getenv('CRAFTER_ACTION_DROP_PROB', '0.0'))
    self._fallback_action = int(os.getenv('CRAFTER_FALLBACK_ACTION', '0'))

    # --------------------------------------------------
    # Fault profile
    # --------------------------------------------------
    self._fault_sampler = bool(int(os.getenv('CRAFTER_FAULT_SAMPLER', '0')))
    self._fault_profile = os.getenv('CRAFTER_FAULT_PROFILE', 'train').strip().lower()
    if self._fault_profile == 'eval':
      self._fault_profile = 'eval_seen'
    assert self._fault_profile in ('train', 'eval_seen', 'eval_holdout')

    self._fault_episode_prob = float(os.getenv('CRAFTER_FAULT_EP_PROB', '0.3'))
    self._fault_episode_prob_min = float(os.getenv('CRAFTER_FAULT_EP_PROB_MIN', '0.1'))
    self._fault_episode_prob_max = float(os.getenv('CRAFTER_FAULT_EP_PROB_MAX', '0.5'))
    self._fault_episode_prob_step = float(os.getenv('CRAFTER_FAULT_EP_PROB_STEP', '0.02'))

    self._fault_severities = self._parse_csv(
        os.getenv('CRAFTER_FAULT_SEVERITIES', '0.1,0.2'),
        cast=float)

    # cooldown
    default_cooldown = '8' if self._fault_profile == 'train' else '0'
    self._fault_cooldown_steps = int(os.getenv('CRAFTER_FAULT_COOLDOWN', default_cooldown))
    self._fault_cooldown_min = int(os.getenv('CRAFTER_FAULT_COOLDOWN_MIN', '0'))
    self._fault_cooldown_max = int(os.getenv('CRAFTER_FAULT_COOLDOWN_MAX', '20'))
    self._fault_cooldown_step = int(os.getenv('CRAFTER_FAULT_COOLDOWN_STEP', '1'))
    self._target_fault_apply_rate = float(os.getenv('CRAFTER_TARGET_FAULT_APPLY_RATE', '0.015'))

    # scheduler
    self._adaptive_scheduler = bool(int(os.getenv('CRAFTER_ADAPTIVE_SCHEDULER', '1')))
    self._sched_baseline_score = float(os.getenv('CRAFTER_SCHED_BASELINE_SCORE', '8.0'))
    self._sched_floor_ratio = float(os.getenv('CRAFTER_SCHED_FLOOR_RATIO', '0.7'))
    self._sched_window = int(os.getenv('CRAFTER_SCHED_WINDOW', '30'))

    if self._fault_profile == 'train':
      default_families = 'action_exec,context_exec,reward_timing'
      default_action = 'remap_after_success_switch,delay_after_success,sticky_after_repeat_switch'
      default_context = 'ignore_nonzero_after_reward,ignore_switch_late_episode'
      default_reward = 'reward_delay_on_positive,reward_scale_half_on_positive_switch'
      default_term = ''

    elif self._fault_profile == 'eval_seen':
      # 학습 때 본 버그들 평가
      default_families = 'action_exec,context_exec,reward_timing'
      default_action = 'remap_after_success_switch,delay_after_success,sticky_after_repeat_switch'
      default_context = 'ignore_nonzero_after_reward,ignore_switch_late_episode'
      default_reward = 'reward_delay_on_positive,reward_scale_half_on_positive_switch'
      default_term = ''

    elif self._fault_profile == 'eval_holdout':
      # 학습에 없던 holdout bug들
      default_families = 'action_exec,context_exec,reward_timing,termination_logic'
      default_action = 'remap_after_repeat_switch,delay_after_late_episode_switch'
      default_context = 'ignore_nonzero_after_two_rewards'
      default_reward = 'reward_zero_after_repeat_switch,reward_delay_after_two_rewards'
      default_term = 'early_done_after_repeat_switch'

    self._fault_families = self._parse_csv(
        os.getenv('CRAFTER_FAULT_FAMILIES', default_families))
    self._action_subtypes = self._parse_csv(
        os.getenv('CRAFTER_ACTION_SUBTYPES', default_action))
    self._context_subtypes = self._parse_csv(
        os.getenv('CRAFTER_CONTEXT_SUBTYPES', default_context))
    self._reward_subtypes = self._parse_csv(
        os.getenv('CRAFTER_REWARD_SUBTYPES', default_reward))
    self._termination_subtypes = self._parse_csv(
        os.getenv('CRAFTER_TERMINATION_SUBTYPES', default_term))

    self._fault_verbose = bool(int(os.getenv('CRAFTER_FAULT_VERBOSE', '0')))

    # trace file
    if trace_path is None:
      trace_path = os.getenv(
          'CRAFTER_TRACE_PATH',
          '/home/railab/logdir/fault_trace.jsonl')
    self._trace_path = Path(trace_path) if trace_path else None
    if self._trace_path is not None:
      self._trace_path.parent.mkdir(parents=True, exist_ok=True)

    # episode state
    self._fault_count = 0
    self._fault_episode = 0
    self._fault_spec = None

    self._last_reward = 0.0
    self._last_requested_action = 0
    self._prev_executed_action = 0

    self._requested_hist = deque([0, 0], maxlen=2)
    self._executed_hist = deque([0, 0], maxlen=2)
    self._reward_hist = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)

    self._sticky_action = None
    self._sticky_remaining = 0
    self._pending_reward = 0.0
    self._after_positive_window = 0
    self._fault_cooldown = 0

    # adaptive scheduler stats
    self._recent_clean_scores = deque(maxlen=self._sched_window)
    self._recent_fault_apply_rates = deque(maxlen=self._sched_window)
    self._subtype_apply_counts = defaultdict(int)

  # --------------------------------------------------
  # Helpers
  # --------------------------------------------------
  def _parse_csv(self, raw, cast=str):
    items = []
    for x in raw.split(','):
      x = x.strip()
      if not x:
        continue
      items.append(cast(x))
    return items

  def _choice(self, values, default=None):
    if not values:
      return default
    idx = int(self._rng.integers(0, len(values)))
    return values[idx]

  def _weighted_choice(self, items, counts):
    if not items:
      return None
    weights = []
    for item in items:
      c = counts.get(item, 0)
      weights.append(1.0 / math.sqrt(c + 1.0))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(self._rng.choice(len(items), p=weights))
    return items[idx]

  def _family_subtypes(self, family):
    if family == 'action_exec':
      return self._action_subtypes
    if family == 'context_exec':
      return self._context_subtypes
    if family == 'reward_timing':
      return self._reward_subtypes
    if family == 'termination_logic':
      return self._termination_subtypes
    return []

  def _valid_families(self):
    valid = []
    for fam in self._fault_families:
      if self._family_subtypes(fam):
        valid.append(fam)
    return valid

  def _trigger_name_from_subtype(self, subtype):
    mapping = {
        'remap_after_success_switch': 'after_positive_window + switched_action',
        'delay_after_success': 'after_positive_window',
        'sticky_after_repeat_switch': 'repeat_then_switch',
        'ignore_nonzero_after_reward': 'after_positive_window + nonzero_action',
        'ignore_switch_late_episode': 'late_episode + switched_action',
        'reward_delay_on_positive': 'positive_reward',
        'reward_scale_half_on_positive_switch': 'positive_reward + switched_action',
        'reward_zero_on_positive': 'positive_reward',
        'early_done_after_success_switch': 'after_positive_window + switched_action',
        'drop_to_fallback': 'everywhere',
        'remap_after_repeat_switch': 'repeat_then_switch',
        'delay_after_late_episode_switch': 'late_episode + switched_action',
        'ignore_nonzero_after_two_rewards': 'two_recent_positive_rewards',
        'reward_zero_after_repeat_switch': 'positive_reward + repeat_then_switch',
        'reward_delay_after_two_rewards': 'positive_reward + two_recent_positive_rewards',
        'early_done_after_repeat_switch': 'repeat_then_switch',
    }
    return mapping.get(subtype, 'custom')

  def _sample_fault_spec(self):
    self._fault_spec = None
    self._fault_episode = 0

    if self._fault_sampler:
      if self._rng.random() >= self._fault_episode_prob:
        return

      families = self._valid_families()
      if not families:
        return

      family_counts = {f: sum(
          self._subtype_apply_counts.get(s, 0) for s in self._family_subtypes(f))
          for f in families}
      family = self._weighted_choice(families, family_counts)
      subtype = self._weighted_choice(
          self._family_subtypes(family), self._subtype_apply_counts)

      severity = float(self._choice(self._fault_severities, 0.1))
      self._fault_spec = {
          'family': family,
          'type': subtype,
          'severity': severity,
          'trigger': self._trigger_name_from_subtype(subtype),
      }
      self._fault_episode = 1
      return

    if self._fault_enabled:
      self._fault_spec = {
          'family': 'legacy_action_exec',
          'type': 'drop_to_fallback',
          'severity': self._action_drop_prob,
          'trigger': 'everywhere',
          'fallback_action': self._fallback_action,
      }
      self._fault_episode = 1

  def _is_nonzero(self, action):
    return int(action) != 0

  def _switched_action(self, requested_action):
    requested_action = int(requested_action)
    prev = int(self._last_requested_action)
    return (
        self._is_nonzero(requested_action) and
        self._is_nonzero(prev) and
        requested_action != prev
    )

  def _repeat_then_switch(self, requested_action):
    requested_action = int(requested_action)
    a0, a1 = int(self._requested_hist[0]), int(self._requested_hist[1])
    return (
        self._is_nonzero(a0) and
        a0 == a1 and
        self._is_nonzero(requested_action) and
        requested_action != a1
    )

  def _late_episode(self):
    return self._length is not None and self._length >= 50

  def _recent_positive_count(self, n=4):
    vals = list(self._reward_hist)[-n:]
    return sum(1 for r in vals if float(r) > 0.0)

  def _success_window_active(self):
    return self._after_positive_window > 0

  def _random_other_action(self, requested_action):
    n = int(self._env.action_space.n)
    if n <= 1:
      return int(requested_action)
    candidates = [a for a in range(n) if a != int(requested_action)]
    idx = int(self._rng.integers(0, len(candidates)))
    return int(candidates[idx])

  def _can_fire_now(self):
    if self._fault_spec is None:
      return False
    if self._episode < self._fault_start_episode:
      return False
    if self._fault_cooldown > 0:
      return False

    # train: deterministic trigger firing
    # eval_seen / eval_holdout: stochastic severity
    if self._fault_profile in ('eval_seen', 'eval_holdout'):
      sev = float(self._fault_spec.get('severity', 0.0))
      return self._rng.random() < sev
    return True

  def _set_cooldown(self):
    self._fault_cooldown = int(self._fault_cooldown_steps)

  def _update_scheduler_after_episode(self):
    if not self._adaptive_scheduler:
      return

    ep_fault_rate = float(self._fault_count / max(1, self._length))
    self._recent_fault_apply_rates.append(ep_fault_rate)

    if self._fault_episode == 0:
      self._recent_clean_scores.append(float(self._reward))

    clean_ok = True
    if self._sched_baseline_score > 0 and len(self._recent_clean_scores) >= max(5, self._sched_window // 3):
      clean_mean = float(np.mean(self._recent_clean_scores))
      clean_ok = clean_mean >= self._sched_floor_ratio * self._sched_baseline_score

    mean_fault_apply = (
        float(np.mean(self._recent_fault_apply_rates))
        if len(self._recent_fault_apply_rates) > 0 else 0.0
    )

    # adaptive fault episode probability
    if clean_ok:
      self._fault_episode_prob = min(
          self._fault_episode_prob_max,
          self._fault_episode_prob + self._fault_episode_prob_step)
    else:
      self._fault_episode_prob = max(
          self._fault_episode_prob_min,
          self._fault_episode_prob - self._fault_episode_prob_step)

    # adaptive cooldown
    if mean_fault_apply < self._target_fault_apply_rate:
      self._fault_cooldown_steps = max(
          self._fault_cooldown_min,
          self._fault_cooldown_steps - self._fault_cooldown_step)
    elif mean_fault_apply > 1.5 * self._target_fault_apply_rate:
      self._fault_cooldown_steps = min(
          self._fault_cooldown_max,
          self._fault_cooldown_steps + self._fault_cooldown_step)

  # --------------------------------------------------
  # Fault application
  # --------------------------------------------------
  def _apply_action_fault(self, requested_action):
    env_action = int(requested_action)
    applied = 0
    fault_type = 'none'

    if self._sticky_remaining > 0 and self._sticky_action is not None:
      env_action = int(self._sticky_action)
      self._sticky_remaining -= 1
      applied = 1
      fault_type = 'sticky_after_repeat_switch'
      self._set_cooldown()
      return env_action, applied, fault_type

    if self._fault_spec is None:
      return env_action, applied, fault_type

    family = self._fault_spec['family']
    subtype = self._fault_spec['type']

    if family == 'legacy_action_exec':
      if self._can_fire_now():
        fallback = int(self._fault_spec.get('fallback_action', self._fallback_action))
        env_action = fallback
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if family not in ('action_exec', 'context_exec'):
      return env_action, applied, fault_type

    if subtype == 'remap_after_success_switch':
      if self._success_window_active() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = self._random_other_action(requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'delay_after_success':
      if self._success_window_active() and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = int(self._last_requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'sticky_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        self._sticky_action = int(requested_action)
        self._sticky_remaining = 1
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'ignore_nonzero_after_reward':
      if self._success_window_active() and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'ignore_nonzero_after_two_rewards':
      if self._recent_positive_count(4) >= 2 and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'ignore_switch_late_episode':
      if self._late_episode() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'remap_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        env_action = self._random_other_action(requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'delay_after_late_episode_switch':
      if self._late_episode() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = int(self._last_requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    return env_action, applied, fault_type

  def _apply_reward_fault(self, reward, requested_action):
    reward = float(reward)
    applied = 0
    fault_type = 'none'

    if self._pending_reward != 0.0:
      reward += float(self._pending_reward)
      self._pending_reward = 0.0

    if self._fault_spec is None:
      return reward, applied, fault_type

    family = self._fault_spec['family']
    subtype = self._fault_spec['type']

    if family != 'reward_timing':
      return reward, applied, fault_type

    if reward <= 0.0:
      return reward, applied, fault_type

    if subtype == 'reward_delay_on_positive':
      if self._can_fire_now():
        self._pending_reward += float(reward)
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_scale_half_on_positive_switch':
      if self._switched_action(requested_action) and self._can_fire_now():
        reward = 0.5 * reward
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_zero_on_positive':
      if self._can_fire_now():
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_zero_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    if subtype == 'reward_delay_after_two_rewards':
      if self._recent_positive_count(4) >= 2 and self._can_fire_now():
        self._pending_reward += float(reward)
        reward = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return reward, applied, fault_type

    return reward, applied, fault_type

  def _apply_termination_fault(self, requested_action, done, info):
    done = bool(done)
    info = dict(info)
    applied = 0
    fault_type = 'none'

    if self._fault_spec is None:
      return done, info, applied, fault_type

    family = self._fault_spec['family']
    subtype = self._fault_spec['type']

    if family != 'termination_logic':
      return done, info, applied, fault_type

    if done:
      return done, info, applied, fault_type

    if subtype == 'early_done_after_success_switch':
      if self._success_window_active() and self._switched_action(requested_action) and self._can_fire_now():
        done = True
        info['discount'] = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return done, info, applied, fault_type

    if subtype == 'early_done_after_repeat_switch':
      if self._repeat_then_switch(requested_action) and self._can_fire_now():
        done = True
        info['discount'] = 0.0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return done, info, applied, fault_type

    return done, info, applied, fault_type

  # --------------------------------------------------
  # Spaces
  # --------------------------------------------------
  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
        'log/fault_applied': elements.Space(np.int32),
        'log/fault_episode': elements.Space(np.int32),
    }
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements
      })
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  # --------------------------------------------------
  # Main step
  # --------------------------------------------------
  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0.0
      self._fault_count = 0
      self._done = False

      self._last_reward = 0.0
      self._last_requested_action = 0
      self._prev_executed_action = 0
      self._pending_reward = 0.0
      self._sticky_action = None
      self._sticky_remaining = 0
      self._after_positive_window = 0
      self._fault_cooldown = 0

      self._requested_hist = deque([0, 0], maxlen=2)
      self._executed_hist = deque([0, 0], maxlen=2)
      self._reward_hist = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)

      self._sample_fault_spec()
      if self._fault_verbose and self._fault_spec is not None:
        print(f'[FaultSpec] episode={self._episode} profile={self._fault_profile} '
              f'p_fault={self._fault_episode_prob:.3f} cooldown={self._fault_cooldown_steps} '
              f'spec={self._fault_spec}')

      image = self._env.reset()
      return self._obs(
          image=image,
          reward=0.0,
          info={'fault_episode': int(self._fault_episode)},
          is_first=True,
          is_last=False,
          is_terminal=False,
      )

    requested_action = int(action['action'])

    # action fault
    env_action, action_fault_applied, action_fault_type = self._apply_action_fault(
        requested_action)

    # env step
    image, raw_reward, self._done, info = self._env.step(env_action)

    # reward fault
    reward, reward_fault_applied, reward_fault_type = self._apply_reward_fault(
        raw_reward, requested_action)

    # termination fault
    self._done, info, term_fault_applied, term_fault_type = self._apply_termination_fault(
        requested_action, self._done, info)

    fault_applied = int(
        action_fault_applied or reward_fault_applied or term_fault_applied)

    if fault_applied:
      self._fault_count += 1
      subtype = None
      if action_fault_applied:
        subtype = action_fault_type
      elif reward_fault_applied:
        subtype = reward_fault_type
      elif term_fault_applied:
        subtype = term_fault_type
      if subtype is not None and subtype != 'none':
        self._subtype_apply_counts[subtype] += 1

    if action_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = action_fault_type
    elif reward_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = reward_fault_type
    elif term_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = term_fault_type
    else:
      fault_family = self._fault_spec['family'] if self._fault_spec is not None else 'none'
      fault_type = 'none'

    info = dict(info)
    info['reward'] = float(reward)
    info['fault_applied'] = int(fault_applied)
    info['fault_episode'] = int(self._fault_episode)
    info['fault_family'] = fault_family
    info['fault_type'] = fault_type
    info['fault_trigger'] = (
        self._fault_spec.get('trigger', 'none')
        if self._fault_spec is not None else 'none')
    info['fault_severity'] = (
        float(self._fault_spec.get('severity', 0.0))
        if self._fault_spec is not None else 0.0)
    info['requested_action'] = int(requested_action)
    info['executed_action'] = int(env_action)

    self._reward += reward
    self._length += 1

    if self._trace_path is not None:
      record = {
          'episode': int(self._episode),
          'episode_step': int(self._length),
          'reward': float(reward),
          'fault_episode': int(self._fault_episode),
          'fault_applied': int(fault_applied),
          'fault_family': fault_family,
          'fault_type': fault_type,
          'fault_trigger': info['fault_trigger'],
          'fault_severity': float(info['fault_severity']),
          'requested_action': int(requested_action),
          'executed_action': int(env_action),
          'fault_count_so_far': int(self._fault_count),
          'is_last': bool(self._done),
          'is_terminal': bool(info.get('discount', 1.0) == 0.0),
      }
      with self._trace_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    if self._fault_verbose and fault_applied:
      print(
          f'[FaultApplied] ep={self._episode} step={self._length} '
          f'family={fault_family} type={fault_type} '
          f'requested={requested_action} executed={env_action} reward={reward:.3f}'
      )

    if self._done:
      self._update_scheduler_after_episode()

    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)

    # success window based on raw reward event
    if raw_reward > 0.0:
      self._after_positive_window = 3
    elif self._after_positive_window > 0:
      self._after_positive_window -= 1

    if self._fault_cooldown > 0:
      self._fault_cooldown -= 1

    self._last_reward = float(reward)
    self._last_requested_action = int(requested_action)
    self._prev_executed_action = int(env_action)
    self._requested_hist.append(int(requested_action))
    self._executed_hist.append(int(env_action))
    self._reward_hist.append(float(reward))

    return self._obs(
        image=image,
        reward=reward,
        info=info,
        is_first=False,
        is_last=self._done,
        is_terminal=(info.get('discount', 1.0) == 0.0),
    )

  def _obs(
      self,
      image,
      reward,
      info,
      is_first=False,
      is_last=False,
      is_terminal=False,
  ):
    log_reward = 0.0
    if info and 'reward' in info:
      log_reward = info['reward']

    fault_applied = 0
    if info and 'fault_applied' in info:
      fault_applied = info['fault_applied']

    fault_episode = int(self._fault_episode)
    if info and 'fault_episode' in info:
      fault_episode = info['fault_episode']

    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(log_reward)},
        **{'log/fault_applied': np.int32(fault_applied)},
        **{'log/fault_episode': np.int32(fault_episode)},
    )

    if self._logs:
      achievements = info.get('achievements', {}) if info else {}
      log_achievements = {
          f'log/achievement_{k}': achievements.get(k, 0)
          for k in self._achievements
      }
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})

    return obs

  def _write_stats(self, length, reward, info):
    achievements = info.get('achievements', {}) if info else {}
    stats = {
        'episode': self._episode,
        'length': int(length),
        'reward': round(float(reward), 1),
        'fault_episode': int(info.get('fault_episode', 0)),
        'fault_count': int(self._fault_count),
        'fault_rate': float(self._fault_count / max(1, length)),
        'fault_family': info.get('fault_family', 'none'),
        'fault_type': info.get('fault_type', 'none'),
        'fault_trigger': info.get('fault_trigger', 'none'),
        'fault_severity': float(info.get('fault_severity', 0.0)),
        'fault_episode_prob': float(self._fault_episode_prob),
        'fault_cooldown_steps': int(self._fault_cooldown_steps),
        **{
            f'achievement_{k}': achievements.get(k, 0)
            for k in self._achievements
        },
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines, mode='w')
