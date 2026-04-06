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
      # probing / revisit / delayed-context 기반 holdout bug들
      default_families = 'action_exec,context_exec'
      default_action = 'revisit_action_delay,delayed_switch_failure'
      default_context = 'revisit_action_ignore'
      default_reward = ''
      default_term = ''

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

    # --------------------------------------------------
    # Unified output directory and file paths
    # --------------------------------------------------
    explicit_trace = None
    if trace_path is not None:
      explicit_trace = Path(str(trace_path)).expanduser()
    else:
      env_trace = os.getenv('CRAFTER_TRACE_PATH', '').strip()
      if env_trace:
        explicit_trace = Path(env_trace).expanduser()

    if self._logdir:
      self._output_dir = Path(str(self._logdir)).expanduser()
    elif explicit_trace is not None:
      self._output_dir = explicit_trace.resolve().parent
    else:
      self._output_dir = Path('/home/railab/logdir')

    self._output_dir.mkdir(parents=True, exist_ok=True)

    # Keep all env-side files in the same directory.
    self._trace_path = self._output_dir / 'fault_trace.jsonl'
    self._stats_path = self._output_dir / 'stats.jsonl'
    self._episode_summary_path = self._output_dir / 'episode_summary.jsonl'

    self._trace_path.touch(exist_ok=True)
    self._stats_path.touch(exist_ok=True)
    self._episode_summary_path.touch(exist_ok=True)

    if self._fault_verbose:
      print(f'[OutputDir] {self._output_dir}')
      print(f'[FaultTrace] {self._trace_path}')
      print(f'[Stats] {self._stats_path}')
      print(f'[EpisodeSummary] {self._episode_summary_path}')

    # episode state
    self._fault_count = 0
    self._fault_episode = 0
    self._semantic_fault_episode_seen = 0
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

    # global-step aligned logging for stable plotting
    self._global_step = 0
    self._episode_start_global_step = 0

    self._reset_episode_aggregation()

    # --------------------------------------------------
    # Probe-required holdout support
    # --------------------------------------------------
    self._num_actions = int(self._env.action_space.n)

    # episode step counter for delayed-context bugs
    self._episode_step = 0
    self._last_positive_reward_step = -10**9

    # revisit/context memory
    self._ctx_visit_steps = defaultdict(lambda: deque(maxlen=4))
    self._last_ctx_key = None

    # tunable thresholds
    self._revisit_gap = int(os.getenv('CRAFTER_REVISIT_GAP', '12'))
    self._delayed_switch_min = int(os.getenv('CRAFTER_DELAYED_SWITCH_MIN', '2'))
    self._delayed_switch_max = int(os.getenv('CRAFTER_DELAYED_SWITCH_MAX', '3'))


    # --------------------------------------------------
    # Tester-oriented semantic reward shaping
    # --------------------------------------------------
    self._tester_reward_enabled = bool(int(os.getenv('CRAFTER_TESTER_REWARD', '1')))
    self._tester_alpha_task = float(os.getenv('CRAFTER_TESTER_ALPHA_TASK', '1.0'))
    self._tester_ctx_reward = float(os.getenv('CRAFTER_TESTER_CTX_REWARD', '0.02'))
    self._tester_anom_reward = float(os.getenv('CRAFTER_TESTER_ANOM_REWARD', '0.02'))
    self._tester_reproduce_reward = float(os.getenv('CRAFTER_TESTER_REPRODUCE_REWARD', '0.10'))
    self._tester_compare_reward = float(os.getenv('CRAFTER_TESTER_COMPARE_REWARD', '0.10'))
    self._tester_confirm_reward = float(os.getenv('CRAFTER_TESTER_CONFIRM_REWARD', '0.12'))
    self._tester_followup_reward = float(os.getenv('CRAFTER_TESTER_FOLLOWUP_REWARD', '0.08'))
    self._tester_repeat_penalty = float(os.getenv('CRAFTER_TESTER_REPEAT_PENALTY', '0.01'))
    self._tester_same_action_cap = int(os.getenv('CRAFTER_TESTER_SAME_ACTION_CAP', '3'))
    self._tester_reward_cap = float(os.getenv('CRAFTER_TESTER_REWARD_CAP', '0.35'))

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
        'revisit_action_ignore': 'recent_revisit + nonzero_action',
        'revisit_action_delay': 'recent_revisit + switched_action',
        'delayed_switch_failure': 'positive_reward_after_2to3_steps + switched_action',
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

  def _context_key_from_image(self, image):
    image = np.asarray(image)
    if image.ndim != 3:
      return None
    small = image[::8, ::8]
    if small.shape[-1] == 3:
      small = small.mean(-1)
    small = small.astype(np.uint8, copy=False)
    return small.tobytes()

  def _record_context(self, image):
    key = self._context_key_from_image(image)
    self._last_ctx_key = key
    if key is not None:
      self._ctx_visit_steps[key].append(self._episode_step)

  def _recent_revisit_active(self):
    key = self._last_ctx_key
    if key is None:
      return False
    hist = self._ctx_visit_steps.get(key, None)
    if hist is None or len(hist) < 2:
      return False
    return (hist[-1] - hist[-2]) <= self._revisit_gap

  def _delayed_switch_after_success(self, requested_action):
    delay = int(self._episode_step - self._last_positive_reward_step)
    return (
        self._delayed_switch_min <= delay <= self._delayed_switch_max
        and self._switched_action(requested_action)
    )


  def _reset_tester_reward_state(self):
    self._episode_task_reward = 0.0
    self._episode_tester_bonus = 0.0
    self._episode_repeat_penalty = 0.0
    self._same_action_run = 0
    self._last_requested_action_for_reward = None
    self._semantic_proc_state = defaultdict(lambda: {
        'ctx': 0,
        'anom': 0,
        'reproduce': 0,
        'compare': 0,
        'confirm': 0,
        'followup': 0,
    })
    self._active_semantic_family = None

  def _semantic_context_keys(self, info):
    info = dict(info or {})
    keys = []
    mapping = [
        ('upgrade_collect', info.get('semantic_ctx_upgrade_collect', 0), info.get('semantic_upgrade_collect_count', 0)),
        ('retry_craft', info.get('semantic_ctx_retry_craft', 0), info.get('semantic_retry_craft_count', 0)),
        ('relocate_station', info.get('semantic_ctx_relocate_station', 0), info.get('semantic_relocate_station_count', 0)),
        ('valid_progress', info.get('semantic_valid_progress', 0), info.get('semantic_valid_progress_count', 0)),
        ('station_reuse', info.get('semantic_station_reuse', 0), info.get('semantic_station_reuse_count', 0)),
        ('delayed_after_use', info.get('semantic_delayed_after_use', 0), info.get('semantic_delayed_after_use_count', 0)),
    ]
    for name, flag, count in mapping:
      if int(flag) > 0 or int(count) > 0:
        keys.append(name)
    return keys

  def _tester_reward_from_semantics(self, task_reward, requested_action, info):
    task_reward = float(task_reward)
    info = dict(info or {})
    final_reward = self._tester_alpha_task * task_reward
    bonus = 0.0
    repeat_penalty = 0.0
    components = {}

    if self._last_requested_action_for_reward is None or int(requested_action) != int(self._last_requested_action_for_reward):
      self._same_action_run = 1
    else:
      self._same_action_run += 1
    self._last_requested_action_for_reward = int(requested_action)

    if self._same_action_run > self._tester_same_action_cap:
      repeat_penalty = self._tester_repeat_penalty * float(self._same_action_run - self._tester_same_action_cap)

    context_keys = self._semantic_context_keys(info)
    for key in context_keys:
      state = self._semantic_proc_state[key]
      if not state['ctx']:
        state['ctx'] = 1
        bonus += self._tester_ctx_reward
        components[f'ctx:{key}'] = components.get(f'ctx:{key}', 0.0) + self._tester_ctx_reward
        self._active_semantic_family = key

    semantic_fault_applied = int(info.get('semantic_fault_applied', 0))
    semantic_fault_type = str(info.get('semantic_fault_type', 'none'))
    if semantic_fault_applied and semantic_fault_type != 'none':
      key = semantic_fault_type
      self._active_semantic_family = key
      state = self._semantic_proc_state[key]
      if not state['anom']:
        state['anom'] = 1
        bonus += self._tester_anom_reward
        components[f'anom:{key}'] = components.get(f'anom:{key}', 0.0) + self._tester_anom_reward
      elif not state['reproduce']:
        state['reproduce'] = 1
        bonus += self._tester_reproduce_reward
        components[f'reproduce:{key}'] = components.get(f'reproduce:{key}', 0.0) + self._tester_reproduce_reward

    active_key = self._active_semantic_family
    if active_key is not None:
      state = self._semantic_proc_state[active_key]
      if int(info.get('semantic_post_fault_window', 0)) > 0 and state['anom'] and not state['followup']:
        state['followup'] = 1
        bonus += self._tester_followup_reward
        components[f'followup:{active_key}'] = components.get(f'followup:{active_key}', 0.0) + self._tester_followup_reward
      if int(info.get('semantic_post_fault_switch', 0)) > 0 and state['anom'] and not state['compare']:
        state['compare'] = 1
        bonus += self._tester_compare_reward
        components[f'compare:{active_key}'] = components.get(f'compare:{active_key}', 0.0) + self._tester_compare_reward
      if int(info.get('semantic_post_fault_nonzero', 0)) > 0 and state['anom'] and not state['confirm']:
        state['confirm'] = 1
        bonus += self._tester_confirm_reward
        components[f'confirm:{active_key}'] = components.get(f'confirm:{active_key}', 0.0) + self._tester_confirm_reward

    if bonus > self._tester_reward_cap:
      bonus = self._tester_reward_cap

    final_reward = final_reward + bonus - repeat_penalty

    self._episode_task_reward += task_reward
    self._episode_tester_bonus += bonus
    self._episode_repeat_penalty += repeat_penalty

    info['task_reward_raw'] = float(task_reward)
    info['tester_bonus'] = float(bonus)
    info['tester_repeat_penalty'] = float(repeat_penalty)
    info['tester_reward'] = float(final_reward)
    info['tester_reward_components'] = components
    return float(final_reward), info
  def _reset_episode_aggregation(self):
    self._ep_fault_applied_any = 0
    self._ep_semantic_fault_applied_any = 0
    self._ep_semantic_fault_count = 0
    self._ep_fault_family = 'none'
    self._ep_fault_type = 'none'
    self._ep_fault_trigger = 'none'
    self._ep_semantic_trigger_count = 0
    self._ep_semantic_first_trigger_step = -1
    self._ep_semantic_ctx_upgrade_collect_count = 0
    self._ep_semantic_ctx_retry_craft_count = 0
    self._ep_semantic_ctx_relocate_station_count = 0
    self._ep_semantic_valid_progress_count = 0
    self._ep_semantic_station_reuse_count = 0
    self._ep_semantic_delayed_after_use_count = 0
    self._ep_semantic_post_fault_window_count = 0
    self._ep_semantic_post_fault_nonzero_count = 0
    self._ep_semantic_post_fault_switch_count = 0
    self._reset_tester_reward_state()

  def _update_episode_aggregation(
      self,
      fault_applied,
      fault_family,
      fault_type,
      fault_trigger,
      semantic_fault_applied,
      semantic_trigger_count,
      semantic_first_trigger_step,
      semantic_ctx_upgrade_collect,
      semantic_ctx_retry_craft,
      semantic_ctx_relocate_station,
      semantic_valid_progress_count,
      semantic_station_reuse_count,
      semantic_delayed_after_use_count,
      semantic_post_fault_window,
      semantic_post_fault_nonzero,
      semantic_post_fault_switch):
    self._ep_fault_applied_any = max(self._ep_fault_applied_any, int(fault_applied))
    self._ep_semantic_fault_applied_any = max(
        self._ep_semantic_fault_applied_any, int(semantic_fault_applied))
    self._ep_semantic_fault_count += int(semantic_fault_applied)

    if fault_family and fault_family != 'none':
      self._ep_fault_family = fault_family
    if fault_type and fault_type != 'none':
      self._ep_fault_type = fault_type
    if fault_trigger and fault_trigger != 'none':
      self._ep_fault_trigger = fault_trigger

    self._ep_semantic_trigger_count = max(
        self._ep_semantic_trigger_count, int(semantic_trigger_count))
    if int(semantic_first_trigger_step) >= 0:
      if self._ep_semantic_first_trigger_step < 0:
        self._ep_semantic_first_trigger_step = int(semantic_first_trigger_step)
      else:
        self._ep_semantic_first_trigger_step = min(
            self._ep_semantic_first_trigger_step, int(semantic_first_trigger_step))

    self._ep_semantic_ctx_upgrade_collect_count = max(
        self._ep_semantic_ctx_upgrade_collect_count, int(semantic_ctx_upgrade_collect))
    self._ep_semantic_ctx_retry_craft_count = max(
        self._ep_semantic_ctx_retry_craft_count, int(semantic_ctx_retry_craft))
    self._ep_semantic_ctx_relocate_station_count = max(
        self._ep_semantic_ctx_relocate_station_count, int(semantic_ctx_relocate_station))
    self._ep_semantic_valid_progress_count = max(
        self._ep_semantic_valid_progress_count, int(semantic_valid_progress_count))
    self._ep_semantic_station_reuse_count = max(
        self._ep_semantic_station_reuse_count, int(semantic_station_reuse_count))
    self._ep_semantic_delayed_after_use_count = max(
        self._ep_semantic_delayed_after_use_count, int(semantic_delayed_after_use_count))

    self._ep_semantic_post_fault_window_count += int(semantic_post_fault_window)
    self._ep_semantic_post_fault_nonzero_count += int(semantic_post_fault_nonzero)
    self._ep_semantic_post_fault_switch_count += int(semantic_post_fault_switch)

  def _resolve_episode_fault_identity(self, info=None):
    info = dict(info or {})
    fault_family = getattr(self, '_ep_fault_family', 'none')
    fault_type = getattr(self, '_ep_fault_type', 'none')
    fault_trigger = getattr(self, '_ep_fault_trigger', 'none')

    ctx_upgrade = int(getattr(self, '_ep_semantic_ctx_upgrade_collect_count', 0))
    ctx_retry = int(getattr(self, '_ep_semantic_ctx_retry_craft_count', 0))
    ctx_relocate = int(getattr(self, '_ep_semantic_ctx_relocate_station_count', 0))
    ctx_progress = int(getattr(self, '_ep_semantic_valid_progress_count', 0))
    ctx_station_reuse = int(getattr(self, '_ep_semantic_station_reuse_count', 0))
    ctx_delayed = int(getattr(self, '_ep_semantic_delayed_after_use_count', 0))
    semantic_fault_episode = int(info.get('semantic_fault_episode', 0)) or int(getattr(self, '_semantic_fault_episode_seen', 0))

    if fault_family == 'none' and semantic_fault_episode:
      fault_family = 'semantic_high_level'

    if fault_type == 'none' and fault_family == 'semantic_high_level':
      candidates = [
          ('tool_collect_desync_on_upgrade', ctx_upgrade),
          ('craft_result_missing_on_retry', ctx_retry),
          ('station_place_ghost_on_relocate', ctx_relocate),
          ('achievement_unlock_missing_after_valid_progress', ctx_progress),
          ('station_usable_flag_broken_after_relocate', ctx_station_reuse),
          ('delayed_inventory_desync_after_station_use', ctx_delayed),
      ]
      best_name, best_count = max(candidates, key=lambda x: x[1])
      if best_count > 0:
        fault_type = best_name
        trigger_map = {
            'tool_collect_desync_on_upgrade': 'first_valid_collect_after_pickaxe_upgrade',
            'craft_result_missing_on_retry': 'repeat_craft_same_recipe_within_episode',
            'station_place_ghost_on_relocate': 'second_or_later_table_or_furnace_placement',
            'achievement_unlock_missing_after_valid_progress': 'valid_progress_milestone_reached',
            'station_usable_flag_broken_after_relocate': 'station_reuse_after_relocate',
            'recipe_precondition_mischeck_on_retry': 'repeat_craft_same_recipe_within_episode',
            'delayed_inventory_desync_after_station_use': 'delayed_after_station_use',
        }
        if fault_trigger == 'none':
          fault_trigger = trigger_map.get(best_name, 'custom')

    return fault_family, fault_type, fault_trigger

  def _write_episode_summary(self, length, reward, info):
    if self._episode_summary_path is None:
      return

    info = dict(info or {})
    semantic_first_trigger_step = int(self._ep_semantic_first_trigger_step)
    semantic_first_trigger_global_step = (
        int(self._episode_start_global_step + semantic_first_trigger_step)
        if semantic_first_trigger_step >= 0 else -1)

    ctx_upgrade = int(self._ep_semantic_ctx_upgrade_collect_count)
    ctx_retry = int(self._ep_semantic_ctx_retry_craft_count)
    ctx_relocate = int(self._ep_semantic_ctx_relocate_station_count)
    fault_family, fault_type, fault_trigger = self._resolve_episode_fault_identity(info)

    summary = {
        'episode': int(self._episode),
        'episode_length': int(length),
        'episode_reward': round(float(reward), 6),
        'start_global_step': int(self._episode_start_global_step),
        'end_global_step': int(self._global_step),
        'fault_episode': int(info.get('fault_episode', int(self._fault_episode or self._semantic_fault_episode_seen))),
        'fault_applied_any': int(self._ep_fault_applied_any),
        'fault_count': int(self._fault_count),
        'fault_rate': float(self._fault_count / max(1, length)),
        'fault_family': fault_family,
        'fault_type': fault_type,
        'fault_trigger': fault_trigger,
        'semantic_fault_episode': int(info.get('semantic_fault_episode', 0) or self._semantic_fault_episode_seen),
        'semantic_fault_applied_any': int(self._ep_semantic_fault_applied_any),
        'semantic_fault_count': int(self._ep_semantic_fault_count),
        'semantic_fault_type': fault_type if fault_family == 'semantic_high_level' else info.get('semantic_fault_type', 'none'),
        'semantic_trigger_reach': int(self._ep_semantic_trigger_count > 0),
        'semantic_trigger_count': int(self._ep_semantic_trigger_count),
        'semantic_first_trigger_step': semantic_first_trigger_step,
        'semantic_first_trigger_global_step': semantic_first_trigger_global_step,
        'semantic_ctx_upgrade_collect_count': ctx_upgrade,
        'semantic_ctx_retry_craft_count': ctx_retry,
        'semantic_ctx_relocate_station_count': ctx_relocate,
        'semantic_valid_progress_count': int(self._ep_semantic_valid_progress_count),
        'semantic_station_reuse_count': int(self._ep_semantic_station_reuse_count),
        'semantic_delayed_after_use_count': int(self._ep_semantic_delayed_after_use_count),
        'semantic_bug_family_coverage': int(
            (ctx_upgrade > 0) + (ctx_retry > 0) + (ctx_relocate > 0) +
            (int(self._ep_semantic_valid_progress_count) > 0) +
            (int(self._ep_semantic_station_reuse_count) > 0) +
            (int(self._ep_semantic_delayed_after_use_count) > 0)
        ),
        'episode_task_reward': round(float(self._episode_task_reward), 6),
        'episode_tester_bonus': round(float(self._episode_tester_bonus), 6),
        'episode_repeat_penalty': round(float(self._episode_repeat_penalty), 6),
        'semantic_post_fault_window_count': int(self._ep_semantic_post_fault_window_count),
        'semantic_post_fault_nonzero_count': int(self._ep_semantic_post_fault_nonzero_count),
        'semantic_post_fault_switch_count': int(self._ep_semantic_post_fault_switch_count),
    }

    with self._episode_summary_path.open('a', encoding='utf-8') as f:
      f.write(json.dumps(summary) + '\n')
      f.flush()

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

    if (self._fault_episode == 0) and (self._semantic_fault_episode_seen == 0):
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

    if subtype == 'revisit_action_ignore':
      if self._recent_revisit_active() and self._is_nonzero(requested_action) and self._can_fire_now():
        env_action = 0
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'revisit_action_delay':
      if self._recent_revisit_active() and self._switched_action(requested_action) and self._can_fire_now():
        env_action = int(self._last_requested_action)
        applied = 1
        fault_type = subtype
        self._set_cooldown()
      return env_action, applied, fault_type

    if subtype == 'delayed_switch_failure':
      if self._delayed_switch_after_success(requested_action) and self._can_fire_now():
        env_action = self._random_other_action(requested_action)
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
        'log/semantic_fault_applied': elements.Space(np.int32),
        'log/semantic_fault_episode': elements.Space(np.int32),
        'log/semantic_trigger_context': elements.Space(np.int32),
        'log/semantic_trigger_count': elements.Space(np.int32),
        'log/semantic_first_trigger_step': elements.Space(np.int32),
        'log/semantic_ctx_upgrade_collect': elements.Space(np.int32),
        'log/semantic_ctx_retry_craft': elements.Space(np.int32),
        'log/semantic_ctx_relocate_station': elements.Space(np.int32),
        'log/semantic_post_fault_window': elements.Space(np.int32),
        'log/semantic_post_fault_nonzero': elements.Space(np.int32),
        'log/semantic_post_fault_switch': elements.Space(np.int32),
        'log/semantic_upgrade_collect_count': elements.Space(np.int32),
        'log/semantic_retry_craft_count': elements.Space(np.int32),
        'log/semantic_relocate_station_count': elements.Space(np.int32),
        'log/semantic_valid_progress_count': elements.Space(np.int32),
        'log/semantic_station_reuse_count': elements.Space(np.int32),
        'log/semantic_delayed_after_use_count': elements.Space(np.int32),
        'log/task_reward_raw': elements.Space(np.float32),
        'log/tester_bonus': elements.Space(np.float32),
        'log/tester_repeat_penalty': elements.Space(np.float32),
        'log/revisit_context': elements.Space(np.int32),
        'log/delayed_switch_context': elements.Space(np.int32),
        'log/global_step': elements.Space(np.int32),
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
      self._semantic_fault_episode_seen = 0
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

      self._episode_step = 0
      self._last_positive_reward_step = -10**9
      self._ctx_visit_steps = defaultdict(lambda: deque(maxlen=4))
      self._last_ctx_key = None

      self._episode_start_global_step = int(self._global_step)
      self._reset_episode_aggregation()

      self._sample_fault_spec()
      if self._fault_verbose and self._fault_spec is not None:
        print(f'[FaultSpec] episode={self._episode} profile={self._fault_profile} '
              f'p_fault={self._fault_episode_prob:.3f} cooldown={self._fault_cooldown_steps} '
              f'spec={self._fault_spec}')

      image = self._env.reset()
      self._record_context(image)
      return self._obs(
          image=image,
          reward=0.0,
          info={
              'fault_episode': int(self._fault_episode),
              'fault_applied': np.int32(0),
              'semantic_fault_episode': np.int32(0),
              'semantic_fault_applied': np.int32(0),
              'fault_family': 'none',
              'fault_type': 'none',
              'fault_trigger': 'none',
              'fault_severity': 0.0,
              'task_reward_raw': 0.0,
              'tester_bonus': 0.0,
              'tester_repeat_penalty': 0.0,
          },
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

    # semantic high-level fault info from crafter.Env
    info = dict(info)
    semantic_fault_episode = int(info.get('semantic_fault_episode', 0))
    semantic_fault_applied = int(info.get('semantic_fault_applied', 0))
    semantic_fault_family = info.get(
        'semantic_fault_family',
        'semantic_high_level' if (semantic_fault_episode or semantic_fault_applied) else 'none')
    semantic_fault_type = info.get('semantic_fault_type', 'none')
    semantic_fault_trigger = info.get('semantic_fault_trigger', 'none')
    semantic_fault_severity = float(info.get(
        'semantic_fault_severity',
        1.0 if semantic_fault_applied else 0.0))
    semantic_trigger_context = int(info.get('semantic_trigger_context', 0))
    semantic_trigger_type = info.get('semantic_trigger_type', 'none')
    semantic_trigger_count = int(info.get('semantic_trigger_count', 0))
    semantic_first_trigger_step = int(info.get('semantic_first_trigger_step', -1))
    semantic_ctx_upgrade_collect = int(info.get('semantic_ctx_upgrade_collect', 0))
    semantic_ctx_retry_craft = int(info.get('semantic_ctx_retry_craft', 0))
    semantic_ctx_relocate_station = int(info.get('semantic_ctx_relocate_station', 0))
    semantic_post_fault_window = int(info.get('semantic_post_fault_window', 0))
    semantic_post_fault_nonzero = int(info.get('semantic_post_fault_nonzero', 0))
    semantic_post_fault_switch = int(info.get('semantic_post_fault_switch', 0))
    semantic_upgrade_collect_count = int(info.get('semantic_upgrade_collect_count', 0))
    semantic_retry_craft_count = int(info.get('semantic_retry_craft_count', 0))
    semantic_relocate_station_count = int(info.get('semantic_relocate_station_count', 0))
    semantic_valid_progress_count = int(info.get('semantic_valid_progress_count', 0))
    semantic_station_reuse_count = int(info.get('semantic_station_reuse_count', 0))
    semantic_delayed_after_use_count = int(info.get('semantic_delayed_after_use_count', 0))
    self._semantic_fault_episode_seen = max(
        int(self._semantic_fault_episode_seen), semantic_fault_episode, semantic_fault_applied)

    # reward fault
    reward, reward_fault_applied, reward_fault_type = self._apply_reward_fault(
        raw_reward, requested_action)

    # termination fault
    self._done, info, term_fault_applied, term_fault_type = self._apply_termination_fault(
        requested_action, self._done, info)

    # tester-oriented semantic reward shaping
    task_reward_for_agent = float(reward)
    if self._tester_reward_enabled:
      reward, info = self._tester_reward_from_semantics(
          task_reward_for_agent, requested_action, info)
    else:
      info['task_reward_raw'] = float(task_reward_for_agent)
      info['tester_bonus'] = 0.0
      info['tester_repeat_penalty'] = 0.0
      info['tester_reward'] = float(task_reward_for_agent)
      self._episode_task_reward += float(task_reward_for_agent)

    fault_applied = int(
        action_fault_applied or reward_fault_applied or term_fault_applied or semantic_fault_applied)

    if fault_applied:
      self._fault_count += 1
      subtype = None
      if semantic_fault_applied:
        subtype = semantic_fault_type
      elif action_fault_applied:
        subtype = action_fault_type
      elif reward_fault_applied:
        subtype = reward_fault_type
      elif term_fault_applied:
        subtype = term_fault_type
      if subtype is not None and subtype != 'none':
        self._subtype_apply_counts[subtype] += 1

    if semantic_fault_applied:
      fault_family = semantic_fault_family
      fault_type = semantic_fault_type
      fault_trigger = semantic_fault_trigger
      fault_severity = semantic_fault_severity
    elif action_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = action_fault_type
      fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
      fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0
    elif reward_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = reward_fault_type
      fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
      fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0
    elif term_fault_applied:
      fault_family = self._fault_spec['family']
      fault_type = term_fault_type
      fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
      fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0
    else:
      if semantic_fault_episode:
        fault_family = semantic_fault_family
        fault_type = 'none'
        fault_trigger = semantic_fault_trigger
        fault_severity = semantic_fault_severity
      else:
        fault_family = self._fault_spec['family'] if self._fault_spec is not None else 'none'
        fault_type = 'none'
        fault_trigger = self._fault_spec.get('trigger', 'none') if self._fault_spec is not None else 'none'
        fault_severity = float(self._fault_spec.get('severity', 0.0)) if self._fault_spec is not None else 0.0

    info['reward'] = float(reward)
    info['fault_applied'] = int(fault_applied)
    episode_fault_flag = int(self._fault_episode or semantic_fault_episode or self._semantic_fault_episode_seen)
    info['fault_episode'] = episode_fault_flag
    info['fault_family'] = fault_family
    info['fault_type'] = fault_type
    info['fault_trigger'] = fault_trigger
    info['fault_severity'] = float(fault_severity)
    info['semantic_fault_episode'] = int(semantic_fault_episode)
    info['semantic_fault_applied'] = int(semantic_fault_applied)
    info['semantic_fault_family'] = semantic_fault_family
    info['semantic_fault_type'] = semantic_fault_type
    info['semantic_fault_trigger'] = semantic_fault_trigger
    info['semantic_trigger_context'] = int(semantic_trigger_context)
    info['semantic_trigger_type'] = semantic_trigger_type
    info['semantic_trigger_count'] = int(semantic_trigger_count)
    info['semantic_first_trigger_step'] = int(semantic_first_trigger_step)
    info['semantic_ctx_upgrade_collect'] = int(semantic_ctx_upgrade_collect)
    info['semantic_ctx_retry_craft'] = int(semantic_ctx_retry_craft)
    info['semantic_ctx_relocate_station'] = int(semantic_ctx_relocate_station)
    info['semantic_post_fault_window'] = int(semantic_post_fault_window)
    info['semantic_post_fault_nonzero'] = int(semantic_post_fault_nonzero)
    info['semantic_post_fault_switch'] = int(semantic_post_fault_switch)
    info['semantic_upgrade_collect_count'] = int(semantic_upgrade_collect_count)
    info['semantic_retry_craft_count'] = int(semantic_retry_craft_count)
    info['semantic_relocate_station_count'] = int(semantic_relocate_station_count)
    info['semantic_valid_progress_count'] = int(semantic_valid_progress_count)
    info['semantic_station_reuse_count'] = int(semantic_station_reuse_count)
    info['semantic_delayed_after_use_count'] = int(semantic_delayed_after_use_count)
    info['requested_action'] = int(requested_action)
    info['executed_action'] = int(env_action)

    self._reward += reward
    self._length += 1
    self._global_step += 1

    self._update_episode_aggregation(
        fault_applied=fault_applied,
        fault_family=fault_family,
        fault_type=fault_type,
        fault_trigger=fault_trigger,
        semantic_fault_applied=semantic_fault_applied,
        semantic_trigger_count=semantic_trigger_count,
        semantic_first_trigger_step=semantic_first_trigger_step,
        semantic_ctx_upgrade_collect=semantic_upgrade_collect_count,
        semantic_ctx_retry_craft=semantic_retry_craft_count,
        semantic_ctx_relocate_station=semantic_relocate_station_count,
        semantic_valid_progress_count=semantic_valid_progress_count,
        semantic_station_reuse_count=semantic_station_reuse_count,
        semantic_delayed_after_use_count=semantic_delayed_after_use_count,
        semantic_post_fault_window=semantic_post_fault_window,
        semantic_post_fault_nonzero=semantic_post_fault_nonzero,
        semantic_post_fault_switch=semantic_post_fault_switch)

    if self._trace_path is not None:
      record = {
          'episode': int(self._episode),
          'episode_step': int(self._length),
          'global_step': int(self._global_step),
          'reward': float(reward),
          'fault_episode': int(episode_fault_flag),
          'semantic_fault_episode': int(self._semantic_fault_episode_seen or semantic_fault_episode),
          'fault_applied': int(fault_applied),
          'fault_family': fault_family,
          'fault_type': fault_type,
          'semantic_fault_type': semantic_fault_type,
          'task_reward_raw': float(info.get('task_reward_raw', 0.0)),
          'tester_bonus': float(info.get('tester_bonus', 0.0)),
          'tester_repeat_penalty': float(info.get('tester_repeat_penalty', 0.0)),
          'fault_trigger': info['fault_trigger'],
          'fault_severity': float(info['fault_severity']),
          'requested_action': int(requested_action),
          'executed_action': int(env_action),
          'semantic_trigger_count': int(semantic_trigger_count),
          'semantic_first_trigger_step': int(semantic_first_trigger_step),
          'semantic_ctx_upgrade_collect': int(semantic_ctx_upgrade_collect),
          'semantic_ctx_retry_craft': int(semantic_ctx_retry_craft),
          'semantic_ctx_relocate_station': int(semantic_ctx_relocate_station),
          'semantic_post_fault_window': int(semantic_post_fault_window),
          'semantic_post_fault_nonzero': int(semantic_post_fault_nonzero),
          'semantic_post_fault_switch': int(semantic_post_fault_switch),
          'fault_count_so_far': int(self._fault_count),
          'semantic_trigger_context': int(semantic_trigger_context),
          'semantic_trigger_type': semantic_trigger_type,
          'semantic_trigger_count': int(semantic_trigger_count),
          'semantic_first_trigger_step': int(semantic_first_trigger_step),
          'semantic_ctx_upgrade_collect': int(semantic_ctx_upgrade_collect),
          'semantic_ctx_retry_craft': int(semantic_ctx_retry_craft),
          'semantic_ctx_relocate_station': int(semantic_ctx_relocate_station),
          'semantic_post_fault_window': int(semantic_post_fault_window),
          'semantic_post_fault_nonzero': int(semantic_post_fault_nonzero),
          'semantic_post_fault_switch': int(semantic_post_fault_switch),
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

    if self._done:
      self._write_stats(self._length, self._reward, info)
      self._write_episode_summary(self._length, self._reward, info)

    # success window based on raw reward event
    if raw_reward > 0.0:
      self._after_positive_window = 3
      self._last_positive_reward_step = int(self._episode_step + 1)
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

    self._episode_step += 1
    self._record_context(image)

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

    revisit_context = 1 if self._recent_revisit_active() else 0
    delayed_switch_context = 1 if (
        self._delayed_switch_min <= (self._episode_step - self._last_positive_reward_step) <= self._delayed_switch_max
    ) else 0

    semantic_fault_applied = 0
    if info and 'semantic_fault_applied' in info:
      semantic_fault_applied = info['semantic_fault_applied']

    semantic_fault_episode = 0
    if info and 'semantic_fault_episode' in info:
      semantic_fault_episode = info['semantic_fault_episode']

    semantic_trigger_context = int(info.get('semantic_trigger_context', 0)) if info else 0
    semantic_trigger_count = int(info.get('semantic_trigger_count', 0)) if info else 0
    semantic_first_trigger_step = int(info.get('semantic_first_trigger_step', -1)) if info else -1
    semantic_ctx_upgrade_collect = int(info.get('semantic_ctx_upgrade_collect', 0)) if info else 0
    semantic_ctx_retry_craft = int(info.get('semantic_ctx_retry_craft', 0)) if info else 0
    semantic_ctx_relocate_station = int(info.get('semantic_ctx_relocate_station', 0)) if info else 0
    semantic_post_fault_window = int(info.get('semantic_post_fault_window', 0)) if info else 0
    semantic_post_fault_nonzero = int(info.get('semantic_post_fault_nonzero', 0)) if info else 0
    semantic_post_fault_switch = int(info.get('semantic_post_fault_switch', 0)) if info else 0
    semantic_upgrade_collect_count = int(info.get('semantic_upgrade_collect_count', 0)) if info else 0
    semantic_retry_craft_count = int(info.get('semantic_retry_craft_count', 0)) if info else 0
    semantic_relocate_station_count = int(info.get('semantic_relocate_station_count', 0)) if info else 0
    semantic_valid_progress_count = int(info.get('semantic_valid_progress_count', 0)) if info else 0
    semantic_station_reuse_count = int(info.get('semantic_station_reuse_count', 0)) if info else 0
    semantic_delayed_after_use_count = int(info.get('semantic_delayed_after_use_count', 0)) if info else 0
    task_reward_raw = float(info.get('task_reward_raw', 0.0)) if info else 0.0
    tester_bonus = float(info.get('tester_bonus', 0.0)) if info else 0.0
    tester_repeat_penalty = float(info.get('tester_repeat_penalty', 0.0)) if info else 0.0

    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(log_reward)},
        **{'log/fault_applied': np.int32(fault_applied)},
        **{'log/fault_episode': np.int32(fault_episode)},
        **{'log/semantic_fault_applied': np.int32(semantic_fault_applied)},
        **{'log/semantic_fault_episode': np.int32(semantic_fault_episode)},
        **{'log/semantic_trigger_context': np.int32(semantic_trigger_context)},
        **{'log/semantic_trigger_count': np.int32(semantic_trigger_count)},
        **{'log/semantic_first_trigger_step': np.int32(semantic_first_trigger_step)},
        **{'log/semantic_ctx_upgrade_collect': np.int32(semantic_ctx_upgrade_collect)},
        **{'log/semantic_ctx_retry_craft': np.int32(semantic_ctx_retry_craft)},
        **{'log/semantic_ctx_relocate_station': np.int32(semantic_ctx_relocate_station)},
        **{'log/semantic_post_fault_window': np.int32(semantic_post_fault_window)},
        **{'log/semantic_post_fault_nonzero': np.int32(semantic_post_fault_nonzero)},
        **{'log/semantic_post_fault_switch': np.int32(semantic_post_fault_switch)},
        **{'log/semantic_upgrade_collect_count': np.int32(semantic_upgrade_collect_count)},
        **{'log/semantic_retry_craft_count': np.int32(semantic_retry_craft_count)},
        **{'log/semantic_relocate_station_count': np.int32(semantic_relocate_station_count)},
        **{'log/semantic_valid_progress_count': np.int32(semantic_valid_progress_count)},
        **{'log/semantic_station_reuse_count': np.int32(semantic_station_reuse_count)},
        **{'log/semantic_delayed_after_use_count': np.int32(semantic_delayed_after_use_count)},
        **{'log/task_reward_raw': np.float32(task_reward_raw)},
        **{'log/tester_bonus': np.float32(tester_bonus)},
        **{'log/tester_repeat_penalty': np.float32(tester_repeat_penalty)},
        **{'log/revisit_context': np.int32(revisit_context)},
        **{'log/delayed_switch_context': np.int32(delayed_switch_context)},
        **{'log/global_step': np.int32(self._global_step)},
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
    if self._stats_path is None:
      return

    info = dict(info or {})
    achievements = info.get('achievements', {})
    stats = {
        'episode': self._episode,
        'length': int(length),
        'reward': round(float(reward), 1),
        'task_reward': round(float(info.get('task_reward_raw', self._episode_task_reward)), 6),
        'episode_tester_bonus': round(float(self._episode_tester_bonus), 6),
        'episode_repeat_penalty': round(float(self._episode_repeat_penalty), 6),
        'start_global_step': int(self._episode_start_global_step),
        'end_global_step': int(self._global_step),
        'fault_episode': int(info.get('fault_episode', 0)),
        'fault_count': int(self._fault_count),
        'fault_rate': float(self._fault_count / max(1, length)),
        'fault_family': self._ep_fault_family if getattr(self, '_ep_fault_family', 'none') != 'none' else info.get('fault_family', 'none'),
        'fault_type': self._ep_fault_type if getattr(self, '_ep_fault_type', 'none') != 'none' else info.get('fault_type', 'none'),
        'fault_trigger': self._ep_fault_trigger if getattr(self, '_ep_fault_trigger', 'none') != 'none' else info.get('fault_trigger', 'none'),
        'fault_severity': float(info.get('fault_severity', 0.0)),
        'fault_episode_prob': float(self._fault_episode_prob),
        'fault_cooldown_steps': int(self._fault_cooldown_steps),
        **{
            f'achievement_{k}': achievements.get(k, 0)
            for k in self._achievements
        },
    }
    with self._stats_path.open('a', encoding='utf-8') as f:
      f.write(json.dumps(stats) + '\n')
      f.flush()
