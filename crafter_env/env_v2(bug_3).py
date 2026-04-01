import collections
import os

import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen


# Gym is an optional dependency.
try:
  import gym
  DiscreteSpace = gym.spaces.Discrete
  BoxSpace = gym.spaces.Box
  DictSpace = gym.spaces.Dict
  BaseClass = gym.Env
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object


class Env(BaseClass):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, seed=None):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
    self._area = area
    self._view = view
    self._size = size
    self._reward = reward
    self._length = length
    self._seed = seed
    self._episode = 0
    self._world = engine.World(area, constants.materials, (12, 12))
    self._textures = engine.Textures(constants.root / 'assets')
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._local_view = engine.LocalView(
        self._world, self._textures, [view[0], view[1] - item_rows])
    self._item_view = engine.ItemView(
        self._textures, [view[0], item_rows])
    self._sem_view = engine.SemanticView(self._world, [
        objects.Player, objects.Cow, objects.Zombie,
        objects.Skeleton, objects.Arrow, objects.Plant])
    self._step = None
    self._player = None
    self._last_health = None
    self._unlocked = None

    # --------------------------------------------------
    # High-level semantic fault injection
    # --------------------------------------------------
    self._semantic_fault_sampler = bool(int(os.getenv('CRAFTER_SEMANTIC_FAULT_SAMPLER', '0')))
    self._semantic_fault_profile = os.getenv(
        'CRAFTER_SEMANTIC_FAULT_PROFILE',
        os.getenv('CRAFTER_FAULT_PROFILE', 'eval_holdout')).strip().lower()
    self._semantic_fault_ep_prob = float(os.getenv('CRAFTER_SEMANTIC_FAULT_EP_PROB', '0.5'))
    self._semantic_retry_gap = int(os.getenv('CRAFTER_SEMANTIC_RETRY_GAP', '40'))
    self._semantic_verbose = bool(int(os.getenv('CRAFTER_SEMANTIC_FAULT_VERBOSE', '0')))

    default_subtypes = {
        'train': '',
        'eval_seen': '',
        'eval_holdout': (
            'tool_collect_desync_on_upgrade,'
            'craft_result_missing_on_retry,'
            'station_place_ghost_on_relocate'
        ),
    }
    subtype_raw = os.getenv(
        'CRAFTER_SEMANTIC_SUBTYPES',
        default_subtypes.get(self._semantic_fault_profile, default_subtypes['eval_holdout']))
    self._semantic_subtypes = [x.strip() for x in subtype_raw.split(',') if x.strip()]

    self._semantic_fault_episode = 0
    self._semantic_fault_spec = None
    self._semantic_fault_count = 0

    self._last_successful_make_step = {}
    self._successful_place_counts = collections.defaultdict(int)
    self._pending_tool_collect_bug = None

    # semantic metrics
    self._semantic_trigger_count = 0
    self._semantic_first_trigger_step = -1
    self._semantic_last_fault_step = -10**9
    self._semantic_post_window = int(os.getenv('CRAFTER_SEMANTIC_POST_WINDOW', '10'))
    self._last_action_name = 'noop'
    self._semantic_context_counts = collections.defaultdict(int)

    # Some libraries expect these attributes to be set.
    self.reward_range = None
    self.metadata = None

  @property
  def observation_space(self):
    return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)

  @property
  def action_space(self):
    return DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def _sample_semantic_fault_spec(self):
    self._semantic_fault_episode = 0
    self._semantic_fault_spec = None
    self._semantic_fault_count = 0

    if not self._semantic_fault_sampler:
      return
    if not self._semantic_subtypes:
      return
    if self._world.random.uniform() >= self._semantic_fault_ep_prob:
      return

    idx = self._world.random.randint(0, len(self._semantic_subtypes))
    subtype = self._semantic_subtypes[idx]
    self._semantic_fault_spec = {
        'family': 'semantic_high_level',
        'type': subtype,
        'severity': 1.0,
        'trigger': self._semantic_trigger_name(subtype),
    }
    self._semantic_fault_episode = 1

    if self._semantic_verbose:
      print(f'[SemanticFaultSpec] ep={self._episode} profile={self._semantic_fault_profile} spec={self._semantic_fault_spec}')

  def _semantic_trigger_name(self, subtype):
    mapping = {
        'tool_collect_desync_on_upgrade': 'first_valid_collect_after_pickaxe_upgrade',
        'craft_result_missing_on_retry': 'repeat_craft_same_recipe_within_episode',
        'station_place_ghost_on_relocate': 'second_or_later_table_or_furnace_placement',
    }
    return mapping.get(subtype, 'custom')

  def _make_target(self, pos, facing):
    pos = np.array(pos)
    facing = np.array(facing)
    return tuple((pos + facing).tolist())

  def _snapshot_state(self):
    return {
        'inventory': self._player.inventory.copy(),
        'achievements': self._player.achievements.copy(),
        'pos': np.array(self._player.pos).copy(),
        'facing': tuple(self._player.facing),
    }

  def _inventory_delta(self, prev_inv):
    delta = {}
    for key, value in self._player.inventory.items():
      before = prev_inv.get(key, 0)
      diff = value - before
      if diff != 0:
        delta[key] = diff
    return delta

  def _achievement_delta(self, prev_ach):
    delta = {}
    for key, value in self._player.achievements.items():
      before = prev_ach.get(key, 0)
      diff = value - before
      if diff != 0:
        delta[key] = diff
    return delta

  def _arm_tool_collect_bug(self, action_name, inventory_delta):
    if action_name not in ('make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe'):
      return
    crafted_name = action_name[len('make_'):]
    if inventory_delta.get(crafted_name, 0) <= 0:
      return

    targets = {
        'wood_pickaxe': {'stone', 'coal'},
        'stone_pickaxe': {'iron'},
        'iron_pickaxe': {'diamond'},
    }.get(crafted_name, set())
    if not targets:
      return
    self._pending_tool_collect_bug = {
        'crafted_name': crafted_name,
        'targets': set(targets),
        'armed_step': int(self._step),
    }

  def _mark_semantic_trigger(self, result, trigger_type):
    result['trigger_context'] = 1
    result['trigger_type'] = trigger_type
    self._semantic_trigger_count += 1
    self._semantic_context_counts[trigger_type] += 1
    if self._semantic_first_trigger_step < 0:
      self._semantic_first_trigger_step = int(self._step)
    result['trigger_count'] = int(self._semantic_trigger_count)
    result['first_trigger_step'] = int(self._semantic_first_trigger_step)


  def _apply_semantic_fault(
      self,
      action_name,
      prev_state,
      pre_target,
      pre_target_material,
      pre_target_obj,
      pre_target_achievement_name,
      inventory_delta,
      achievement_delta,
  ):
    result = {
        'fault_applied': 0,
        'fault_type': 'none',
        'fault_trigger': 'none',
        'trigger_context': 0,
        'trigger_type': 'none',
        'trigger_count': int(self._semantic_trigger_count),
        'first_trigger_step': int(self._semantic_first_trigger_step),
        'ctx_upgrade_collect': 0,
        'ctx_retry_craft': 0,
        'ctx_relocate_station': 0,
    }

    if self._semantic_fault_spec is None:
      # Even if no fault spec was sampled, keep arming progression context so
      # later semantic bug profiles behave consistently when enabled in future episodes.
      self._arm_tool_collect_bug(action_name, inventory_delta)
      if action_name.startswith('make_') and inventory_delta.get(action_name[len('make_'):], 0) > 0:
        self._last_successful_make_step[action_name[len('make_'):]] = int(self._step)
      if action_name in ('place_table', 'place_furnace') and pre_target_material != action_name[len('place_'):] and self._world[pre_target][0] == action_name[len('place_'):]:
        self._successful_place_counts[action_name[len('place_'):]] += 1
      return result

    subtype = self._semantic_fault_spec['type']

    # --------------------------------------------------
    # Bug 1: first valid collect after pickaxe upgrade gives no resource.
    # World/resource state still changes, so this becomes an inventory desync.
    # --------------------------------------------------
    if subtype == 'tool_collect_desync_on_upgrade':
      self._arm_tool_collect_bug(action_name, inventory_delta)
      if action_name == 'do' and self._pending_tool_collect_bug is not None:
        if pre_target_material in self._pending_tool_collect_bug['targets']:
          collect_info = constants.collect.get(pre_target_material, None)
          if collect_info:
            result['ctx_upgrade_collect'] = 1
            self._mark_semantic_trigger(result, 'upgrade_collect')
            fired = False
            for item, amount in collect_info['receive'].items():
              if inventory_delta.get(item, 0) > 0:
                self._player.inventory[item] = prev_state['inventory'][item]
                ach_name = f'collect_{item}'
                if ach_name in self._player.achievements:
                  self._player.achievements[ach_name] = prev_state['achievements'][ach_name]
                fired = True
            if fired:
              result['fault_applied'] = 1
              result['fault_type'] = subtype
              result['fault_trigger'] = self._semantic_fault_spec['trigger']
              self._semantic_fault_count += 1
              self._semantic_last_fault_step = int(self._step)
              self._pending_tool_collect_bug = None
              return result
      # Keep pending bug armed until matching collect happens.

    # --------------------------------------------------
    # Bug 2: repeated crafting of the same recipe consumes inputs but no output.
    # --------------------------------------------------
    elif subtype == 'craft_result_missing_on_retry':
      if action_name.startswith('make_'):
        item = action_name[len('make_'):]
        crafted = inventory_delta.get(item, 0) > 0
        if crafted:
          if item in self._last_successful_make_step:
            result['ctx_retry_craft'] = 1
            self._mark_semantic_trigger(result, 'retry_craft')
            self._player.inventory[item] = prev_state['inventory'][item]
            ach_name = f'make_{item}'
            if ach_name in self._player.achievements:
              self._player.achievements[ach_name] = prev_state['achievements'][ach_name]
            result['fault_applied'] = 1
            result['fault_type'] = subtype
            result['fault_trigger'] = self._semantic_fault_spec['trigger']
            self._semantic_fault_count += 1
            self._semantic_last_fault_step = int(self._step)
            return result
          self._last_successful_make_step[item] = int(self._step)
          self._arm_tool_collect_bug(action_name, inventory_delta)

    # --------------------------------------------------
    # Bug 3: second and later table/furnace placement becomes a ghost placement.
    # Resources are consumed, but the station disappears immediately.
    # --------------------------------------------------
    elif subtype == 'station_place_ghost_on_relocate':
      if action_name in ('place_table', 'place_furnace'):
        place_name = action_name[len('place_'):]
        post_material, post_obj = self._world[pre_target]
        placed_success = (post_material == place_name and pre_target_obj is None)
        if placed_success:
          if self._successful_place_counts[place_name] >= 1:
            result['ctx_relocate_station'] = 1
            self._mark_semantic_trigger(result, 'relocate_station')
            self._world[pre_target] = pre_target_material
            result['fault_applied'] = 1
            result['fault_type'] = subtype
            result['fault_trigger'] = self._semantic_fault_spec['trigger']
            self._semantic_fault_count += 1
            self._semantic_last_fault_step = int(self._step)
            self._successful_place_counts[place_name] += 1
            return result
          self._successful_place_counts[place_name] += 1

    # Context bookkeeping even if fault did not fire.
    if action_name.startswith('make_') and inventory_delta.get(action_name[len('make_'):], 0) > 0:
      self._arm_tool_collect_bug(action_name, inventory_delta)
      self._last_successful_make_step[action_name[len('make_'):]] = int(self._step)

    return result

  def reset(self):
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    self._episode += 1
    self._step = 0
    self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._update_time()
    self._player = objects.Player(self._world, center)
    self._last_health = self._player.health
    self._world.add(self._player)
    self._unlocked = set()

    self._last_successful_make_step = {}
    self._successful_place_counts = collections.defaultdict(int)
    self._pending_tool_collect_bug = None
    self._semantic_trigger_count = 0
    self._semantic_first_trigger_step = -1
    self._semantic_last_fault_step = -10**9
    self._last_action_name = 'noop'
    self._semantic_context_counts = collections.defaultdict(int)

    worldgen.generate_world(self._world, self._player)
    self._sample_semantic_fault_spec()
    return self._obs()

  def step(self, action):
    self._step += 1
    self._update_time()

    prev_state = self._snapshot_state()
    action_name = constants.actions[action]
    pre_target = self._make_target(prev_state['pos'], prev_state['facing'])
    pre_target_material, pre_target_obj = self._world[pre_target]

    self._player.action = action_name
    for obj in self._world.objects:
      if self._player.distance(obj) < 2 * max(self._view):
        obj.update()
    if self._step % 10 == 0:
      for chunk, objs in self._world.chunks.items():
        self._balance_chunk(chunk, objs)

    inventory_delta = self._inventory_delta(prev_state['inventory'])
    achievement_delta = self._achievement_delta(prev_state['achievements'])

    semantic_fault = self._apply_semantic_fault(
        action_name=action_name,
        prev_state=prev_state,
        pre_target=pre_target,
        pre_target_material=pre_target_material,
        pre_target_obj=pre_target_obj,
        pre_target_achievement_name=None,
        inventory_delta=inventory_delta,
        achievement_delta=achievement_delta,
    )

    post_fault_window = int(0 < (self._step - self._semantic_last_fault_step) <= self._semantic_post_window)
    post_fault_nonzero = int(post_fault_window and action_name != 'noop')
    post_fault_switch = int(post_fault_window and action_name != 'noop' and action_name != self._last_action_name)

    obs = self._obs()
    reward = (self._player.health - self._last_health) / 10
    self._last_health = self._player.health

    unlocked = {
        name for name, count in self._player.achievements.items()
        if count > 0 and name not in self._unlocked}
    if unlocked:
      self._unlocked |= unlocked
      reward += 1.0

    dead = self._player.health <= 0
    over = self._length and self._step >= self._length
    done = dead or over
    info = {
        'inventory': self._player.inventory.copy(),
        'achievements': self._player.achievements.copy(),
        'discount': 1 - float(dead),
        'semantic': self._sem_view(),
        'player_pos': self._player.pos,
        'reward': reward,
        'semantic_fault_episode': int(self._semantic_fault_episode),
        'semantic_fault_applied': int(semantic_fault['fault_applied']),
        'semantic_fault_family': 'semantic_high_level' if self._semantic_fault_spec else 'none',
        'semantic_fault_type': semantic_fault['fault_type'],
        'semantic_fault_trigger': semantic_fault['fault_trigger'],
        'semantic_fault_count': int(self._semantic_fault_count),
        'semantic_fault_profile': self._semantic_fault_profile,
        'semantic_trigger_context': int(semantic_fault['trigger_context']),
        'semantic_trigger_type': semantic_fault['trigger_type'],
        'semantic_trigger_count': int(self._semantic_trigger_count),
        'semantic_first_trigger_step': int(self._semantic_first_trigger_step),
        'semantic_ctx_upgrade_collect': int(semantic_fault['ctx_upgrade_collect']),
        'semantic_ctx_retry_craft': int(semantic_fault['ctx_retry_craft']),
        'semantic_ctx_relocate_station': int(semantic_fault['ctx_relocate_station']),
        'semantic_post_fault_window': int(post_fault_window),
        'semantic_post_fault_nonzero': int(post_fault_nonzero),
        'semantic_post_fault_switch': int(post_fault_switch),
        'semantic_upgrade_collect_count': int(self._semantic_context_counts.get('upgrade_collect', 0)),
        'semantic_retry_craft_count': int(self._semantic_context_counts.get('retry_craft', 0)),
        'semantic_relocate_station_count': int(self._semantic_context_counts.get('relocate_station', 0)),
    }
    if not self._reward:
      reward = 0.0
    self._last_action_name = action_name
    return obs, reward, done, info

  def render(self, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    local_view = self._local_view(self._player, unit)
    item_view = self._item_view(self._player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))

  def _obs(self):
    return self.render()

  def _update_time(self):
    # https://www.desmos.com/calculator/grfbc6rs3h
    progress = (self._step / 300) % 1 + 0.3
    daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    self._world.daylight = daylight

  def _balance_chunk(self, chunk, objs):
    light = self._world.daylight
    self._balance_object(
        chunk, objs, objects.Zombie, 'grass', 6, 0, 0.3, 0.4,
        lambda pos: objects.Zombie(self._world, pos, self._player),
        lambda num, space: (
            0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
    self._balance_object(
        chunk, objs, objects.Skeleton, 'path', 7, 7, 0.1, 0.1,
        lambda pos: objects.Skeleton(self._world, pos, self._player),
        lambda num, space: (0 if space < 6 else 1, 2))
    self._balance_object(
        chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
        lambda pos: objects.Cow(self._world, pos),
        lambda num, space: (0 if space < 30 else 1, 1.5 + light))

  def _balance_object(
      self, chunk, objs, cls, material, span_dist, despan_dist,
      spawn_prob, despawn_prob, ctor, target_fn):
    xmin, xmax, ymin, ymax = chunk
    random = self._world.random
    creatures = [obj for obj in objs if isinstance(obj, cls)]
    mask = self._world.mask(*chunk, material)
    target_min, target_max = target_fn(len(creatures), mask.sum())
    if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
      xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
      ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
      xs, ys = xs[mask], ys[mask]
      i = random.randint(0, len(xs))
      pos = np.array((xs[i], ys[i]))
      empty = self._world[pos][1] is None
      away = self._player.distance(pos) >= span_dist
      if empty and away:
        self._world.add(ctor(pos))
    elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
      obj = creatures[random.randint(0, len(creatures))]
      away = self._player.distance(obj.pos) >= despan_dist
      if away:
        self._world.remove(obj)
