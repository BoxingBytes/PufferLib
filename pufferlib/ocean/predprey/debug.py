"""Human-readable decoding of a single PredPrey observation for pdb."""
import numpy as np

# Single source of truth for observation layout; mirror any change to
# compute_observations() in predprey.h here. Window size is derived from the
# observation length, so changing `vision` needs no edit.
CELL_CHANNELS = ('terrain', 'item', 'entity', 'hp', 'food')
SCALARS = ('direction', 'r_norm', 'c_norm', 'fireplace_lit',
           'fire_time', 'wood_amt', 'chest_food', 'coldness')

TERRAIN = {0: 'Soil', 1: 'Hous', 2: 'Watr', 3: 'Gras'}
ITEM = {0: '-', 1: 'Wood', 2: 'Food', 3: 'Chest', 4: 'FireL', 5: 'Fire'}
DIRECTION = {0: 'DOWN', 1: 'UP', 2: 'RIGHT', 3: 'LEFT'}
ACTION = {0: 'DOWN', 1: 'UP', 2: 'RIGHT', 3: 'LEFT',
          4: 'NO_MOVE', 5: 'INTERACT', 6: 'EAT'}


def render_obs(obs, action=None, vision=None):
    """Return a formatted multi-line view of one PredPrey observation.

    Args:
        obs: 1-D observation for a single agent (numpy array or torch tensor).
        action: optional integer action to append.
        vision: optional expected vision radius; asserted against the layout.

    Returns:
        A string with the vision window (agent at center in brackets), a
        scalar row, and an optional action row.

    Raises:
        ValueError: if the length does not match CELL_CHANNELS + SCALARS.
    """
    if hasattr(obs, 'detach'):
        obs = obs.detach().cpu().numpy()
    obs = np.asarray(obs).reshape(-1).astype(float)
    n_cell, n_scalar = len(CELL_CHANNELS), len(SCALARS)

    n_grid = obs.size - n_scalar
    if n_grid <= 0 or n_grid % n_cell != 0:
        raise ValueError(
            f'obs length {obs.size} does not fit {n_cell} cell channels + '
            f'{n_scalar} scalars; update CELL_CHANNELS/SCALARS to match '
            f'compute_observations().')
    n_cells = n_grid // n_cell
    window = int(round(n_cells ** 0.5))
    if window * window != n_cells:
        raise ValueError(f'{n_cells} cells is not a square vision window '
                         f'(obs length {obs.size}).')
    if vision is not None and window != 2 * vision + 1:
        raise ValueError(f'window {window} != 2*vision+1 for vision={vision}.')

    grid = obs[:n_grid].reshape(window, window, n_cell)
    scalars = obs[n_grid:]
    center = window // 2

    def fmt_cell(c, is_center):
        terrain = TERRAIN.get(int(c[0]), f'?{int(c[0])}')
        item = ITEM.get(int(c[1]), f'?{int(c[1])}')
        ent = int(c[2])
        ent_s = '.' if ent < 0 else str(ent)
        hp_s = '--' if ent < 0 else f'{int(round(c[3] * 100))}'
        food_s = '--' if ent < 0 else f'{int(round(c[4] * 100))}'
        s = f'{terrain},{item},{ent_s},{hp_s},{food_s}'
        return f'[{s}]' if is_center else f' {s} '

    cells = [[fmt_cell(grid[r, cc], r == center and cc == center)
              for cc in range(window)] for r in range(window)]
    w = max(len(s) for row in cells for s in row)
    rows = [' | '.join(s.ljust(w) for s in row) for row in cells]

    bits = []
    for i, name in enumerate(SCALARS):
        v = scalars[i]
        if name == 'direction':
            bits.append(f'direction={DIRECTION.get(int(v), int(v))}')
        elif name == 'fireplace_lit':
            bits.append(f'fireplace_lit={int(v)}')
        else:
            bits.append(f'{name}={v:.3f}')

    out = [f'vision {window}x{window} (center [] = agent) '
           f'| cell = {",".join(CELL_CHANNELS)} (hp,food in %)']
    out += rows
    out.append('-' * len(rows[0]))
    out.append('  '.join(bits))
    if action is not None:
        out.append(f'action={ACTION.get(int(action), int(action))}')
    return '\n'.join(out)
