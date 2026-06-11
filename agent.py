import random
import numpy as np
from collections import deque

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.actions = {"STOP": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "BOMB": 5}
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.dir_to_action = {
            (-1, 0): self.actions["UP"], (1, 0): self.actions["DOWN"],
            (0, -1): self.actions["LEFT"], (0, 1): self.actions["RIGHT"]
        }

    def act(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        
        my_row, my_col, my_alive, my_bombs, my_radius = players[self.agent_id]
        if not my_alive: return self.actions["STOP"]

        enemy_positions = set()
        min_enemy_dist = np.inf
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                er, ec = int(p[0]), int(p[1])
                enemy_positions.add((er, ec))
                min_enemy_dist = min(min_enemy_dist, abs(my_row - er) + abs(my_col - ec))

        # UPGRADE 1: Dynamic Search Depth. 
        # If the map is dense (early game), keep search shallow to guarantee <100ms.
        # If open (late game), expand search to hunt across the map.
        current_max_depth = 300

        # 1. FAST DANGER MAP
        danger_map, has_bomb = self.build_danger_map(grid, players, bombs)
        in_danger = danger_map[my_row, my_col] != np.inf

        # 2. IMMEDIATE EVASION
        if in_danger:
            move = self.find_safe_move(my_row, my_col, grid, danger_map, has_bomb)
            return self.dir_to_action.get(move, self.actions["STOP"])

        # 3. SMART OFFENSE (Will not trap itself)
        if my_bombs > 0 and self.is_lucrative_bomb_position(my_row, my_col, grid, my_radius, enemy_positions):
            if self.can_survive_bomb_at(my_row, my_col, grid, danger_map, has_bomb, my_radius):
                return self.actions["BOMB"]

        # 4. TACTICAL HEURISTIC PATHFINDING
        move = self.find_best_weighted_target(
            my_row, my_col, grid, danger_map, has_bomb, enemy_positions, 
            my_radius, my_bombs, current_max_depth
        )
        if move is not None:
            return self.dir_to_action.get(move, self.actions["STOP"])

        # 5. SMART WANDER (UPGRADE 4: Anti-Stalking Dead-End Avoidance)
        safe_moves = []
        for dx, dy in self.directions:
            nx, ny = my_row + dx, my_col + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] in [0, 3, 4] and not has_bomb[nx, ny] and danger_map[nx, ny] == np.inf:
                    
                    # If an enemy is close, do NOT walk into a dead end (a tile with >= 3 walls/boxes)
                    if min_enemy_dist <= 5:
                        blocked_sides = sum(1 for ddx, ddy in self.directions 
                                            if not (0 <= nx+ddx < grid.shape[0] and 0 <= ny+ddy < grid.shape[1]) 
                                            or grid[nx+ddx, ny+ddy] in [1, 2, 5])
                        if blocked_sides >= 3:
                            continue # Skip this move, it's a trap
                            
                    safe_moves.append((dx, dy))
                    
        if safe_moves: return self.dir_to_action[random.choice(safe_moves)]
        
        return self.actions["STOP"]

    # -------------------------------------------------------------
    # CORE LOGIC
    # -------------------------------------------------------------

    def build_danger_map(self, grid, players, bombs):
        danger_map = np.full(grid.shape, np.inf)
        has_bomb = np.zeros(grid.shape, dtype=bool)
        bombs_info = [[int(b[0]), int(b[1]), int(b[2]), 1 + int(players[int(b[3])][4])] for b in bombs]
        
        changed = True
        while changed:
            changed = False
            for i, b1 in enumerate(bombs_info):
                for j, b2 in enumerate(bombs_info):
                    if i == j: continue
                    if b1[2] < b2[2]:
                        if b1[0] == b2[0] and abs(b1[1] - b2[1]) <= b1[3]:
                            if not any(grid[b1[0], col] in [1, 2] for col in range(min(b1[1], b2[1])+1, max(b1[1], b2[1]))):
                                b2[2] = b1[2]
                                changed = True
                        elif b1[1] == b2[1] and abs(b1[0] - b2[0]) <= b1[3]:
                            if not any(grid[row, b1[1]] in [1, 2] for row in range(min(b1[0], b2[0])+1, max(b1[0], b2[0]))):
                                b2[2] = b1[2]
                                changed = True

        for bx, by, timer, radius in bombs_info:
            danger_map[bx, by] = min(danger_map[bx, by], timer)
            has_bomb[bx, by] = True
            for dx, dy in self.directions:
                for r in range(1, radius + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if grid[nx, ny] == 1: break
                        danger_map[nx, ny] = min(danger_map[nx, ny], timer)
                        if grid[nx, ny] == 2: break
                    else: break
        return danger_map, has_bomb

    def find_safe_move(self, start_r, start_c, grid, danger_map, has_bomb):
        q = deque([(start_r, start_c, None, 0)])
        visited = set([(start_r, start_c)])
        best_move, best_danger = None, danger_map[start_r, start_c]
        
        while q:
            r, c, first_move, dist = q.popleft()
            if danger_map[r, c] == np.inf: return first_move
            if danger_map[r, c] > best_danger:
                best_danger = danger_map[r, c]
                best_move = first_move
            
            for dx, dy in self.directions:
                nr, nc = r + dx, c + dy
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if (nr, nc) not in visited and grid[nr, nc] not in [1, 2] and not has_bomb[nr, nc]:
                        if dist + 1 < danger_map[nr, nc]:
                            visited.add((nr, nc))
                            q.append((nr, nc, first_move if first_move else (dx, dy), dist + 1))
        return best_move

    def is_lucrative_bomb_position(self, r, c, grid, my_radius, enemy_positions):
        for dx, dy in self.directions:
            for dist in range(1, my_radius + 2):
                nx, ny = r + dx * dist, c + dy * dist
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if grid[nx, ny] == 1: break 
                    if grid[nx, ny] == 2 or (nx, ny) in enemy_positions: return True
                else: break
        return False

    def can_survive_bomb_at(self, r, c, grid, danger_map, has_bomb, my_radius):
        sim_danger = danger_map.copy()
        sim_has_bomb = has_bomb.copy()
        sim_has_bomb[r, c] = True
        
        explode_tick = min(7, sim_danger[r, c])
        sim_danger[r, c] = explode_tick
        
        for dx, dy in self.directions:
            for dist in range(1, my_radius + 2):
                nx, ny = r + dx * dist, c + dy * dist
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    if grid[nx, ny] == 1: break
                    sim_danger[nx, ny] = min(sim_danger[nx, ny], explode_tick)
                    if grid[nx, ny] == 2: break
                else: break
                    
        q = deque([(r, c, 0)])
        visited = set([(r, c, 0)])
        
        while q:
            curr_r, curr_c, tick = q.popleft()
            if sim_danger[curr_r, curr_c] == np.inf or tick > 7: return True
                
            next_tick = tick + 1
            if next_tick < sim_danger[curr_r, curr_c]:
                if (curr_r, curr_c, next_tick) not in visited:
                    visited.add((curr_r, curr_c, next_tick))
                    q.append((curr_r, curr_c, next_tick))
            
            for dx, dy in self.directions:
                nr, nc = curr_r + dx, curr_c + dy
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] not in [1, 2]:
                        if not sim_has_bomb[nr, nc] or (nr == r and nc == c and curr_r == r and curr_c == c):
                            if next_tick < sim_danger[nr, nc]:
                                if (nr, nc, next_tick) not in visited:
                                    visited.add((nr, nc, next_tick))
                                    q.append((nr, nc, next_tick))
        return False

    def find_best_weighted_target(self, start_r, start_c, grid, danger_map, has_bomb, enemy_positions, my_radius, my_bombs, max_depth):
        """UPGRADE 3: Evaluates ALL targets within radius and picks the mathematically optimal one."""
        q = deque([(start_r, start_c, None, 0)])
        visited = set([(start_r, start_c)])
        
        best_move = None
        best_score = -np.inf
        
        while q:
            r, c, first_move, dist = q.popleft()
            
            if dist > max_depth: continue
            
            if dist > 0:
                score = 0
                # Reward picking up items immediately
                if grid[r, c] in [3, 4]:
                    score = 500 - (dist * 5)
                
                # UPGRADE 2: Ammo Awareness (Only hunt boxes/enemies if we can shoot)
                elif my_bombs > 0 and self.is_lucrative_bomb_position(r, c, grid, my_radius, enemy_positions):
                    # Check if this position threatens an enemy (high value) or just a box (medium value)
                    threatens_enemy = any(abs(r - er) + abs(c - ec) <= my_radius for er, ec in enemy_positions)
                    if threatens_enemy:
                        score = 800 - (dist * 10)
                    else:
                        score = 200 - (dist * 10)
                        
                # Update optimal route if this path scores higher
                if score > best_score:
                    best_score = score
                    best_move = first_move
                    
            for dx, dy in self.directions:
                nr, nc = r + dx, c + dy
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if (nr, nc) not in visited:
                        if grid[nr, nc] in [0, 3, 4] and not has_bomb[nr, nc] and danger_map[nr, nc] == np.inf:
                            visited.add((nr, nc))
                            q.append((nr, nc, first_move if first_move else (dx, dy), dist + 1))
                            
        return best_move