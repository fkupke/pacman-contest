# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import time
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveFroschAgent', second='DefensiveFroschAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


#############
# My Agents #
#############
class FroschCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        self.cooldown = 0
        self.was_pacman = False
        self.fled = False
        self.avoid_pos = None

        # position of own side
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        # the middle of the "board" is half of the width
        middle = width // 2
        # if we are on the red team, our half is everything from 0 to middle -1
        if self.red:
            self.mid_x = middle - 1
        # if we are on the blue team, our half is everything from middle to width -1
        else:
            self.mid_x = middle
        # the boundary between the two sides is in the middle
        # except for the parts where there are walls
        self.boundary = [(self.mid_x, y) for y in range(height) if not walls[self.mid_x][y]] 

    def choose_action(self, game_state):
        self.update_cooldown(game_state)
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)

        ghosts = self.get_dangerous_ghosts(game_state)
        food = self.get_food(game_state).as_list()
        food_left = len(food)

    
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        if self.cooldown > 0 and not my_state.is_pacman and food:
            plan = self.astar(game_state, my_pos, food, ghosts)
            if plan:
                return plan[0]

       
        if my_state.is_pacman and self.should_return(game_state):
            if ghosts:
                self.fled = True
            plan = self.astar(game_state, my_pos, self.boundary, ghosts)
            if plan:
                return plan[0]


        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)

        # if we are carrying food, 
        # the distance to our team's side should be considered,
        # otherwise it doesn't matter
        if game_state.num_carrying > 0:
            features['distance_to_own_side'] = self.get_distance_to_own_side
        else:
            features['distance_to_own_side'] = 0

        foods = self.get_food(successor).as_list()
        pos = game_state.get_position()
        food_distances = []

        # go through all foods and find the nearest one
        for food in foods:
            food_distances.append(self.get_maze_distance(pos, food))
        features['nearest_food'] = min(food_distances)

        # going through the visible ghosts
        ghosts = self.get_dangerous_ghosts(game_state)
        for ghost in ghosts:
            ghost_position = ghost.get_position()
            distance = self.get_maze_distance(pos, ghost_position)
            timer = ghost.scared_timer
            if distance <= timer:
                features['near_ghost'] = distance
            else:
                features['near_ghost'] = 0

        if (pos in self.find_dead_ends and features["near_ghost"] > 0):
            features['dead_end'] = 1
        else:
            features['dead_end'] = 0

        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        if self.should_return(game_state):
        # food doesnt matter anymore, we need to return to our side asap
            return {'successor_score': 100, 'nearest_food': 0,
                    'distance_to_own_side': -10, 'near_ghost': 10, 'dead_end': 7}
        
        return {'successor_score': 100, 'nearest_food': -2,
                'distance_to_own_side': 0, 'near_ghost': 7, 'dead_end': 3}

    def get_distance_to_own_side(self, game_state):
        # own position
        pos = game_state.get_agent_position(self.index)
        # calculate the nearest boundary
        distance_to_own_side = min(self.get_maze_distance(pos, boundary) 
                                    for boundary in self.boundary)
        return distance_to_own_side

    def get_dangerous_ghosts(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        ghosts = []
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            pos = enemy.get_position()    
            # check if enemy is visible and a ghost    
            if pos is None or enemy.is_pacman:
                continue  
            # check is enemy is scared (for a long time to come)                  
            if enemy.scared_timer >= self.get_maze_distance(my_pos, pos):
                continue                     
            ghosts.append(enemy)
        return ghosts
        
    def should_return(self, game_state):
        pos = game_state.get_agent_position(self.index)
        state = game_state.get_agent_state(self.index)
        # how much food we carry
        carrying = state.num_carrying
        # if we dont carry any, we dont need to return
        if carrying == 0:
            return False

        # distance to our team's side
        home_distance = self.get_distance_to_own_side(game_state)

        ghosts = self.get_dangerous_ghosts(game_state)
        for ghost in ghosts:
            ghost_pos = ghost.get_position()
            distance = self.get_maze_distance(pos, ghost_pos)
            timer = ghost.scared_timer
            # checking if there is a visible ghost near us with a low timer
            # taking into consideration how much food we carry atm
            # if there is a ghost near, we should retreat now!!
            if (distance*carrying) <= 20 and timer <= distance:
                return True            

        # if we carry a lot of food and our home is far away, we should return
        # otherwise we can keep exploring 
        return carrying >= 5 and home_distance > 5


    # function to find dead ends
    def find_dead_ends(self, game_state):
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        dead_ends = set()
        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue
                wall_count = sum(walls[x + dx][y + dy] 
                                 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)])
                if wall_count == 3:
                    dead_ends.add((x, y))
        return dead_ends

    def is_enemy_side(self, pos):
        return pos[0] > self.mix_x if self.red else pos[0] < self.mid_x

    # this function updates the cooldown so that after pacman fled an enemy
    # and arrived at its own side, it does not immediately return to the 
    # other side again, because most of the time,
    # the enemy is still waiting there
    def update_cooldown(self, game_state):
        state = game_state.get_agent_state(self.index)
        pos = game_state.get_agent_position(self.index)
        # this checks if the agent was pacman before and now is a ghost,
        # but didn't spawn back at the start
        # so basically it checks if the agent crossed the border to its side
        if self.was_pacman and not state.is_pacman:
            if pos != self.start and self.fled:
                self.cooldown = 10
                self.avoid_pos = pos
            self.fled = False
        elif self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown == 0:
                self.avoid_pos = None
        self.was_pacman = state.is_pacman


    ################
    # path-finding #
    ################

    MOVES = [(Directions.NORTH, (0, 1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST,  (1, 0)),
            (Directions.WEST,  (-1, 0))]

    # calculate cost for an action
    def step_cost(self, game_state, pos, ghosts):
        cost = 0
        for ghost in ghosts:
            ghost_pos = ghost.get_position()
            distance = self.get_maze_distance(pos, ghost_pos)
            dead_ends = self.find_dead_ends(game_state)
            # the nearer the ghost is, the higher the cost of going there
            if distance <= 4:
                cost += (5 - distance) * 5   
            # if the "node" is a dead end, taking it is risky   
            if ghosts and pos in dead_ends:
                cost += 20   
            # if we are in cooldown-mode, 
            # we should avoid the place where we crossed the border
            # because there might be a ghost waiting there  
            if self.cooldown > 0 and self.avoid_pos is not None and self.is_enemy_side(pos):
                d = self.get_maze_distance(pos, self.avoid_pos)
                if d <= 4:
                    cost += (5 - d) * 4
        return cost

    def heuristic(self, pos, goals):
        # find the nearest goal "node" 
        # (so most of the time the nearest "node" on our home side)
        return min(self.get_maze_distance(pos, goal) for goal in goals)

    def astar(self, game_state, start, goals, ghosts):
        walls = game_state.get_walls()
        goals = set(goals)

        prio_queue = util.PriorityQueue()
        visited = set()

        prio_queue.push((start, [], 0), 0)

        while not prio_queue.is_empty():
            pos, path, cost = prio_queue.pop()

            if pos in visited:
                continue
            visited.add(pos)

            if pos in goals:
                return path

            x, y = pos
            for direction, (dx, dy) in self.MOVES:
                nxt = (x + dx, y + dy)
                if walls[nxt[0]][nxt[1]] or nxt in visited:
                    continue
                new_cost = cost + 1 + self.step_cost(game_state, nxt, ghosts)
                estimate = self.heuristic(pos, goals)
                prio_queue.push((nxt, path + [direction], new_cost),
                                new_cost + estimate)
        return []
            
    
class OffensiveFroschAgent(FroschCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        
        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True, but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveFroschAgent(FroschCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}     