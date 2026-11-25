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
        self.last_eaten = None

        self.dead_ends = self.find_dead_ends(game_state)

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
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

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
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

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
        return False


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
        return pos[0] > self.mid_x if self.red else pos[0] < self.mid_x

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
            # the nearer the ghost is, the higher the cost of going there
            if distance <= 4:
                cost += (5 - distance) * 5   
            # if the "node" is a dead end, taking it is risky   
            if ghosts and pos in self.dead_ends:
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
                estimate = self.heuristic(nxt, goals)
                prio_queue.push((nxt, path + [direction], new_cost),
                                new_cost + estimate)
        return []
            
    
class OffensiveFroschAgent(FroschCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def choose_action(self, game_state):
        self.update_cooldown(game_state)
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)

        ghosts = self.get_dangerous_ghosts(game_state)
        food = self.get_food(game_state).as_list()
        food_left = len(food)

        if my_state.is_pacman and ghosts:
            d = min(self.get_maze_distance(my_pos, g.get_position()) for g in ghosts)
            if d <= 5:
                self.fled = True
    
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
            print(self.index, my_pos, 'mode=food<2', self.cooldown)
            return best_action

        if self.cooldown > 0 and not my_state.is_pacman and food:
            plan = self.astar(game_state, my_pos, food, ghosts)
            if plan:
                print(self.index, my_pos, 'mode=cooldown', self.cooldown)
                return plan[0]

        
        if my_state.is_pacman and self.should_return(game_state):
            if ghosts:
                self.fled = True
            plan = self.astar(game_state, my_pos, self.boundary, ghosts)
            if plan:
                print(self.index, my_pos, 'mode=flee', self.cooldown)
                return plan[0]


        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        print(self.index, my_pos, 'mode=eval', self.cooldown)
        return random.choice(best_actions)

    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        pos = my_state.get_position()
        foods = self.get_food(successor).as_list()

        # the less food left, the better
        features['successor_score'] = -len(foods) 
        
        # go through all foods and find the closest one
        if foods:
            features['distance_to_food'] = min(
                self.get_maze_distance(pos, f) for f in foods)

        # ghosts only matter if we are pacman atm
        ghosts = self.get_dangerous_ghosts(successor) if my_state.is_pacman else []
        if ghosts:
            # find the closest ghost
            ghost_dist = min(self.get_maze_distance(pos, g.get_position())
                             for g in ghosts)
            # ghost next to us
            if ghost_dist <= 1:
                features['ghost_next'] = 1
            # ghost close to us
            if ghost_dist <= 5:
                features['ghost_close'] = 6 - ghost_dist
            # we are in a dead-end
            if pos in self.dead_ends:
                features['in_dead_end'] = 1
 
            capsules = self.get_capsules(successor)
            # find the closest capsule, we can use it if a ghost is near
            if capsules:
                features['distance_to_capsule'] = min(
                    self.get_maze_distance(pos, c) for c in capsules)
 
        
        carrying = my_state.num_carrying
        if carrying > 0 and my_state.is_pacman:
            # the more food we carry, the sooner we should return to our side
            features['return_home'] = carrying * self.get_distance_to_own_side(successor)
 
        # to punish stopping or going back (to stop pacman from oscillating)
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
 
        return features

    def get_weights(self, game_state, action):
        if self.should_return(game_state):
            # food doesnt matter anymore, we need to return to our side asap
            # or get a capsule
            return {
                    'successor_score': 100,
                    'distance_to_food': 0,
                    'distance_to_capsule': -10,
                    'ghost_adjacent': -1000,
                    'ghost_proximity': -50,
                    'in_dead_end': -50,
                    'return_home': -3,
                    'stop': -100,
                    'reverse': -2,
                    }
        return {
            'successor_score': 100,
            'distance_to_food': -3,
            'distance_to_capsule': -2,
            'ghost_adjacent': -1000,
            'ghost_proximity': -20,
            'in_dead_end': -20,
            'return_home': -0.5,
            'stop': -100,
            'reverse': -2,
        }

        
class DefensiveFroschAgent(FroschCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
 
    def choose_action(self, game_state):
        self.update_eaten_guess(game_state)
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)
 
    # finds the place where food on our side was eaten last
    # to try to track enemies we cannot see
    def update_eaten_guess(self, game_state):
        prev = self.get_previous_observation()
        if prev is not None:
            before = set(self.get_food_you_are_defending(prev).as_list())
            now = set(self.get_food_you_are_defending(game_state).as_list())
            eaten = before - now
            if eaten:
                self.last_eaten = next(iter(eaten))
        # if we went to the place and no one is there, forget about it
        my_pos = game_state.get_agent_position(self.index)
        if self.last_eaten is not None and my_pos == self.last_eaten:
            self.last_eaten = None
 
    # finds a patrol point close to our own food
    def get_patrol_target(self, game_state):
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return self.boundary[len(self.boundary) // 2]
        return min(self.boundary,
                   key=lambda b: sum(self.get_maze_distance(b, f) for f in food))

 
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
 
        # to stay on our side as defense
        features['on_defense'] = 0 if my_state.is_pacman else 1
 
        # number of visible enemies on our side
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(invaders)
 
        if invaders:
            # find the closest invader
            d = min(self.get_maze_distance(my_pos, e.get_position()) for e in invaders)
            if my_state.scared_timer > 0:
                # if we are scared, we shouldnt get close as to not get eaten
                features['scared_invader_dist'] = abs(d - 2)
            else:
                features['invader_distance'] = d
        elif self.last_eaten is not None:
            # if we dont see any invaders but food was eaten, we know someone is there
            features['distance_to_eaten'] = self.get_maze_distance(my_pos, self.last_eaten)
        else:
            # if there is no one visible and no one eating, we go on patrol
            features['patrol_distance'] = self.get_maze_distance(
                my_pos, self.get_patrol_target(game_state))

        # to stop pacman from oscillating
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
 
        return features
 
    def get_weights(self, game_state, action):
        return {
            'on_defense': 100,
            'num_invaders': -1000,
            'invader_distance': -10,
            'scared_invader_dist': -10,
            'distance_to_eaten': -5,
            'patrol_distance': -3,
            'stop': -100,
            'reverse': -2,
        }