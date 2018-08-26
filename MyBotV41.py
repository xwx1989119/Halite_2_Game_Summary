"""
Welcome to your first Halite-II bot!
This bot's name is Settler. It's purpose is simple (don't expect it to win complex games :) ):
1. Initialize game
2. If a ship is not docked and there are unowned planets
2.a. Try to Dock in the planet if close enough
2.b If not, go towards the planet
Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need
to log anything use the logging module.
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
# Then let's import the logging module so we can print out information
import logging
import time
import csv
import numpy as np
from utl import *
import heapq
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import logging

np.seterr(divide='ignore')

logger = logging.getLogger('Stategy_log')
logger2 = logging.getLogger('Anti_collision_Log')
logger2.setLevel(logging.INFO)
logger.setLevel(logging.INFO)


class Bot:
    def __init__(self, name):
        self._name = name
        self.prev_assignment = []
        self.planet_status = []
        self.planet_heap_fleet1 = []
        self.ship_heaps_fleet1 = []
        self.planet_space_heap_fleet1 = []
        self.planet_heap_fleet2 = []
        self.ship_heaps_fleet2 = []
        self.planet_space_heap_fleet2 = []
        self.fleet_status = []
        self.rabbit = []
        self.turns = 0
        self.player_num = 2
        self.fleet_size = 1
        self.defense_fleet_size = 0
        self.all_planet_locations = []
        np.seterr(all='ignore')

    def assess_defense_priority(self,
                                health_current,
                                health_previous,
                                range25_current,
                                range70_current,
                                range25_preivous,
                                range70_previous):

        """
        this function is to assess the priority of planet defense strategy.
        generally, if a planet is under attack, we assume following will happen:
        1. surrounding enemy ship number will increase
        2. docked ship health will decrease
        :param health_current: current docking ships total health
        :param health_previous: previous docking ships total health
        :param range25_current: current range 25 enemy ships count
        :param range70_current: previous range 70 enemy ships count
        :param range25_preivous: current range 25 enemy ships count
        :param range70_previous: previous range 70 enemy ships count
        :return: defense periority
        """

        def_priority = 0
        if health_current - health_previous < 0:
            def_priority = 5

        if range25_preivous == 0 and range25_current != 0:
            def_priority = 4
        elif range70_previous == 0:
            def_priority = 0
        elif range25_current == 0 and range25_preivous == 0:
            def_priority = 0
        elif (range70_current - range70_previous > 1) \
                and (range70_current / range70_previous > 1.2)\
                and (range25_current / range25_preivous <= 1.3):
            def_priority = 2
        elif (range70_current / range70_previous > 1.2)  \
                and (range70_current - range70_previous > 2)\
                and (range25_current / range25_preivous > 1.3)\
                and (range25_current - range25_preivous > 2):
                def_priority = 4
        return def_priority

    def check_planet_status(self, game_map):
        plant_status = dict()
        for planet in game_map.all_planets():
            if planet.owner == game_map.get_me():
                # logger.info('currently updates the status of planet %f', planet.id)
                range_25_count = 0
                range_70_count = 0
                entities_by_distance = game_map.nearby_entities_by_distance(planet)
                for entity_distance in sorted(entities_by_distance):
                    current_ship_iter = (nearest_entity for nearest_entity in entities_by_distance[entity_distance] if
                                         (isinstance(nearest_entity, hlt.entity.Ship)))
                    nearest_enemy_ship = next(current_ship_iter, None)
                    if not nearest_enemy_ship:
                        continue
                    elif nearest_enemy_ship.owner != game_map.get_me() \
                            and nearest_enemy_ship.docking_status == nearest_enemy_ship.DockingStatus.UNDOCKED \
                            and entity_distance < 25.0:
                        range_25_count += 1
                        range_70_count += 1
                    elif nearest_enemy_ship.owner != game_map.get_me() \
                            and nearest_enemy_ship.docking_status == nearest_enemy_ship.DockingStatus.UNDOCKED \
                            and entity_distance < 70.0:
                        range_70_count += 1
                docked_ships = planet.all_docked_ships()
                docked_ships_count = len(docked_ships)
                docked_ships_health = 0
                for s in docked_ships:
                    docked_ships_health += s.health
                # logger.info('current planet %f has total health %f', planet.id, docked_ships_health)

                if planet.id in self.planet_status:
                    defense_priortiy = self.assess_defense_priority(docked_ships_health,
                                                                    self.planet_status[planet.id]['curr_dock_ship_health'],
                                                                    range_25_count,
                                                                    range_70_count,
                                                                    self.planet_status[planet.id]['Range_25_enemy'],
                                                                    self.planet_status[planet.id]['Range_70_enemy'])
                    curr_planet_status = {
                        'id': planet.id,
                        'curr_dock_ship_num': docked_ships_count,
                        'curr_dock_ship_health': docked_ships_health,
                        'previous_dock_ship_num': self.planet_status[planet.id]['curr_dock_ship_num'],
                        'previous_dock_ship_health': self.planet_status[planet.id]['curr_dock_ship_health'],
                        'Range_25_enemy': range_25_count,
                        'Range_70_enemy': range_70_count,
                        'Previous_25_enemy': self.planet_status[planet.id]['Range_25_enemy'],
                        'Previous_70_enemy': self.planet_status[planet.id]['Range_70_enemy'],
                        'defense_priority': defense_priortiy
                    }
                else:
                    curr_planet_status = {
                        'id': planet.id,
                        'curr_dock_ship_num': docked_ships_count,
                        'curr_dock_ship_health': docked_ships_health,
                        'previous_dock_ship_num': docked_ships_count,
                        'previous_dock_ship_health': docked_ships_health,
                        'Range_25_enemy': range_25_count,
                        'Range_70_enemy': range_70_count,
                        'Previous_25_enemy': range_25_count,
                        'Previous_70_enemy': range_70_count,
                        'defense_priority': 0
                    }
                plant_status[planet.id] = curr_planet_status
        return plant_status

    def enemy_health(self, game_map, planet, distance_range):
        """
        For each planet produce a set of features that we will feed to the neural net. We always return an array
        with PLANET_MAX_NUM rows - if planet is not present in the game, we set all featurse to 0.
        :param game_map: game map
        :param planet: target planet
        :param distance_range: the distance range of the assessment
        :return: number of the total enemey ship health
        """
        entities_by_distance = game_map.nearby_entities_by_distance(planet)
        enemey_health = 0
        distance_range = planet.radius + distance_range
        for distance_val in sorted(entities_by_distance):
            if distance_val <= distance_range:
                current_ship_iter = (nearest_entity for nearest_entity in entities_by_distance[distance_val] if
                                    (isinstance(nearest_entity, hlt.entity.Ship)))
                for ship in current_ship_iter:
                    if ship.owner != game_map.get_me():
                        enemey_health += ship.health
            else:
                break
        return enemey_health

    def local_navigate(self, game_map, start_of_round, ship, destination, speed):
        """
        Send a ship to its destination. Because "navigate" method in Halite API is expensive, we use that method only if
        we haven't used too much time yet.
        :param game_map: game map
        :param start_of_round: time (in seconds) between the Epoch and the start of this round
        :param ship: ship we want to send
        :param destination: destination to which we want to send the ship to
        :param speed: speed with which we would like to send the ship to its destination
        :return:
        """
        current_time = time.time()
        have_time = (current_time - start_of_round) < 1.2
        navigate_command = None
        if have_time:
            navigate_command = ship.navigate(destination,
                                             game_map,
                                             speed=speed,
                                             max_corrections=180,
                                             avoid_obstacles=True,
                                             angular_step=3)
        if navigate_command is None:
            logger.warning('not enough time')
            # ship.navigate may return None if it cannot find a path. In such a case we just thrust.
            dist = ship.calculate_distance_between(destination)
            speed = speed if (dist >= speed) else dist
            navigate_command = ship.thrust(speed, ship.calculate_angle_between(destination))
        return navigate_command

    def centeral_points(self, km,  fleets, game_map):
        x_central = 0.0
        y_central = 0.0
        center_points = np.array([0.0, 0.0])
        fleet_status_updates = np.array(self.fleet_status)
        if len(self.planet_status) > 0:
            for key in self.planet_status:
                x_central = x_central + game_map.get_planet(key).x
                y_central = y_central + game_map.get_planet(key).y
            x_central = x_central / len(self.planet_status)
            y_central = y_central / len(self.planet_status)
            all_center = np.array([x_central, y_central])
        else:
            all_center = [np.sum(fleet_status_updates[:, 1] * fleet_status_updates[:, 3])
                          / np.sum(fleet_status_updates[:, 3]),
                          np.sum(fleet_status_updates[:, 2] * fleet_status_updates[:, 3])
                          / np.sum(fleet_status_updates[:, 3])]

        if len(fleets) == 0:
            logger.info('Current run has no undock ships. will assign random central.')
        elif len(fleets) < 5:
            center_points = all_center
        else:
            # logger.info('current cords are %s', self.fleet_status[:,1:3])
            km.fit(self.fleet_status[:, 1:3])
            fleet_status_updates[:, 4] = km.predict(self.fleet_status[:, 1:3])
            results = km.cluster_centers_
            tri_distance = np.zeros(3)
            tri_distance[0] = distance(all_center[0], all_center[1], results[0, 0], results[0, 1])
            tri_distance[1] = distance(all_center[0], all_center[1], results[1, 0], results[1, 1])
            tri_distance[2] = distance(results[0, 0], results[0, 1], results[1, 0], results[1, 1])
            max_distance = np.max(tri_distance)

            if max_distance < game_map.width / 5:
                fleet_status_updates[:, 4] = 0
                center_points = all_center
            else:
                logger.info('fleet divided. two centers created')
                center_points = results
        return {'center_point': np.array(center_points), 'fleet_status': fleet_status_updates}

    def nearest_enemy(self, ship, game_map):
        entities_by_distance = game_map.nearby_entities_by_distance(ship)
        for entity_distance in sorted(entities_by_distance):
            current_ship_iter = (nearest_entity for nearest_entity in entities_by_distance[entity_distance] if
                                 (isinstance(nearest_entity, hlt.entity.Ship)))
            nearest_enemy_ship = next(current_ship_iter, None)
            if not nearest_enemy_ship:
                continue
            elif nearest_enemy_ship.owner != game_map.get_me():
                    return nearest_enemy_ship

    def planet_score(self, game_map, central_points, fleets):
        planet_heap = []
        planet_space_heap = [0 for _ in range(PLANET_MAX_NUM)]
        ship_heaps = [[] for _ in range(PLANET_MAX_NUM)]
        fleet_size = len(fleets)
        player_factor = 1 if self.player_num == 2 else 0
        fleet_threshold = max(2 * 4 / self.player_num, int(self.fleet_size * 0.45))
        occupy_max = 5 if self.fleet_size <= 5 else self.fleet_size * 0.75
        for planet in game_map.all_planets():
            distance_from_center = distance(planet.x, planet.y, game_map.width / 2, game_map.height / 2)
            distance_from_center_factor = (distance_from_center / (game_map.width / 6))
            # center_factor = 1.1 ** (player_factor * distance_from_center_factor)
            center_factor = 1 if self.player_num == 2 else 2 ** (-1 * distance_from_center_factor)
            is_planet_friendly = (not planet.is_owned()) or (planet.owner == game_map.get_me())

            logger.debug('distance from center %f ,  %f', distance_from_center, distance_from_center_factor)
            logger.debug('central factor is %f', center_factor)
            # enemy planet scoring
            if not is_planet_friendly:
                logger.debug('enemy planet strength is %f' % self.enemy_health(game_map, planet, 10))
                org_distance = distance(planet.x, planet.y, central_points[0], central_points[1])
                enemy_ship_adj_factor = 1.1 ** (self.enemy_health(game_map, planet, 25) / 255 / fleet_threshold
                                                - player_factor * 1.75)
                adj_distance = org_distance * 2.0 * enemy_ship_adj_factor \
                               / 1.08 ** planet.num_docking_spots * center_factor
                planet_distance = adj_distance
                logger.debug('enemy planet original distance is  %f, adjusted distance is %f ' % (org_distance,
                                                                                                  adj_distance))
                planet_space = max(1, int(fleet_size * .9))

            # defense planet scoring
            elif planet.id in self.planet_status and self.planet_status[planet.id]['defense_priority'] != 0:
                    logger.debug('defense_strategy activated: current planet is %f, and defense priority is %f',
                                 planet.id, self.planet_status[planet.id]['defense_priority'])
                    defense_level = self.planet_status[planet.id]['defense_priority']
                    planet_distance = distance(planet.x, planet.y,  central_points[0], central_points[1]) /\
                        1.08 ** planet.num_docking_spots / defense_level
                    # defense_ratio = 0.9 if self.player_num == 2 else 0.65
                    defense_ratio = 0.8
                    planet_space = int(max(1.0,
                                           min(self.planet_status[planet.id]['Range_25_enemy'] + 3,
                                               int(fleet_size * defense_ratio / (4/self.player_num)))
                                           )
                                       )

            # occupying planet scoring
            else:
                logger.debug('distance from center %f ,  %f',  distance_from_center ,  distance_from_center_factor)
                logger.debug('central factor is %f', center_factor)
                planet_distance = distance(planet.x, planet.y,  central_points[0], central_points[1]) /\
                                  1.08 ** (planet.num_docking_spots - len(planet.all_docked_ships())) * center_factor
                # planet_space = int(min(max(1.0, fleet_size * 0.75),
                #                        planet.num_docking_spots - len(planet.all_docked_ships()),
                #                        2.0))
                planet_space = int(min(occupy_max, planet.num_docking_spots - len(planet.all_docked_ships())))
                # planet_space = min(max(1, self.fleet_size * 0.75)),
                # planet.num_docking_spots - len(planet.all_docked_ships()))
            logger.info('current planet %f, distance is %f, available spots %f, friendly? %s ',
                        planet.id,  planet_distance, planet_space, is_planet_friendly)
            heapq.heappush(planet_heap, (planet_distance, planet.id))
            planet_space_heap[planet.id] = planet_space
            h = []
            for ship in fleets:
                d = ship.calculate_distance_between(planet)
                heapq.heappush(h, (d, ship.id))
            ship_heaps[planet.id] = h
        return planet_heap, ship_heaps, planet_space_heap

    def check_strategy_continuity(self, assignment_pair, game_map):
        new_assignment = assignment_pair
        #  and self.turns <= 35
        if self.prev_assignment:
                # logger.info('strategy continuation check: current ship %s', assignment_pair[0])
                # in case somebody blow up the planet, we would just skip the previous one
                if assignment_pair[0].id in self.prev_assignment and \
                        isinstance(self.prev_assignment[assignment_pair[0].id], hlt.entity.Planet) and \
                        isinstance(game_map.get_planet(self.prev_assignment[assignment_pair[0].id].id),
                                   hlt.entity.Planet):
                    previous_target = self.prev_assignment[assignment_pair[0].id]
                    # logger.info('strategy continuation check: current ship %s in previous strategy %s',
                    #           assignment_pair[0].id, assignment_pair[0].id in self.prev_assignment)
                    target_distance = distance(assignment_pair[0].x, assignment_pair[0].y,
                                               previous_target.x,
                                               previous_target.y)
                    # logger.info('previous target is %s, current target distance is %f',
                    # self.prev_assignment[assignment_pair[0].id], target_distance)
                    # logger.info('type of strategy assignment is %s', type(assignment_pair[0]))
                    logger.debug('current target is %s', assignment_pair[1])
                    if (target_distance <= 40.0) \
                            and (previous_target.num_docking_spots != len(previous_target.all_docked_ships())) \
                            and( previous_target != assignment_pair[1]):
                        new_assignment = (assignment_pair[0], game_map.get_planet(previous_target.id))
                        logger.info('strategy continuation, %s is replaced by %s', assignment_pair[1].id,
                                     self.prev_assignment[new_assignment[0].id].id)
        return new_assignment

    def strategy_generating(self, planet_heap, ship_heaps, planet_space_heap, game_map, target_ships):
        # logger.info('I have %f ships' %len(map.get_me().all_ships()))
        number_of_ships_to_assign = len(target_ships)

        assignment = []
        already_assigned_ships = set()
        # logger.info('Round Start: undock ships # %f, assigned #: %f', number_of_ships_to_assign,
        #              len(already_assigned_ships))

        defense_fleet_size = 0
        defense_fleet_cap = min(25, max(1, int(0.8 * self.fleet_size - self.defense_fleet_size)))
        # logger.info('undocked ships # %f', number_of_ships_to_assign)

        while number_of_ships_to_assign > len(already_assigned_ships):
            try:
                current_distance, best_planet_id = heapq.heappop(planet_heap)
            except IndexError:
                break

            ships_needed = planet_space_heap[best_planet_id]

            if game_map.get_planet(best_planet_id).owner == game_map.get_me() \
                    and defense_fleet_size < defense_fleet_cap \
                    and self.planet_status[best_planet_id]['defense_priority'] != 0:
                defense_fleet_size += 1
                planet_space_heap[best_planet_id] = ships_needed - 1
            elif game_map.get_planet(best_planet_id).owner == game_map.get_me() \
                    and defense_fleet_size == defense_fleet_cap\
                    and self.planet_status[best_planet_id]['defense_priority'] != 0:
                planet_space_heap[best_planet_id] = 0
                ships_needed = 0  # set ship need for this turn to zeros
            else:
                planet_space_heap[best_planet_id] = ships_needed - 1

            # if ship needed is non-zero, we need to push the planet back so next round it will be picked up.

            if ships_needed > 0:
                heapq.heappush(planet_heap, (current_distance, best_planet_id))
            elif ships_needed == 0:
                continue

            _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])
            while best_ship_id in already_assigned_ships:
                _, best_ship_id = heapq.heappop(ship_heaps[best_planet_id])
            logger.info("ship %f has been assign to planet %s with distance %f", best_ship_id, best_planet_id,
                         current_distance)
            new_assignment = self.check_strategy_continuity([game_map.get_me().get_ship(best_ship_id),
                                                             game_map.get_planet(best_planet_id)], game_map)

            assignment.append(new_assignment)
            already_assigned_ships.add(best_ship_id)

        if number_of_ships_to_assign > len(already_assigned_ships):
            unassigned_ships = [ship for ship in target_ships if ship.id not in already_assigned_ships]
            for ship in unassigned_ships:
                target = self.nearest_enemy(ship, game_map)
                already_assigned_ships.add(ship.id)
                new_assignment = self.check_strategy_continuity([game_map.get_me().get_ship(ship.id), target], game_map)
                assignment.append(new_assignment)

        return {'assignment': assignment,
                'defense_fleet_size': defense_fleet_size}

    def defense_strategy(self, game_map, avaiable_ships, defense_list):
        pass

    def collision_detect(self, movement):
        fleet_location = np.array(self.fleet_status)
        collision_pair = []
        planet_collision_pair = []

        for i in range(len(movement) - 1):
            ship_id_i = float(movement[i, 1])
            cord_org_i = fleet_location[np.where(fleet_location[:, 0] == ship_id_i)][:, 1:3]

            for j in range(i + 1, len(movement)):
                ship_id_j = float(movement[j, 1])
                cord_org_j = fleet_location[np.where(fleet_location[:, 0] == ship_id_j)][:, 1:3]
                # logger.info('calculation check %s', cord_org_j)
                # logger.info('calculation check %s, %s, and %s, %s ',
                #              # float(movement[j, 2]) * math.cos(float(movement[j, 3]) / 180.0 * math.pi),
                #              float(movement[i, 3]) / 180.0  , math.cos(float(movement[i, 3]) / 180.0 * math.pi),
                #               float(movement[i, 2]), math.sin(float(movement[i, 3]) / 180.0 * math.pi))
                dx = cord_org_i[0][0] - cord_org_j[0][0]
                dy = cord_org_i[0][1] - cord_org_j[0][1]
                dvx = float(movement[i, 2]) * math.cos(float(movement[i, 3]) / 180.0 * math.pi) -\
                    float(movement[j, 2]) * math.cos(float(movement[j, 3]) / 180.0 * math.pi)
                dvy = float(movement[i, 2]) * math.sin(float(movement[i, 3]) / 180.0 * math.pi) - \
                    float(movement[j, 2]) * math.sin(float(movement[j, 3]) / 180.0 * math.pi)
                dvdr = dx * dvx + dy * dvy
                # logger.info('current ship pair %s and %s', ship_id_i, ship_id_j)
                # logger.info('current dvdr %f', dvdr)
                if dvdr <= 0:
                    dvdv = dvx * dvx + dvy * dvy
                    drdr = dx * dx + dy * dy
                    sigma = 2.0
                    d = (dvdr*dvdr) - dvdv * (drdr - sigma * sigma)
                    # logger.info('current d %f', d)
                    if d > 0:
                        t = - (dvdr + math.sqrt(d)) / dvdv
                        if 0 < t <= 1:
                            logger2.debug('collision detected with ship id %f and %f with time %f and dvdr %f, d %f',
                                          ship_id_i,
                                          ship_id_j,
                                          t, dvdr, d)
                            logger2.debug('Calculation Details:  dx %f and dy %f '
                                          'dvx %f and dvy %f and movement  i (%f, %f, %f) and movement  j (%f, %f, %f)',
                                          dx,dy, dvx, dvy,
                                          ship_id_i,
                                          float(movement[i, 2]),
                                          float(movement[i, 3]),
                                          ship_id_j,
                                          float(movement[j, 2]),
                                          float(movement[j, 3]),
                                          )
                            logger2.debug('and movement  i (%f, %f, %f) and movement  j (%f, %f, %f)',
                                          ship_id_i,
                                          float(movement[i, 2]),
                                          float(movement[i, 3]),
                                          ship_id_j,
                                          float(movement[j, 2]),
                                          float(movement[j, 3]),
                                          )
                            heapq.heappush(collision_pair, (t, [int(i), int(j)]))

            # check collision with planets
            for j in range(len(self.all_planet_locations)):
                cor_planet = self.all_planet_locations[j, 1:3]
                radius = float(self.all_planet_locations[j, 3])
                dx = cord_org_i[0][0] - cor_planet[0]
                dy = cord_org_i[0][1] - cor_planet[1]
                dvx = float(movement[i, 2]) * math.cos(float(movement[i, 3]) / 180.0 * math.pi)
                dvy = float(movement[i, 2]) * math.sin(float(movement[i, 3]) / 180.0 * math.pi)

                dvdr = dx * dvx + dy * dvy
                # logger.info('current ship pair %s and %s', ship_id_i, ship_id_j)
                # logger.debug('current dvdr %f', dvdr)
                if dvdr <= 0:
                    dvdv = dvx * dvx + dvy * dvy
                    drdr = dx * dx + dy * dy
                    sigma = 1.0 + radius
                    d = (dvdr*dvdr) - dvdv * (drdr - sigma * sigma)
                    # logger.info('current d %f', d)
                    if d > 0:
                        t = - (dvdr + math.sqrt(d)) / dvdv
                        if 0 < t <= 1:
                            logger2.debug('collision detected with ship id %f and planet id %f '
                                          'with time %f and dvdr %f, d %f',
                                          ship_id_i,
                                          self.all_planet_locations[j, 0],
                                          t, dvdr, d)
                            logger2.debug('Calculation Details:  dx %f and dy %f '
                                          'dvx %f and dvy %f',
                                          dx,dy, dvx, dvy)
                            heapq.heappush(planet_collision_pair, (t, [int(i), self.all_planet_locations[j, 0]]))
        return {'ship_collision_pair': collision_pair, 'planet_collision_pair': planet_collision_pair}

    def collision_check(self, thrust_move, start_time, recursive_count, previous_time_used):
        """
        recursively reslove the collision events, for each time, just change the angle a little bit.
        :param thrust_move: movement generated, following the fomart <movement type t/d/..> <ship id> <speed> <angle>
        :param start_time: time (in seconds) between the Epoch and the start of this round
        :param previous_time_used: current time
        :return: resolved movement
        """
        current_time = time.time()
        time_remain = (2.0 - (current_time - start_time)) * 1000
        logger2.info('current have %f ms left and now is %f recursion', time_remain, recursive_count)
        logger2.info('current time %s, start time %s', current_time, start_time)
        recursive_count += 1
        costly_flag = previous_time_used > 100.0 and time_remain < 170.0
        logging.info('is collision avoiding too costly? %s', costly_flag)
        if recursive_count >= 900.0:
            logger2.warning('Max Recursion Meet.')
            return thrust_move
        elif time_remain > 150 and not costly_flag:
            logger2.debug('current have %f thrust movement', len(thrust_move))
            collision_pairs = self.collision_detect(thrust_move)
            ship_pairs = collision_pairs['ship_collision_pair']
            planet_pairs = collision_pairs['planet_collision_pair']
            if len(ship_pairs) >= 1 or len(planet_pairs) >= 1:
                while len(ship_pairs) > 0:
                    t, item = heapq.heappop(ship_pairs)
                    logger2.info('resolving ship collision %s', item)
                    logger2.info('remaining ship paris %s', len(ship_pairs))
                    ang1 = float(thrust_move[item[0], 3])
                    ang2 = float(thrust_move[item[1], 3])
                    abs_ang_diff = abs(ang1 - ang2)
                    if abs_ang_diff > 90.0:
                        speed1 = int(thrust_move[item[0], 2])
                        spead2 = int(thrust_move[item[1], 2])
                        spead_adj_flag = ((speed1 == 0) and (spead2 == 7)) \
                                            or ((speed1 == 7) and (spead2 == 0))
                        # logging.debug('current meeting speed boundary? %s',spead_adj_flag)
                        # logging.debug('what worong !!! %s, %s ', thrust_move[item[0], 2], thrust_move[item[1], 2])
                        # logging.debug('what worong !!! %s, %s ', thrust_move[item[0], 2] ==0.0,
                        #  thrust_move[item[0], 2] == 7.0)
                        # logging.debug('what worong !!! %s, %s ', thrust_move[item[1], 2] == 0.0,
                        #               thrust_move[item[1], 2] == 7.0)
                        spead_adj = 1.0 if spead_adj_flag else 0.0
                        if ang1 > ang2:
                            thrust_move[item[0], 2] = max(0, int(float(thrust_move[item[0], 2]) * t - 1))
                            thrust_move[item[1], 2] = min(hlt.constants.MAX_SPEED,
                                                          int(float(thrust_move[item[1], 2]) * 1.5) + 1)
                            thrust_move[item[1], 3] = int(float(thrust_move[item[1], 3]) + spead_adj * 5.0)
                        else:
                            thrust_move[item[1], 2] = max(0, int(float(thrust_move[item[1], 2]) * t - 1))
                            thrust_move[item[0], 2] = min(hlt.constants.MAX_SPEED,
                                                          int(float(thrust_move[item[0], 2]) * 1.5) + 1)
                            thrust_move[item[0], 3] = int(float(thrust_move[item[0], 3]) + spead_adj * 5.0)
                        logger2.debug('pre-adj speed %s, $%s; post adj speed %s, %s', speed1, spead2,
                                      thrust_move[item[0], 2], thrust_move[item[1], 2])
                    elif abs_ang_diff > 2:
                        step_factor = 1 if time_remain > 400 else 4
                        if ang1 > ang2:
                            thrust_move[item[0], 3] = max(0, int(thrust_move[item[0], 3]) - 2 * step_factor)
                            thrust_move[item[1], 3] = int(thrust_move[item[1], 3]) + 2 * step_factor
                        elif ang1 < ang2:
                            thrust_move[item[0], 3] = int(thrust_move[item[0], 3]) + 2 * step_factor
                            thrust_move[item[1], 3] = max(0, int(thrust_move[item[1], 3]) - 2 * step_factor)
                    else:
                        thrust_move[item[0], 3] = int(thrust_move[item[0], 3]) + 90
                        thrust_move[item[1], 3] = max(0, int(thrust_move[item[1], 3]) - 90)

                while len(planet_pairs) > 0:
                    t, item = heapq.heappop(planet_pairs)
                    logger2.info('resolving ship collision %s', item)
                    logger2.info('remaining ship paris %s', len(ship_pairs))
                    thrust_move[item[0], 3] = int(float(thrust_move[item[0], 3]) + 5.0)

                end_time = time.time()
                time_used = (end_time - current_time) * 1000
                logger2.info('this recursion takes %f ms', time_used)
                return self.collision_check(thrust_move=thrust_move,
                                            start_time=start_time,
                                            recursive_count=recursive_count,
                                            previous_time_used=time_used)
            else:
                logger2.info('no collision found/left. return movement')
                return thrust_move
        else:
            logger2.info('more than 900 recursion or time out. Return original Movement')
            return thrust_move

    def produce_instructions(self, game_map, ships_to_planets_assignment, round_start_time):
        """
        Given list of pairs (ship, planet) produce instructions for every ship to go to its respective planet.
        If the planet belongs to the enemy, we go to the weakest docked ship.
        If it's ours or is unoccupied, we try to dock.
        :param game_map: game map
        :param ships_to_planets_assignment: list of (ship, planet)
        :param round_start_time: time (in seconds) between the Epoch and the start of this round
        :return: list of instructions to send to the Halite engine
        """
        command_queue = []
        # Send each ship to its planet
        for ship, planet in ships_to_planets_assignment:
            speed = hlt.constants.MAX_SPEED
            if isinstance(planet, hlt.entity.Ship):
                logger.info('currently targeting a ship, enemy ship %s with our ship %s', planet.id, ship.id)
                command_queue.append(
                    self.local_navigate(game_map, round_start_time, ship, ship.closest_point_to(planet), speed))
            elif isinstance(planet, hlt.entity.Planet):
                is_planet_friendly = (not planet.is_owned()) or (planet.owner == game_map.get_me())
                # logger.info('current assign instruction of ship %f, to planet %f, with avaiable spots %f',
                #              ship.id,
                #              planet.id,
                #              (planet.num_docking_spots - len(planet.all_docked_ships())) )
                # logger.info('current planet %f, is friendly? %s, has spots? %s, is owned? %s is owned by me? %s',
                #              planet.id,
                #              is_planet_friendly,
                #              (planet.num_docking_spots != len(planet.all_docked_ships())),
                #              (not planet.is_owned()),(planet.owner == game_map.get_me()))
                if is_planet_friendly:
                    if planet.num_docking_spots != len(planet.all_docked_ships()):
                        # logger.info('start dock ship %f, to planet %f', ship.id, planet.id)
                        if ship.can_dock(planet):
                            command_queue.append(ship.dock(planet))
                        else:
                            command_queue.append(
                                self.local_navigate(game_map, round_start_time, ship, ship.closest_point_to(planet), speed))
                    # elif planet.num_docking_spots == len(planet.all_docked_ships()):
                    else:
                        target = self.nearest_enemy(planet, game_map)
                        command_queue.append(
                            self.local_navigate(game_map, round_start_time, ship, ship.closest_point_to(target), speed))
                else:
                    docked_ships = planet.all_docked_ships()
                    assert len(docked_ships) > 0
                    weakest_ship = None
                    for s in docked_ships:
                        if weakest_ship is None or weakest_ship.health > s.health:
                            weakest_ship = s
                    command_queue.append(
                        self.local_navigate(game_map, round_start_time, ship, ship.closest_point_to(weakest_ship), speed))
            else:
                # somebody may blow up the target....
                target = self.nearest_enemy(ship, game_map)
                command_queue.append(
                    self.local_navigate(game_map, round_start_time, ship, ship.closest_point_to(target), speed))
        return command_queue

    def rabbit_strategy_check(self, available_fleet, game_map):
        rabbit = []
        if self.turns == 1:
            rabbbit_candidate = available_fleet[0]
            target = self.nearest_enemy(rabbbit_candidate,game_map)
            target_distance = distance(rabbbit_candidate.x, rabbbit_candidate.y, target.x, target.y)
            if target_distance < 7*30 * 2 / self.player_num:
                logger.info('rabbit strategy is enabled, current ship %s', rabbbit_candidate)
                rabbit = rabbbit_candidate
        elif self.turns > 1 and isinstance(self.rabbit, hlt.entity.Ship):
            rabbit = game_map.get_me().get_ship(self.rabbit.id)
        # logger.info('current rabbit is %s, and target fleet size is %f', self.rabbit, len(available_fleet))
        if isinstance(rabbit, hlt.entity.Ship):
            available_fleet = [x for x in available_fleet if x.id != rabbit.id]
            # logger.info('current target fleet is %s', available_fleet)
            # logger.info('remove rabbit from assignment, and target fleet size is %f', len(available_fleet))
        return rabbit, available_fleet

    def play(self):
        game = hlt.Game(self._name)

        km = KMeans(2, init='k-means++')

        while True:
            self.turns = self.turns + 1
            self.defense_fleet_size = 0
            # TURN START
            # Update the map for the new turn and get the latest version
            game_map = game.update_map()
            self.player_num = game_map.all_players().__len__()
            start_time = time.time()

            # check the owned planet's status
            self.planet_status = self.check_planet_status(game_map)
            current_time = time.time()
            logger.info('The planet status update takes %f ms', (current_time - start_time) * 1000)

            logger.info('The planet status is %s', self.planet_status)

            # grab available ships & status
            undocked_ships = [ship for ship in game_map.get_me().all_ships() if
                              ship.docking_status == ship.DockingStatus.UNDOCKED]

            self.fleet_size = len(undocked_ships)

            # logger.info('current available fleet %s', undocked_ships)
            self.fleet_status = np.array([[ship.id, ship.x, ship.y, ship.health, 0] for ship in undocked_ships])
            self.all_planet_locations = np.array([[planet.id, planet.x, planet.y, planet.radius, planet.health]
                                                  for planet in game_map.all_planets()])

            # central points groups
            # {'center_point': np.array(center_points), 'fleet_statut': fleet_status_updates}

            central_analysis_result = self.centeral_points(km=km, fleets=self.fleet_status, game_map=game_map)
            central_points = central_analysis_result['center_point']
            self.fleet_status = central_analysis_result['fleet_status']
            logger.info('current fleet has %f available ships status is:', self.fleet_size)
            # logger.info('\n')
            logger.info('%s', self.fleet_status)
            # create rabbit
            self.rabbit, undocked_ships = self.rabbit_strategy_check(undocked_ships, game_map)

            # created strategy
            strategy_assignment = []

            if self.rabbit:
                target = self.nearest_enemy(self.rabbit, game_map)
                logger.info('rabbit target is %s', target)
                strategy_assignment.append([self.rabbit, target])
                # logger.info('current stage 1 strategy is %s', strategy_assignment)
            logger.info("current centers are %s and shape is %s", central_points, central_points.shape)
            if central_points.shape == (2,):
                self.planet_heap_fleet1, self.ship_heaps_fleet1, self.planet_space_heap_fleet1 = \
                    self.planet_score(game_map=game_map,
                                      central_points=central_points,
                                      fleets=undocked_ships)
                strategy_result = self.strategy_generating(self.planet_heap_fleet1,
                                                           self.ship_heaps_fleet1,
                                                           self.planet_space_heap_fleet1,
                                                           game_map,
                                                           undocked_ships)
                strategy_assignment = strategy_assignment + strategy_result['assignment']

            elif central_points.shape == (2, 2):
                for i in range(2):

                    ship_ids = self.fleet_status[np.where(self.fleet_status[:, 4] == i)][:, 0]
                    logger.info("fleet divided, current fleet %f with %f ships", i, len(ship_ids))
                    target_fleet = [ships for ships in undocked_ships if ships.id in ship_ids ]
                    planet_heap, ship_heaps, planet_space_heap = self.planet_score(game_map,
                                                                                   central_points[i],
                                                                                   target_fleet)
                    strategy_result = self.strategy_generating(planet_heap,
                                                               ship_heaps,
                                                               planet_space_heap,
                                                               game_map,
                                                               target_fleet)
                    self.defense_fleet_size += strategy_result['defense_fleet_size']
                    strategy_assignment += strategy_result['assignment']

            else:
                raise ValueError('more then 2 central points found. No action will be made. Please check.')

            # if self.planet_status:
            #     logger.info('the current occupancy status are as %s', self.planet_status)

            # logger.info('current strategy is %s', strategy_assignment)

            # update the stored assignment
            self.prev_assignment = dict((items[0].id, items[1]) for items in strategy_assignment)
            # logger.info('current assignment dict is %s', self.prev_assignment)

            # generate ship movements
            logger.info('there are %f ms left for movement generating in turn %f',
                        (2 - (current_time - start_time)) * 1000, self.turns)
            ship_movement = self.produce_instructions(game_map=game_map,
                                                      ships_to_planets_assignment=strategy_assignment,
                                                      round_start_time=start_time)
            logger.info('current movement is %s', ship_movement)

            # check collision
            thrust_move_list = [item for item in ship_movement if item.split()[0] == 't']
            other_move_list = [item for item in ship_movement if item.split()[0] != 't']
            thrust_move = np.array([item.split() for item in thrust_move_list])

            if (2 - (current_time - start_time)) * 1000 > 700 and len(thrust_move) >= 2:
                logger2.info('check the collision')
                recurisve_count = 0.0
                previous_recursion_time_used = 0.0
                ship_movement = self.collision_check(thrust_move=thrust_move,
                                                     start_time=start_time,
                                                     recursive_count=recurisve_count,
                                                     previous_time_used=previous_recursion_time_used)
                logger.info('non collision movement is %s', ship_movement)
                new_thrust = [" ".join(thrust_move[i]) for i in range(len(thrust_move))]
                ship_movement = new_thrust + other_move_list

            # submitting commands
            current_time = time.time()
            logger.info('there are %f ms left in turn %f', (2 - (current_time - start_time))*1000, self.turns)
            game.send_command_queue(ship_movement)


if __name__ == '__main__':
    BotV3 = Bot('NewBotV12')
    BotV3.play()
