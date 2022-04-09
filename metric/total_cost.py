from . import BaseMetric
import numpy as np

class TotalCostMetric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """
    def __init__(self, world):

        self.c_t = 1/60/5 * 2.4
        self.c_f = 1

        self.world = world
        self.world.subscribe(["vehicles", "time"])

        self.vehicle_total_cost = {}

        self.vehicle_enter_time = {}
        self.travel_times = {}

        self.vehicle_route = {}
        self.vehicle_fuel = {}

        # calculate road lengths
        self.road_length = {}
        for road in self.world.roadnet["roads"]:
            points = road["points"]
            self.road_length[road["id"]] = \
                pow(((points[0]["x"] - points[1]["x"]) ** 2 + (points[0]["y"] - points[1]["y"]) ** 2), 0.5)

    def cal_fuel(self, routes):
        distance = 0
        ratio = 71.294 / (68593.74-0)
        for road in routes:
            distance += self.road_length[road]

        return distance * ratio * 2.4


    def update(self, done=False):
        vehicles = self.world.get_info("vehicles")
        current_time = self.world.get_info("time")
        lane_vehicles = self.world.get_info("lane_vehicles")

        for vehicle in vehicles:
            if vehicle not in self.vehicle_enter_time:
                self.vehicle_enter_time[vehicle] = current_time

        for vehicle in list(self.vehicle_enter_time):
            if done or vehicle not in vehicles:
                self.travel_times[vehicle] = (current_time - self.vehicle_enter_time[vehicle])
                del self.vehicle_enter_time[vehicle]

        dic_vehicle_lane = {}
        for lane_id, lane in lane_vehicles.items():
            for vehicle in lane:
                dic_vehicle_lane[vehicle] = "_".join(lane_id.split("_")[:-1])

        for vehicle in vehicles:
            if vehicle not in self.vehicle_route:
                if vehicle in dic_vehicle_lane:
                    self.vehicle_route[vehicle] = [dic_vehicle_lane[vehicle]]
                else:
                    continue

        for vehicle in list(self.vehicle_route):
            if vehicle in dic_vehicle_lane:
                if self.vehicle_route[vehicle][-1] != dic_vehicle_lane[vehicle]:
                    self.vehicle_route[vehicle].append(dic_vehicle_lane[vehicle])

        for vehicle in self.vehicle_route:
            self.vehicle_fuel[vehicle] = (self.cal_fuel(self.vehicle_route[vehicle]))

        for vehicle in self.travel_times.keys():
            if vehicle in self.vehicle_fuel:
                self.vehicle_total_cost[vehicle] = self.c_t * self.travel_times[vehicle] + self.c_f * self.vehicle_fuel[vehicle]

        if done:
            print("eveluated vehicles:", len(self.vehicle_total_cost))

        return np.mean(list(self.vehicle_total_cost.values())) if len(self.vehicle_total_cost) else 0

