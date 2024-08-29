# Dataset Description
| Description | NuScenes | Ours |
|-------------|----------|------|
| Frequency   | 2hz      | 2hz  |
| Frames      |          |      |
 | Citys|          |      |   



* we collected 1000 driving scenes in Boston and Singapore
* the scenes of 20 second length are manually selected to show a diverse and interesting set of driving maneuvers
* we annotate 23 object classes with accurate 3D bounding boxes at 2Hz
* The full dataset includes approximately 1.4M camera images, 390k LIDAR sweeps, 1.4M RADAR sweeps and 1.4M object bounding boxes in 40k keyframes.]
* In nuScenes-lidarseg, we annotate each lidar point from a keyframe in nuScenes with one of 32 possible semantic labels (i.e. lidar semantic segmentation). As a result, nuScenes-lidarseg contains 1.4 billion annotated points across 40,000 pointclouds and 1000 scenes (850 scenes for training and validation, and 150 scenes for testing).

| Category                                     | nuScenes cuboids | Cuboid ratio | Lidarseg points | Point ratio |
|----------------------------------------------|------------------|--------------|-----------------|-------------|
| animal                                       | 787              | 0.07%        | 5,385           | 0.01%       |
| human.pedestrian.adult                       | 208,240          | 17.86%       | 2,156,470       | 2.73%       |
| human.pedestrian.child                       | 2,066            | 0.18%        | 9,655           | 0.01%       |
| human.pedestrian.construction_worker         | 9,161            | 0.79%        | 139,443         | 0.18%       |
| human.pedestrian.personal_mobility           | 395              | 0.03%        | 8,723           | 0.01%       |
| human.pedestrian.police_officer              | 727              | 0.06%        | 9,159           | 0.01%       |
| human.pedestrian.stroller                    | 1,072            | 0.09%        | 8,809           | 0.01%       |
| human.pedestrian.wheelchair                  | 503              | 0.04%        | 12,168          | 0.02%       |
| movable_object.barrier                       | 152,087          | 13.04%       | 9,305,106       | 11.79%      |
| movable_object.debris                        | 3,016            | 0.26%        | 66,861          | 0.08%       |
| movable_object.pushable_pullable             | 24,605           | 2.11%        | 718,641         | 0.91%       |
| movable_object.trafficcone                   | 97,959           | 8.40%        | 736,239         | 0.93%       |
| static_object.bicycle_rack *                 | 2,713            | 0.23%        | 163,126         | 0.21%       |
| vehicle.bicycle                              | 11,859           | 1.02%        | 141,351         | 0.18%       |
| vehicle.bus.bendy                            | 1,820            | 0.16%        | 357,463         | 0.45%       |
| vehicle.bus.rigid                            | 14,501           | 1.24%        | 4,247,297       | 5.38%       |
| vehicle.car                                  | 493,322          | 42.30%       | 38,104,219      | 48.27%      |
| vehicle.construction                         | 14,671           | 1.26%        | 1,514,414       | 1.92%       |
| vehicle.emergency.ambulance                  | 49               | 0.00%        | 2,218           | 0.00%       |
| vehicle.emergency.police                     | 638              | 0.05%        | 59,590          | 0.08%       |
| vehicle.motorcycle                           | 12,617           | 1.08%        | 427,391         | 0.54%       |
| vehicle.trailer                              | 24,860           | 2.13%        | 4,907,511       | 6.22%       |
| vehicle.truck                                | 88,519           | 7.59%        | 15,841,384      | 20.07%      |
| **Total**                                    | 1,166,187        | 100.00%      | 78,942,623      | 100.00%     |

| Category                                     | nuScenes cuboids | Cuboid ratio | Lidarseg points | Point ratio |
|----------------------------------------------|------------------|--------------|-----------------|-------------|
| flat.driveable_surface                       | -                | -            | 316,958,899     | 28.64%      |
| flat.other                                   | -                | -            | 8,559,216       | 0.77%       |
| flat.sidewalk                                | -                | -            | 70,197,461      | 6.34%       |
| flat.terrain                                 | -                | -            | 70,289,730      | 6.35%       |
| static.manmade                               | -                | -            | 178,178,063     | 16.10%      |
| static.other                                 | -                | -            | 817,150         | 0.07%       |
| static.vegetation                            | -                | -            | 122,581,273     | 11.08%      |
| vehicle.ego                                  | -                | -            | 337,070,621     | 30.46%      |
| noise                                        | -                | -            | 2,061,156       | 0.19%       |
| **Total**                                    | -                | -            | 1,106,713,569   | 100.00%     |


## carla label
| Value | Tag          | Converted color | Description                                                                                                                                                                                                                            |
|-------|--------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | Unlabeled    | (0, 0, 0)       | Elements that have not been categorized are considered Unlabeled. This category is meant to be empty or at least contain elements with no collisions.                                                                                  |
| 1     | Roads        | (128, 64, 128)  | Part of ground on which cars usually drive. E.g. lanes in any directions, and streets.                                                                                                                                                 |
| 2     | SideWalks    | (244, 35, 232)  | Part of ground designated for pedestrians or cyclists. Delimited from the road by some obstacle (such as curbs or poles), not only by markings. This label includes a possibly delimiting curb, traffic islands, and pedestrian zones. |
| 3     | Building     | (70, 70, 70)    | Buildings like houses, skyscrapers, etc., and the elements attached to them like air conditioners, scaffolding, awning, ladders and much more.                                                                                         |
| 4     | Wall         | (102, 102, 156) | Individual standing walls. Not part of a building.                                                                                                                                                                                     |
| 5     | Fence        | (190, 153, 153) | Barriers, railing, or other upright structures. Basically wood or wire assemblies that enclose an area of ground.                                                                                                                      |
| 6     | Pole         | (153, 153, 153) | Small mainly vertically oriented pole. E.g., sign pole, traffic light poles.                                                                                                                                                           |
| 7     | TrafficLight | (250, 170, 30)  | Traffic light boxes without their poles.                                                                                                                                                                                               |
| 8     | TrafficSign  | (220, 220, 0)   | Signs installed by the state/city authority, usually for traffic regulation. This category does not include the poles where signs are attached to. E.g., traffic signs, parking signs, direction signs...                              |
| 9     | Vegetation   | (107, 142, 35)  | Trees, hedges, all kinds of vertical vegetation. Ground-level vegetation is considered Terrain.                                                                                                                                        |
| 10    | Terrain      | (152, 251, 152) | Grass, ground-level vegetation, soil or sand. These areas are not meant to be driven on. This label includes a possibly delimiting curb.                                                                                               |
| 11    | Sky          | (70, 130, 180)  | Open sky. Includes clouds and the sun.                                                                                                                                                                                                 |
| 12    | Pedestrian   | (220, 20, 60)   | Humans that walk.                                                                                                                                                                                                                      |
| 13    | Rider        | (255, 0, 0)     | Humans that ride/drive any kind of vehicle or mobility system like bicycles, scooters, skateboards, horses, roller-blades, wheel-chairs, etc.                                                                                          |
| 14    | Car          | (0, 0, 142)     | Cars, vans.                                                                                                                                                                                                                            |
| 15    | Truck        | (0, 0, 70)      | Trucks.                                                                                                                                                                                                                                |
| 16    | Bus          | (0, 60, 100)    | Busses.                                                                                                                                                                                                                                |
| 17    | Train        | (0, 60, 100)    | Trains.                                                                                                                                                                                                                                |
| 18    | Motorcycle   | (0, 0, 230)     | Motorcycle, Motorbike.                                                                                                                                                                                                                 |
| 19    | Bicycle      | (119, 11, 32)   | Bicycles.                                                                                                                                                                                                                              |
| 20    | Static       | (110, 190, 160) | Elements in the scene and props that are immovable. E.g., fire hydrants, fixed benches, fountains, bus stops, etc.                                                                                                                     |
| 21    | Dynamic      | (170, 120, 50)  | Elements whose position is susceptible to change over time. E.g., Movable trash bins, buggies, bags, wheelchairs, animals, etc.                                                                                                        |
| 22    | Other        | (55, 90, 80)    | Everything that does not belong to any other category.                                                                                                                                                                                 |
| 23    | Water        | (45, 60, 150)   | Horizontal water surfaces. E.g., Lakes, sea, rivers.                                                                                                                                                                                   |
| 24    | RoadLine     | (157, 234, 50)  | The markings on the road.                                                                                                                                                                                                              |
| 25    | Ground       | (81, 0, 81)     | Any horizontal ground-level structures that does not match any other category. For example areas shared by vehicles and pedestrians, or flat roundabouts delimited from the road by a curb.                                            |
| 26    | Bridge       | (150, 100, 100) | Only the structure of the bridge. Fences, people, vehicles, and other elements on top of it are labeled separately.                                                                                                                    |
| 27    | RailTrack    | (230, 150, 140) | All kind of rail tracks that are non-drivable by cars. E.g., subway and train rail tracks.                                                                                                                                             |
| 28    | GuardRail    | (180, 165, 180) | All types of guard rails/crash barriers.                                                                                                                                                                                               |

## nuScenes
| Label | Category          |
|-------|-------------------|
| 0*    | noise             |
| 1     | barrier           |
| 2     | bicycle           |
| 3     | bus               |
| 4     | car               |
| 5     | construction      |
| 6     | motorcycle        |
| 7     | pedestrian        |
| 8     | trafficcone       |
| 9     | trailer           |
| 10    | truck             |
| 11    | driveable_surface |
| 12    | other             |
| 13    | sidewalk          |
| 14    | terrain           |
| 15    | mannade           |
| 16    | vegetation        |




# 都需要那些数据？？？
# 这样的路径，训练出来的目的是什么？