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
