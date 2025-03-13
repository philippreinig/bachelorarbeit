WITH filtered_weather AS (
  SELECT w.weather_uid
  FROM weather w
  JOIN camera c ON w.weather_uid = c.weather_uid
  JOIN camera_segmentation cs ON c.camera_uid = cs.camera_uid
  JOIN lidar l ON w.weather_uid = l.weather_uid
  WHERE w.dataset = 'waymo' AND w.weather = 'rain'
),

numbered_weather AS (
  SELECT weather_uid, 
         ROW_NUMBER() OVER () AS row_num,
         COUNT(*) OVER () AS total_count
  FROM filtered_weather
),

split_indices AS (
  SELECT weather_uid,
         row_num,
         total_count,
         CASE
           WHEN row_num <= 0.7 * total_count THEN 'train'
           WHEN row_num <= 0.9 * total_count THEN 'val'
           ELSE 'test'
         END AS split
  FROM numbered_weather
)

UPDATE weather
SET split = si.split
FROM split_indices si
WHERE weather.weather_uid = si.weather_uid;

