SELECT
        weather.dataset,
        weather.split,
        COUNT(*) AS count, weather.weather,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY weather.dataset, weather.weather) AS percentage
FROM weather
        JOIN camera ON weather.weather_uid = camera.weather_uid
        JOIN lidar ON lidar.weather_uid = weather.weather_uid
        JOIN camera_segmentation ON camera.camera_uid=camera_segmentation.camera_uid
WHERE   weather.dataset='waymo' AND
        weather.weather='rain' AND
        camera.camera_id=lidar.lidar_id
GROUP BY weather.dataset, weather.split, weather.weather;