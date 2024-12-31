import logging

log = logging.getLogger("rich")

aki_labels = [
    {
        "name": "terrain",
        "id": 0,
        "category": "Nature",
        "cat_id": 0,
        "color": [
            0.4369368498410794,
            0.9912596163254394,
            0.2737769136186745
        ]
    },
    {
        "name": "vegetation",
        "id": 1,
        "category": "Nature",
        "cat_id": 0,
        "color": [
            0.9366944666670686,
            0.14665592330240995,
            0.5538399102782717
        ]
    },
    {
        "name": "lane marking",
        "id": 2,
        "category": "lane marking",
        "cat_id": 1,
        "color": [
            0.6914425130910156,
            0.013149503182243949,
            0.29349825883707803
        ]
    },
    {
        "name": "animal",
        "id": 3,
        "category": "other",
        "cat_id": 2,
        "color": [
            0.0540556064815797,
            0.5116876049377178,
            0.14162267079720203
        ]
    },
    {
        "name": "bird",
        "id": 4,
        "category": "other",
        "cat_id": 2,
        "color": [
            0.6178384443238483,
            0.0043244144593491685,
            0.7330583525812957
        ]
    },
    {
        "name": "dynamic",
        "id": 5,
        "category": "other",
        "cat_id": 2,
        "color": [
            0.0,
            1.0,
            0.0
        ]
    },
    {
        "name": "ground",
        "id": 6,
        "category": "other",
        "cat_id": 2,
        "color": [
            0.3885630797322672,
            0.24559019911365298,
            0.03639096625602467
        ]
    },
    {
        "name": "static",
        "id": 7,
        "category": "other",
        "cat_id": 2,
        "color": [
            0.04145885812814243,
            0.1821743403995666,
            0.968110596119927
        ]
    },
    {
        "name": "person",
        "id": 8,
        "category": "person",
        "cat_id": 3,
        "color": [
            0.38378757912675154,
            0.9958500556598722,
            0.8033970403941427
        ]
    },
    {
        "name": "rider",
        "id": 9,
        "category": "person",
        "cat_id": 3,
        "color": [
            0.0,
            1.0,
            0.5
        ]
    },
    {
        "name": "pole",
        "id": 10,
        "category": "pole",
        "cat_id": 4,
        "color": [
            0.9354677596010909,
            0.8089185617453203,
            0.5480290227214126
        ]
    },
    {
        "name": "road",
        "id": 11,
        "category": "road and sidewalk",
        "cat_id": 5,
        "color": [
            0.03525504846810401,
            0.743142006803481,
            0.7545558231632763
        ]
    },
    {
        "name": "sidewalk",
        "id": 12,
        "category": "road and sidewalk",
        "cat_id": 5,
        "color": [
            0.5,
            0.75,
            0.5
        ]
    },
    {
        "name": "sky",
        "id": 13,
        "category": "sky",
        "cat_id": 6,
        "color": [
            1.0,
            0.0,
            1.0
        ]
    },
    {
        "name": "building",
        "id": 14,
        "category": "structure",
        "cat_id": 7,
        "color": [
            1.0,
            0.5,
            0.0
        ]
    },
    {
        "name": "fence",
        "id": 15,
        "category": "structure",
        "cat_id": 7,
        "color": [
            1.0,
            0.0,
            0.0
        ]
    },
    {
        "name": "wall",
        "id": 16,
        "category": "structure",
        "cat_id": 7,
        "color": [
            0.5593249084697393,
            0.7699780594873692,
            0.009879124642136139
        ]
    },
    {
        "name": "traffic light",
        "id": 17,
        "category": "traffic signal",
        "cat_id": 8,
        "color": [
            0.06914947848828823,
            0.008714826240349915,
            0.665187340544127
        ]
    },
    {
        "name": "traffic sign",
        "id": 18,
        "category": "traffic signal",
        "cat_id": 8,
        "color": [
            0.6526597079949009,
            0.7393384863873999,
            0.9965411612254462
        ]
    },
    {
        "name": "bicycle",
        "id": 19,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.9033186570627191,
            0.9958004499833919,
            0.05175729399482931
        ]
    },
    {
        "name": "bus",
        "id": 20,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.5508030515173521,
            0.3634028264369099,
            0.4536932054350984
        ]
    },
    {
        "name": "car",
        "id": 21,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.0,
            1.0,
            1.0
        ]
    },
    {
        "name": "ego vehicle",
        "id": 22,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.006266431320108734,
            0.4241239591605288,
            0.5738641488600861
        ]
    },
    {
        "name": "motorcycle",
        "id": 23,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.0,
            0.5,
            1.0
        ]
    },
    {
        "name": "other vehicle",
        "id": 24,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.010328804507718004,
            0.16869862065926922,
            0.3396372762077241
        ]
    },
    {
        "name": "truck",
        "id": 25,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.9758767427759272,
            0.4617250857379033,
            0.7144903030168291
        ]
    },
    {
        "name": "void",
        "id": 26,
        "category": "void",
        "cat_id": 10,
        "color": [
            0.5411196047601873,
            0.29265588136512977,
            0.9920718774103889
        ]
    }
]


def get_aki_label_colors():
    aki_label_colors = []
    for aki_label in aki_labels:
        aki_label_rgb = aki_label["color"]
        for i in range(3):
            aki_label_rgb[i] *= 255
        aki_label_colors.append(tuple(aki_label_rgb))

    return aki_label_colors

def get_aki_label_names(void_labels: list[str] = None):
    all_labels = [aki_label["name"] for aki_label in aki_labels] # Filter labels 0 und 7
    log.info(f"All labels: {all_labels}")
    labels_filtered = [label for label in all_labels if label not in void_labels]
    log.info(f"Labels filtered: {labels_filtered}")
    return labels_filtered