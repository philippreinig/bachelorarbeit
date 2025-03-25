import matplotlib.pyplot as plt
import logging
import numpy as np

from matplotlib.colors import ListedColormap, BoundaryNorm

log = logging.getLogger("rich")

aki_labels = [
    {
        "name": "void",
        "id": 0,
        "category": "void",
        "cat_id": 10,
        "color": [
            0.4369368498410794,
            0.9912596163254394,
            0.2737769136186745
        ]
    },
    {
        "name": "terrain",
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
        "name": "vegetation",
        "id": 2,
        "category": "Nature",
        "cat_id": 0,
        "color": [
            0.6914425130910156,
            0.013149503182243949,
            0.29349825883707803
        ]
    },
    {
        "name": "lane marking",
        "id": 3,
        "category": "lane marking",
        "cat_id": 1,
        "color": [
            0.0540556064815797,
            0.5116876049377178,
            0.14162267079720203
        ]
    },
    {
        "name": "animal",
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
        "name": "bird",
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
        "name": "dynamic",
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
        "name": "ground",
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
        "name": "static",
        "id": 8,
        "category": "other",
        "cat_id": 2,
        "color": [
            0.38378757912675154,
            0.9958500556598722,
            0.8033970403941427
        ]
    },
    {
        "name": "person",
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
        "name": "rider",
        "id": 10,
        "category": "person",
        "cat_id": 3,
        "color": [
            0.9354677596010909,
            0.8089185617453203,
            0.5480290227214126
        ]
    },
    {
        "name": "pole",
        "id": 11,
        "category": "pole",
        "cat_id": 4,
        "color": [
            0.03525504846810401,
            0.743142006803481,
            0.7545558231632763
        ]
    },
    {
        "name": "road",
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
        "name": "sidewalk",
        "id": 13,
        "category": "road and sidewalk",
        "cat_id": 5,
        "color": [
            1.0,
            0.0,
            1.0
        ]
    },
    {
        "name": "sky",
        "id": 14,
        "category": "sky",
        "cat_id": 6,
        "color": [
            1.0,
            0.5,
            0.0
        ]
    },
    {
        "name": "building",
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
        "name": "fence",
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
        "name": "wall",
        "id": 17,
        "category": "structure",
        "cat_id": 7,
        "color": [
            0.06914947848828823,
            0.008714826240349915,
            0.665187340544127
        ]
    },
    {
        "name": "traffic light",
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
        "name": "traffic sign",
        "id": 19,
        "category": "traffic signal",
        "cat_id": 8,
        "color": [
            0.9033186570627191,
            0.9958004499833919,
            0.05175729399482931
        ]
    },
    {
        "name": "bicycle",
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
        "name": "bus",
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
        "name": "car",
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
        "name": "ego vehicle",
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
        "name": "motorcycle",
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
        "name": "vehicle",
        "id": 26,
        "category": "vehicle",
        "cat_id": 9,
        "color": [
            0.5411196047601873,
            0.29265588136512977,
            0.9920718774103889
        ]
    }
]

def get_aki_label_colors():
    return [aki_label["color"] for aki_label in aki_labels]

def get_aki_label_colors_rgb():
    colors = get_aki_label_colors()
    colors_rgb = []
    for color in colors:
        r,g,b = color
        colors_rgb.append([r*255, g*255, b*255])
    return colors_rgb

def get_aki_label_listed_color_map():
    return ListedColormap(np.array(get_aki_label_colors_rgb())/255)

def get_aki_label_boundary_norm(listed_cmap=get_aki_label_listed_color_map()):
    boundary_norm = BoundaryNorm(range(len(aki_labels) + 1), listed_cmap.N)
    return boundary_norm

    
def get_aki_label_names():
    label_names = [aki_label["name"] for aki_label in aki_labels]
    return label_names


def visualize_aki_labels(lbls: list[dict] = aki_labels):
    """Creates an image containing a colored bar for each label in lbls

    Args:
        lbls (list[dict]): The list of aki_labels. Each entry is expected to be a
        dictionary with entries: 'name': str, 'id': int and "color": list[float] with 3 entries
        for each color channel in range [0,1]
    """
    num_labels = len(lbls)
    img_size = (1000, 1000)
    stripe_height = img_size[1] // num_labels
    
    _, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100))
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(0, img_size[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    for i, label in enumerate(reversed(lbls)):
        color = label["color"]
        print(f"Color of label {label['name']} is {color}")
        rect = plt.Rectangle((0, i * stripe_height), img_size[0], stripe_height, color=color, ec='black')
        ax.add_patch(rect)
        text_x = 10
        text_y = (i + 0.5) * stripe_height
        ax.text(text_x, text_y, f"{label['id']}: {label['name']}", va='center', ha='left', fontsize=10, color='black', weight='bold')
    
    plt.savefig("aki_labels.png")
    plt.close()

if __name__ == "__main__":
    print(get_aki_label_names())