import math

def euclidean_distance(box1, box2):
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def generate_summary_prompt(objects, distances):
    summary = "Detected objects:\n"
    for i, obj in enumerate(objects):
        summary += f"- {obj['name']} at box {obj['box']}\n"

    summary += "\nRelationships:\n"
    for (i, j), dist in distances.items():
        if dist < 100:
            summary += f"- {objects[i]['name']} is close to {objects[j]['name']} (distance: {dist:.1f})\n"

    summary += "\nSummarize this scene in one sentence."
    return summary
