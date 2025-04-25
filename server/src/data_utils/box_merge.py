import json

# read in json function helper
def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def vertical_overlap(b1, b2):
    """Returns the amount of vertical overlap between two boxes."""
    return max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))

def get_box_height(box):
    return box[3] - box[1]

def get_box_center_y(box):
    return (box[1] + box[3]) / 2

class MergeBoxes:
    def __init__(self):
        pass

    def merge_boxes(self, boxes, vertical_tolerance_multiplier=1.1):
        # If the vertical distance of the new box from the current line exceeds a tolerance 
        # (e.g., 1.1× average height), start a new line
        if not boxes:
            return []

        boxes.sort(key=lambda b: (b[1] + b[3]) / 2)

        merged_result = []

        current_line = [boxes[0]]
        avg_height = boxes[0][3] - boxes[0][1]

        def merge_into_line(line_boxes, new_box): # merge new candidate box with the last merged one
            """Try to merge the new box with the last merged one in the line."""
            last = line_boxes[-1]
            if new_box[0] <= last[2]:  
                # merge last box with new box
                line_boxes[-1] = [
                    min(last[0], new_box[0]),
                    min(last[1], new_box[1]),
                    max(last[2], new_box[2]),
                    max(last[3], new_box[3]),
                ]
            else:
                line_boxes.append(new_box)

        for box in boxes[1:]:
            box_center = (box[1] + box[3]) / 2
            current_line_center = sum((b[1] + b[3]) / 2 for b in current_line) / len(current_line)
            tolerance = avg_height * vertical_tolerance_multiplier

            if abs(box_center - current_line_center) <= tolerance:
                merge_into_line(current_line, box)
                avg_height = sum(b[3] - b[1] for b in current_line) / len(current_line)
            else:
                # finalize current merged line
                merged_result.extend(current_line)
                current_line = [box]
                avg_height = box[3] - box[1]

        # add last group
        if current_line:
            merged_result.extend(current_line)

        return merged_result

    def extract_bboxes(self, data, cls="handwritten"):
        """
        Extract bounding boxes from the given data.

        Args:
            data (dict): The input data containing bounding box information.

        Returns:
            list: A list of bounding boxes.
        """
        bboxes = []
        for candidate in data:
            if candidate["cls"] == cls:
                bboxes.append([float(x) for x in candidate["bbox"]])
        return bboxes
    
if __name__ == "__main__":
    # Example usage
    data = read_json("/Users/deluzhao/Documents/Research/outputs/h1_results.json")
    merge_boxes = MergeBoxes()
    bboxes = merge_boxes.extract_bboxes(data)
    merged_bboxes = merge_boxes.merge_boxes(bboxes)
    print(bboxes)
    print(merged_bboxes)