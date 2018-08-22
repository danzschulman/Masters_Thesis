sys.path.insert(0,'../refer')
from refer import REFER

#import MaskRCNN:
import coco
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MIN_CONFIDENCE = 0.1 # 0.7 (default)

config = InferenceConfig()
config.print()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

data_root = '../refer/data'
dataset = 'refcoco'
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)

ref_ids = refer.getRefIds(split='testB')
images_dir = '/root/refer/data/images/mscoco/images/train2014/'

hyp = open("hyp.txt","w")
ref1 = open("ref1.txt","w")
ref2 = open("ref2.txt","w")
ref3 = open("ref3.txt","w")
ref4 = open("ref4.txt","w")

for ref_id in tqdm(ref_ids):
    ref = refer.Refs[ref_id]
    x,y,w,h = refer.getRefBox(ref_id) # [x, y, w, h]
    x1,y1,x2,y2 = x,y,x+w,y+h
    image_path = images_dir + refer.Imgs[ref['image_id']]['file_name']
    
    image = scipy.misc.imread(image_path)
    if len(image.shape) != 3:
        continue
    
    # Run detection
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]
#     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                                 class_names, r['scores'])
    
    class_ids = results[0]['class_ids']
    boxes = results[0]['rois']
    if len(boxes) == 0:
        continue
    
    subject = None
    for i, box in enumerate(boxes):
        bx1,by1,bx2,by2 = box[1],box[0],box[3],box[2]
        iou = bb_intersection_over_union((x1,y1,x2,y2), (bx1,by1,bx2,by2))
        if iou >= 0.5:
            class_id = class_ids[i]
            correct_index = i
            center = (box[1] + box[3])/2
            subject = class_names[class_id]
            break
    if subject is None:
        continue
        
    subject_boxes = [((box[1] + box[3])/2, i) for i, box in enumerate(boxes) if class_ids[i] == class_id]
    subject_boxes.sort()
    
    sorted_index = subject_boxes.index((center, correct_index))
    generated_refexp = None
    if len(subject_boxes) > 1 and sorted_index == (len(subject_boxes) - 1)/2:
        generated_refexp = subject + ' in the center'
    elif sorted_index == 0:
        generated_refexp = 'leftmost ' + subject
    elif sorted_index == len(subject_boxes) - 1:
        generated_refexp = 'rightmost ' + subject
    elif sorted_index == 1:
        generated_refexp = 'second ' + subject + ' from the left'
    elif sorted_index == len(subject_boxes) - 2:
        generated_refexp = 'second ' + subject + ' from the right'
    elif sorted_index == 2:
        generated_refexp = 'third ' + subject + ' from the left'
    elif sorted_index == len(subject_boxes) - 3:
        generated_refexp = 'third ' + subject + ' from the right'
    else:
        generated_refexp = subject
    
    hyp.write(generated_refexp + '\n')
    
    ref1.write(ref['sentences'][0]['sent'] + '\n')
    
    if len(ref['sentences']) > 1:
        ref2.write(ref['sentences'][1]['sent'] + '\n')
    else:
        ref2.write(ref['sentences'][0]['sent'] + '\n')
        ref3.write(ref['sentences'][0]['sent'] + '\n')
        ref4.write(ref['sentences'][0]['sent'] + '\n')
        
    if len(ref['sentences']) > 2:
        ref3.write(ref['sentences'][2]['sent'] + '\n')
    else:
        ref3.write(ref['sentences'][0]['sent'] + '\n')
        ref4.write(ref['sentences'][0]['sent'] + '\n')
        
    if len(ref['sentences']) > 3:
        ref4.write(ref['sentences'][3]['sent'] + '\n')
    else:
        ref4.write(ref['sentences'][0]['sent'] + '\n')
    
hyp.close()
ref1.close()
ref2.close()
ref3.close()
ref4.close()
