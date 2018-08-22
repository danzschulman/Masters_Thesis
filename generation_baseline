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
