filter_keywords = set([
                   'left', 'right', 'top', 'bottom', 'middle', 
                   'big', 'small', 'long', 'short', 'tall', 'huge', 'tiny',
                   'blue', 'green', 'red', 'yellow', 'pink', 'purple', 'orange', 'black', 'gray', 'white', 'brown'])
                   
my_results = {}

total_correct = 0
total_sentences = 0

keyword_exists = True
mapping_exists = True

for ref_id in tqdm(my_train_ref_ids):
    
    if ref_id not in my_refer_data_all_sentences['boxes']:
        continue
    
    x,y,w,h = my_refer_data_all_sentences['boxes'][ref_id]
    x1,y1,x2,y2 = x,y,x+w,y+h
    
    image = scipy.misc.imread(my_refer_data_all_sentences['image_file_names'][ref_id])
    if len(image.shape) != 3:
        continue
        
    height, width, _ = image.shape
    
    # Run detection
    #results = model.detect([image], verbose=1)
    
    with open('train_ref_detections/' + str(ref_id) + '.pickle', 'rb') as file:
        results = pickle.load(file)
    
    # Visualize results
    #r = results[0]
#     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                                 class_names, r['scores'])
    
    
    class_ids = results['class_ids']
    boxes = results['boxes']
    if len(boxes) == 0:
        continue
        
    current_class_names = [class_names[id] for id in class_ids]
#     print(current_class_names)
    
    sentence_results = []
    
    for sentence_info in my_refer_data_all_sentences['sentences'][ref_id]:
        
        refer_subject = sentence_info['subject']
        
        if mapping_exists:
            if refer_subject not in subject_labels_sorted:
                continue
        else:
            if refer_subject in subject_labels_sorted:
                continue

        found_label = None
        
        if mapping_exists:
            current_subject_labels = subject_labels_sorted[refer_subject]
            for subject_label in current_subject_labels:
                if subject_label[0] in current_class_names:
                    found_label = subject_label[0]
                    break

        tokens = sentence_info['sentence']['sent'].split(' ')
        current_filter_keywords = list(set(tokens) & filter_keywords)
        keyword = current_filter_keywords[0] if len(current_filter_keywords) > 0 else None
        if keyword_exists:
            if len(current_filter_keywords) == 0:
                continue
        else:
            if len(current_filter_keywords) > 0:
                continue
            
        total_sentences += 1
        
        best_i = -1
        best_box = None
        for i, class_name in enumerate(current_class_names):
            if class_name == found_label or not mapping_exists:
                box = boxes[i]
                bx1,by1,bx2,by2 = box[1],box[0],box[3],box[2]

                if best_i == -1:
                    best_i = i
                    best_box = bx1,by1,bx2,by2
                elif keyword == 'left':
                    if (bx1+bx2)/2 < (best_box[0]+best_box[2])/2:
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'right':
                    if (bx1+bx2)/2 > (best_box[0]+best_box[2])/2:
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'bottom':
                    if (by1+by2)/2 > (best_box[1]+best_box[3])/2:
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'top':
                    if (by1+by2)/2 < (best_box[1]+best_box[3])/2:
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'middle':
                    if abs((bx1+bx2)/2-width/2) < abs((best_box[0]+best_box[2])/2-width/2):
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'big' or keyword == 'huge':
                    if (bx2-bx1)*(by2-by1)>(best_box[2]-best_box[0])*(best_box[3]-best_box[1]):
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'small' or keyword == 'tiny':
                    if (bx2-bx1)*(by2-by1)<(best_box[2]-best_box[0])*(best_box[3]-best_box[1]):
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'long' or keyword == 'tall':
                    if (by2-by1)>(best_box[3]-best_box[1]):
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword == 'short':
                    if (by2-by1)<(best_box[3]-best_box[1]):
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                elif keyword is not None: #color
#                     original_mask = results['masks'][:,:,1]
#                     mask = np.zeros(image.shape, dtype=np.uint8)
#                     for i in range(mask.shape[0]):
#                         for j in range(mask.shape[1]):
#                             mask[i,j] = original_mask[i,j]

#                     masked_image = np.multiply(mask,image)
#                     #show_image(masked_image)
#                     recognized_colors = dominant_colors(masked_image, 5)[0]
#                     for recognized_color in recognized_colors:

#                         if keyword == describer.describe(recognized_color):
#                             best_i = i
#                             best_box = bx1,by1,bx2,by2
#                             break
                    with open('ref_boxes_colors/colors_ref_id_' + str(ref_id) + '.pickle', 'rb') as file:
                        ref_boxes_colors = pickle.load(file)
                    if keyword in ref_boxes_colors[i]:
                        best_i = i
                        best_box = bx1,by1,bx2,by2
                    
                
#     print('subject:', my_refer_data['subjects'][ref_id])
    
    
#     cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
#     cv2.rectangle(image,(best_box[0],best_box[1]),(best_box[2],best_box[3]),(255,0,0),3)
    
#     print('sentence:', my_refer_data['sentences'][ref_id])
#     show_image(image)
        if (best_box):
            correct = bb_intersection_over_union((x1,y1,x2,y2),best_box) >= 0.5
            if correct:
                total_correct += 1
            sentence_results.append({'expr': sentence_info['sentence']['sent'], 'predicted_xyxy_box': best_box, 'correct': correct})
    my_results[ref_id] = {'gold_xyxy_box': (x1,y1,x2,y2), 'sentence_results': sentence_results}
    
print(100*total_correct/total_sentences)
