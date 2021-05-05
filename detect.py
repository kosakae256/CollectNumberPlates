import tensorflow as tf
import glob
import copy
import cv2
import numpy as np
import json

model_path = 'C:/Users/kosakae256/OneDrive/Docments/CollectNumbers/model/saved_model'
DEFAULT_FUNCTION_KEY = 'serving_default'
loaded_model = tf.saved_model.load(model_path)
inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

# 推論用関数(Function for inference)
def run_inference_single_image(image, inference_func):
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output

def detection_number(readpath,detectpath,formpath,jsonpath,filename):
    image = cv2.imread(readpath)
    debug_image = copy.deepcopy(image)

    image_width, image_height = image.shape[1], image.shape[0]
    image = image[:, :, [2, 1, 0]]  # BGR2RGB
    image_np_expanded = np.expand_dims(image, axis=0)

    output = run_inference_single_image(image_np_expanded, inference_func)

    c = 0
    num_detections = output['num_detections']

    detectdict = {
            "bbox" : [],
            "filename" : filename
            }

    for i in range(num_detections):
        score = output['detection_scores'][i]
        bbox = output['detection_boxes'][i]

        #この数値以上の信頼度なら通る
        if score < 0.4:
            continue

        x1, y1 = int(bbox[1] * image_width), int(bbox[0] * image_height)
        x2, y2 = int(bbox[3] * image_width), int(bbox[2] * image_height)
        detectdict["bbox"].append([x1,y1,x2,y2])

        # 推論結果描画(Inference result drawing)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(debug_image, str('{:.2f}'.format(score)), (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        c+=1

    #どこにも判定がなければ
    if c!=0:
        cv2.imwrite(f"{detectpath}/detect_{filename}",debug_image)
        cv2.imwrite(f"{formpath}/form_{filename}",image)
        with open(f'{jsonpath}/{filename}.json', 'w') as f: # 第二引数：writableオプションを指定
            json.dump(detectdict, f)
